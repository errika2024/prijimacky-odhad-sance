# =============================================================================
# app.py – Streamlit aplikace: odhad šance na přijetí na SŠ (model V5)
#
# Spuštění:  streamlit run V5/github/app.py
# Potřebuje: V5/vystup/ (model + metadata), V5/data/ (příznaky škol, převod bodů
#            na percentil, názvy a adresy škol).
#
# Jak to počítá:
#   1) Body žáka -> percentil v jeho kohortě (9., 7. nebo 5. třída) podle roku 2026.
#   2) Percentil + historie školy -> CatBoost -> šance na přijetí (bodový odhad).
#   3) Rozmezí šance: stejný výpočet pro pořadí žáka posunuté o typické kolísání
#      laťky přijetí školy (viz 02_model.py, oddíl 4).
#   4) Když má žák body jasně nad laťkou (nad laťkou všech let 2024–2026, nad horním
#      odhadem laťky 2027 a dolní mez šance je alespoň 80 %), ukáže se místo rozmezí
#      „šance minimálně X %“. 100 % aplikace neukazuje nikdy.
#   Laťka přijetí = body, nad kterými bylo 9 z 10 přijatých (10. percentil přijatých).
# =============================================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostClassifier
from scipy.stats import norm

# ---------------- nastavení ----------------
V5_DIR = Path(__file__).resolve().parents[1]                 # složka V5
PROD = V5_DIR / "vystup"
SKOLY_CSV = V5_DIR / "data" / "skoly_2027_nazvy.csv"               # názvy, adresy, body pro texty
SKOLY_MODEL = V5_DIR / "data" / "skoly_2027.parquet"       # příznaky modelu
PERC_CSV = V5_DIR / "data" / "tabulky_percentilu_2026.csv"

KOHORTA_TYPU = {"GY8": 5, "GY6": 7}          # ostatní typy škol = 9. třída
MAX_BODU = 50                                 # maximum bodů z jednoho předmětu
K = 100 / MAX_BODU                            # body -> % skóre (model pracuje s % skóre)
CIL = 0.90                                    # cílová šance pro „kolik bodů chybí“
MIN_DOLNI = 0.80                              # „minimálně X %“ jen když i dolní mez je alespoň 80 %
PRIORITA = 1                                  # šance pro školu na 1. místě přihlášky
MEDIAN_ZAKU = {2024: 110.0, 2025: 96.0, 2026: 102.0}   # medián ČJ+M (% skóre, 0–200)

KRAJE = {
    "CZ010": "Hlavní město Praha", "CZ020": "Středočeský kraj", "CZ031": "Jihočeský kraj",
    "CZ032": "Plzeňský kraj", "CZ041": "Karlovarský kraj", "CZ042": "Ústecký kraj",
    "CZ051": "Liberecký kraj", "CZ052": "Královéhradecký kraj", "CZ053": "Pardubický kraj",
    "CZ063": "Kraj Vysočina", "CZ064": "Jihomoravský kraj", "CZ071": "Olomoucký kraj",
    "CZ072": "Zlínský kraj", "CZ080": "Moravskoslezský kraj",
}
TYPY = {
    "GY4": "Čtyřleté gymnázium", "GY6": "Šestileté gymnázium", "GY8": "Osmileté gymnázium",
    "LYC": "Lyceum", "SOS": "Střední odborná škola", "SOU": "Střední odborné učiliště",
    "SOU_VL": "Učební obory s výučním listem", "SOUbezVL": "Učební obory bez výučního listu",
    "KON": "Konzervatoř", "NAS": "Nástavbové studium",
}
PORADI_TYPU = list(TYPY)
_AB = "aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž"


def cz_key(s):
    """Klíč pro české řazení (Č za C, CH za H)."""
    s, out, i = str(s).lower(), [], 0
    while i < len(s):
        if s[i:i + 2] == "ch":
            out.append(_AB.find("h") + 0.5)
            i += 2
        else:
            out.append(_AB.find(s[i]) + 1 if s[i] in _AB else 0)
            i += 1
    return out


st.set_page_config(page_title="Přijímačky: odhad šance", layout="centered")


# ---------------- načtení (jen jednou za běh serveru) ----------------
@st.cache_resource
def nacti():
    meta = json.load(open(PROD / "metadata.json", encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(str(PROD / "produkce_2025_2026.cbm"))
    perc = pd.read_csv(PERC_CSV)
    zobraz = pd.read_csv(SKOLY_CSV, sep=";", dtype={"REDIZO": str, "KKOV": str, "ZRIZOVATEL_KOD": str,
                                                    "KRAJ_KOD": str, "TYP_SKOLY": str})
    priznaky = pd.read_parquet(SKOLY_MODEL)
    klic = ["REDIZO", "KKOV", "ZRIZOVATEL_KOD"]
    nove = [c for c in priznaky.columns if c not in zobraz.columns or c in klic]
    return meta, model, perc, zobraz.merge(priznaky[nove], on=klic, how="left")


meta, model, perc, skoly = nacti()
FEATURE, KAT = meta["feature"], meta["kategorie"]
DELTA = meta["rozmezi"]["delta_globalni"]


# ---------------- výpočty ----------------
def z(p):
    """Percentil (0–100) -> z-skóre."""
    return norm.ppf(np.clip(np.asarray(p, float) / 100, 5e-4, 1 - 5e-4))


def z_inv(x):
    return 100 * norm.cdf(x)


def tabulka(kohorta, predmet):
    return perc[(perc["KOHORTA"] == kohorta) & (perc["PREDMET"] == predmet)]


def na_percentil(skore, kohorta, predmet):
    """% skóre -> percentil v kohortě podle rozložení roku 2026."""
    t = tabulka(kohorta, predmet)
    return np.interp(skore, t["SKOR"], t["PERCENTIL"])


def kohorty(df):
    return df["TYP_SKOLY"].map(lambda t: KOHORTA_TYPU.get(t, 9)).values


def delta_skoly(df):
    """Šířka rozmezí: větší z běžného kolísání a kolísání této školy (2025 vs. 2026)."""
    vlastni = np.abs(z(df["PERC_P10_LAG1"]) - z(df["PERC_P10_LAG2"]))
    return np.fmax(DELTA, np.nan_to_num(vlastni, nan=0.0))


def sestav(df, cj, ma, posun=0.0):
    """Doplní příznaky žáka k tabulce škol. cj, ma = % skóre; posun = posun pořadí v z-skóre."""
    df = df.copy()
    koh = kohorty(df)
    cj = np.broadcast_to(np.asarray(cj, float), koh.shape)
    ma = np.broadcast_to(np.asarray(ma, float), koh.shape)
    for c, (skore, predmet) in {"PERC": (cj + ma, "SKOR_CJ_MA"), "PERC_CJ": (cj, "SKOR_CJ"),
                                "PERC_MA": (ma, "SKOR_MA")}.items():
        p = np.zeros(len(df))
        for k in np.unique(koh):
            m = koh == k
            p[m] = na_percentil(skore[m], k, predmet)
        df[c] = z_inv(z(p) + posun) if np.any(posun) else p
    df["PRIORITA"] = PRIORITA
    for c in ["PERC_MIN", "PERC_P10", "PERC_MEDIAN"]:
        for s in ["LAG1", "AVG"]:
            df[f"ROZDIL_{c}_{s}"] = df["PERC"] - df[f"{c}_{s}"]
    return df


def predikuj(df):
    x = df.reindex(columns=FEATURE).copy()
    for c in FEATURE:
        x[c] = x[c].fillna("NA").astype(str) if c in KAT else pd.to_numeric(x[c], errors="coerce")
    return model.predict_proba(x)[:, 1]


def sance_s_rozmezim(df, cj_b, ma_b):
    """Vrací (šance, dolní mez, horní mez) pro každý řádek tabulky škol."""
    d = delta_skoly(df)
    stred = predikuj(sestav(df, cj_b * K, ma_b * K))
    dolni = predikuj(sestav(df, cj_b * K, ma_b * K, -d))
    horni = predikuj(sestav(df, cj_b * K, ma_b * K, d))
    return stred, np.minimum(dolni, stred), np.maximum(horni, stred)


def latka_v_bodech(df, delta):
    """Laťka přijetí v bodech (ČJ + M, max 100) pro každou školu v tabulce:
    loňská hodnota a kam se může posunout (dolní a horní odhad pro rok 2027).

    Posun se počítá v místě laťky školy (ne v místě bodů žáka), takže je pro
    školu jedno číslo bez ohledu na to, kolik bodů uživatel zadá."""
    koh = kohorty(df)
    p10 = pd.to_numeric(df["PERC_P10_LAG1"], errors="coerce").values
    delta = np.broadcast_to(np.asarray(delta, float), p10.shape)
    latka, dolu, nahoru = (np.full(len(df), np.nan) for _ in range(3))
    for k in np.unique(koh):
        t = tabulka(k, "SKOR_CJ_MA")
        m = koh == k
        na_body = lambda perc: np.interp(perc, t["PERCENTIL"], t["SKOR"]) / K
        latka[m] = na_body(p10[m])
        dolu[m] = na_body(z_inv(z(p10[m]) - delta[m]))
        nahoru[m] = na_body(z_inv(z(p10[m]) + delta[m]))
    nezname = np.isnan(p10)
    latka[nezname], dolu[nezname], nahoru[nezname] = np.nan, np.nan, np.nan
    return latka, dolu, nahoru


def jasne_nad_latkou(df, soucet_b, delta):
    """True, když má žák víc bodů než laťka KAŽDÉHO známého roku (2024–2026)
    i než horní odhad laťky pro rok 2027. Pak má smysl jen dolní mez šance."""
    minule = (df[["SKOR_P10_2024", "SKOR_P10_2025", "SKOR_P10_2026"]]
              .apply(pd.to_numeric, errors="coerce") * MAX_BODU / 100)
    nejvyssi = minule.max(axis=1).values            # nejvyšší laťka z minulých let
    _, _, nahoru = latka_v_bodech(df, delta)
    return ~np.isnan(nejvyssi) & ~np.isnan(nahoru) & (soucet_b > nejvyssi) & (soucet_b > nahoru)


def pridej(cj_b, ma_b, t):
    """Přidá t bodů střídavě do ČJ a M (dokud je kam)."""
    for k in range(t):
        if k % 2 == 0:                          # sudý krok: ČJ (je-li plná, tak M)
            if cj_b < MAX_BODU: cj_b += 1
            else: ma_b += 1
        else:                                   # lichý krok: M (je-li plná, tak ČJ)
            if ma_b < MAX_BODU: ma_b += 1
            else: cj_b += 1
    return cj_b, ma_b


def kolik_bodu(skola, cj_b, ma_b, cil=CIL):
    """Nejmenší přídavek bodů, se kterým šance dosáhne cíle (jedno dávkové volání modelu).
    Vrací (body navíc, ČJ, M, šance), nebo (None, MAX, MAX, nejvyšší šance)."""
    volno = 2 * MAX_BODU - cj_b - ma_b
    kroky = [pridej(cj_b, ma_b, t) for t in range(1, volno + 1)]
    if not kroky:
        return None, MAX_BODU, MAX_BODU, float(predikuj(sestav(skola.to_frame().T, cj_b * K, ma_b * K))[0])
    cj2, ma2 = np.array(kroky).T
    tab = pd.DataFrame([skola] * len(kroky)).reset_index(drop=True)
    p = predikuj(sestav(tab, cj2 * K, ma2 * K))
    ok = np.flatnonzero(p >= cil)        # model je monotónní, první zásah je ten pravý
    if len(ok):
        return ok[0] + 1, int(cj2[ok[0]]), int(ma2[ok[0]]), float(p[ok[0]])
    return None, MAX_BODU, MAX_BODU, float(p[-1])


def body(n):
    n = int(n)
    return "1 bod" if n == 1 else f"{n} body" if 2 <= n <= 4 else f"{n} bodů"


def veta_historie(rok, prumer, p10, minimum, tvoje):
    if pd.isna(prumer) or pd.isna(p10):
        return f"V roce {rok} nejsou pro tento obor známé body přijatých."
    b = lambda x: x * MAX_BODU / 100
    veta = (f"V roce {rok}: přijatí měli průměrně {b(prumer):.0f} bodů, "
            f"laťka přijetí {b(p10):.0f} bodů")
    if pd.notna(minimum) and (b(p10) - b(minimum)) >= 3:
        veta += f" (nejnižší přijatý měl {b(minimum):.0f} bodů, což bývá výjimka, např. odvolání)"
    return veta + f". Tvoje body: {tvoje}."


# ---------------- rozhraní ----------------
st.title("Odhad šance na přijetí (rok 2027)")
st.caption("Model je trénován na datech CERMAT 2025 a 2026; u škol se ukazuje i historie 2024. "
           "Jde o odhad, ne záruku.")
st.caption("Model pracuje s percentilem (pořadím mezi uchazeči), body převádí podle rozložení "
           "roku 2026. Předpokládá tedy, že test 2027 bude stejně obtížný jako v roce 2026.")
st.caption("Obory pouze s talentovou zkouškou (např. umělecké obory) v nabídce nejsou, "
           "protože o jejich přijetí nerozhodují jen body z testů.")
with st.expander("Jak těžké byly přijímací testy v minulých letech?"):
    body_roku = {r: m * MAX_BODU / 100 for r, m in MEDIAN_ZAKU.items()}
    prumer = sum(body_roku.values()) / len(body_roku)
    for rok, b in body_roku.items():
        popis = "lehčí než průměr" if b - prumer >= 3 else "těžší než průměr" if prumer - b >= 3 else "průměrný"
        st.write(f"**{rok}:** medián uchazečů {b:.0f} bodů ze {2 * MAX_BODU} (součet ČJ + matematika), "
                 f"test byl {popis}.")
    st.caption("Čím nižší medián, tím těžší test (nebo slabší ročník). Body z různých let proto "
               "nejsou přímo srovnatelné. Model s tím počítá a pro rok 2027 předpokládá stejnou "
               "obtížnost jako v roce 2026.")
with st.expander("Co je laťka přijetí a rozmezí šance?"):
    st.write("**Laťka přijetí** = kolik bodů (ČJ + matematika, max 100) na škole loni stačilo "
             "na přijetí. Přesněji: počet bodů, nad kterým bylo 9 z 10 přijatých. Nejslabšího "
             "přijatého nebereme, protože to bývá výjimka (např. přijetí po odvolání).")
    st.write("Laťka se mezi roky posouvá podle toho, kolik silných uchazečů se na školu zrovna "
             "přihlásí. **Rozmezí šance** ukazuje tvoji šanci pro případ, že se laťka posune "
             "nahoru nebo dolů tak, jak se to v letech 2024–2026 stalo u 8 z 10 oborů. Kde laťka "
             "kolísala víc, je rozmezí širší. Když máš body daleko od laťky (hodně nad ní nebo "
             "hodně pod ní), posun laťky na tvou šanci skoro nemá vliv a rozmezí je úzké. "
             "Když máš víc bodů, než byla laťka ve všech letech 2024–2026 i než její horní "
             "odhad pro rok 2027 (a šance ani v horším případě neklesne pod 80 %), ukážeme jen „šanci minimálně X %“. 100 % nepíšeme nikdy: "
             "rozhoduje i pořadí přihlášek, konkurence se mění a škola může mít další kritéria.")
    st.caption(f"Ověření: na datech roku 2026 padla skutečná úspěšnost oboru do rozmezí u "
               f"{meta['rozmezi']['pokryti_2026'] * 100:.0f} % oborů.")

rezim = st.radio("Co chceš zjistit?", ["Šance na konkrétní škole", "Přehled škol v kraji a typu"],
                 horizontal=True)

kraje = sorted(skoly["KRAJ_KOD"].unique(), key=lambda k: KRAJE.get(k, k))
kraj = st.selectbox("Kraj", kraje, format_func=lambda k: KRAJE.get(k, k))
v_kraji = skoly[skoly["KRAJ_KOD"] == kraj]
okres = st.selectbox("Okres", ["Všechny okresy"] + sorted(v_kraji["OKRES_NAZEV"].unique()))
if okres != "Všechny okresy":
    v_kraji = v_kraji[v_kraji["OKRES_NAZEV"] == okres]
typy = sorted(v_kraji["TYP_SKOLY"].unique(),
              key=lambda t: PORADI_TYPU.index(t) if t in PORADI_TYPU else 99)
typ = st.selectbox("Typ školy", typy, format_func=lambda t: TYPY.get(t, t))

nabidka = v_kraji[v_kraji["TYP_SKOLY"] == typ].copy()
nabidka["popis"] = (nabidka["NAZEV_SKOLY"] + " – " + nabidka["NAZEV_OBORU"] + " ("
                    + nabidka["OBEC"] + ", " + nabidka["ULICE"].fillna("") + ", "
                    + nabidka["KKOV"] + ", REDIZO " + nabidka["REDIZO"] + ")")

# ---------- režim 1: přehled všech škol v kraji a typu ----------
if rezim == "Přehled škol v kraji a typu":
    st.subheader("Přehled škol v kraji a typu")
    c1, c2 = st.columns(2)
    cj_p = c1.number_input(f"ČJ (body, max {MAX_BODU})", 0, MAX_BODU, 30, 1, key="cj_p")
    ma_p = c2.number_input(f"Matematika (body, max {MAX_BODU})", 0, MAX_BODU, 30, 1, key="ma_p")

    if st.button("Spočítat přehled", type="primary"):
        if cj_p + ma_p == 0:
            st.warning("S 0 body škola uchazeče nejspíš nepřijme. Zadej skutečné body.")
            st.stop()
        d = nabidka.copy()
        bez_dat = d["PERC_P10_LAG1"].isna() & d["PERC_MEDIAN_LAG1"].isna()
        if bez_dat.any():
            st.caption(f"{int(bez_dat.sum())} oborů bez známých bodů přijatých není v přehledu.")
        d = d[~bez_dat].reset_index(drop=True)
        if d.empty:
            st.info("Pro tuto kombinaci nejsou dostupná data.")
            st.stop()

        stred, dolni, horni = sance_s_rozmezim(d, cj_p, ma_p)

        def poznamka(r):
            p = []
            if r["JE_TALENTOVKA"] == 1: p.append("talentová zkouška")
            if pd.notna(r["PRIJATI_LAG1"]) and r["PRIJATI_LAG1"] < 10: p.append("málo přijatých")
            if r["MA_LAG2"] == 0: p.append("jen 1 rok historie")
            return ", ".join(p)

        tab = pd.DataFrame({
            "Škola": d["NAZEV_SKOLY"], "Obor": d["NAZEV_OBORU"],
            "Obec": d["OBEC"], "Ulice": d["ULICE"].fillna(""),
            "Šance (%)": np.minimum(stred * 100, 99).round(0),   # 100 % nikdy
            "Rozmezí (%)": [f"minimálně {min(a * 100, 99):.0f}" if n else f"{a * 100:.0f}–{min(b * 100, 99):.0f}"
                            for a, b, n in zip(dolni, horni,
                                               jasne_nad_latkou(d, cj_p + ma_p, delta_skoly(d))
                                               & (dolni >= MIN_DOLNI))],
            "Loni průměr přijatých": (d["SKOR_PRUMER_LAG1"] * MAX_BODU / 100).round(0),
            "Loni laťka přijetí": (d["SKOR_P10_2026"] * MAX_BODU / 100).round(0),
            "Poznámka": d.apply(poznamka, axis=1),
            "REDIZO": d["REDIZO"],
        }).sort_values("Šance (%)", ascending=False)

        m1, m2, m3 = st.columns(3)
        m1.metric("Počet škol", len(tab))
        m2.metric("Medián šance", f"{tab['Šance (%)'].median():.0f} %")
        m3.metric("Šance ≥ 70 %", int((tab["Šance (%)"] >= 70).sum()))
        st.dataframe(tab, hide_index=True, use_container_width=True,
                     column_config={"Šance (%)": st.column_config.ProgressColumn(
                         "Šance (%)", min_value=0, max_value=100, format="%.0f %%")})
        st.caption("Laťka přijetí = body, nad kterými bylo loni 9 z 10 přijatých. Rozmezí = šance, "
                   "kdyby se laťka v roce 2027 posunula nahoru nebo dolů tak, jak se mezi roky běžně posouvá. "
                   "„Minimálně X“ = body jsou nad laťkou všech minulých let; jistota to ale není.")
        if typ in ("GY6", "GY8"):
            st.warning("Šestileté a osmileté gymnázium: výsledek ber jen jako hrubou orientaci.")
        st.warning("U škol s talentovou nebo školní zkouškou a dalšími kritérii může být odhad "
                   "zkreslený. Podmínky přijetí vždy ověř na webu konkrétní školy.")
    st.stop()

# ---------- režim 2: jedna škola ----------
poradi = sorted(nabidka.index, key=lambda i: cz_key(nabidka.at[i, "popis"]))
idx = st.selectbox(f"Škola a obor ({len(nabidka)} možností)", poradi,
                   format_func=lambda i: nabidka.at[i, "popis"])
skola = nabidka.loc[idx]

c1, c2 = st.columns(2)
cj_b = c1.number_input(f"ČJ (body, max {MAX_BODU})", 0, MAX_BODU, 30, 1)
ma_b = c2.number_input(f"Matematika (body, max {MAX_BODU})", 0, MAX_BODU, 30, 1)

if st.button("Spočítat", type="primary"):
    tvoje = cj_b + ma_b
    if tvoje == 0:
        st.warning("S 0 body škola uchazeče nejspíš nepřijme (0 bývá i neúčast u zkoušky). "
                   "Zadej skutečné body.")
        st.stop()
    if pd.isna(skola.get("PERC_P10_LAG1")) and pd.isna(skola.get("PERC_MEDIAN_LAG1")):
        st.error("Pro tento obor nejsou známé body přijatých z předchozích let (CERMAT je u "
                 "škol s málo přijatými nezveřejňuje). Odhad proto nelze spolehlivě spočítat.")
        st.stop()

    jedna = skola.to_frame().T.reset_index(drop=True)
    p, lo, hi = (float(v[0]) for v in sance_s_rozmezim(jedna, cj_b, ma_b))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=min(p * 100, 99), number={"suffix": " %", "valueformat": ".0f"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f3b73"},
               "steps": [{"range": [0, 40], "color": "#f4c7c3"},
                         {"range": [40, 70], "color": "#fce8b2"},
                         {"range": [70, 100], "color": "#b7e1cd"},
                         {"range": [lo * 100, hi * 100], "color": "rgba(60,60,60,0.35)",
                          "thickness": 0.35}]}))
    fig.update_layout(height=280, margin=dict(t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    d = float(delta_skoly(jedna)[0])
    kolisava = d > DELTA + 1e-9
    uzke = (hi - lo) < 0.05                     # rozmezí užší než 5 procentních bodů
    nad = bool(jasne_nad_latkou(jedna, tvoje, d)[0]) and lo >= MIN_DOLNI
    if nad:
        st.markdown(f"**Šance minimálně {min(lo * 100, 99):.0f} %.** Tvoje body ({tvoje}) jsou nad laťkou "
                    f"přijetí všech let 2024–2026 i nad jejím horním odhadem pro rok 2027.")
        st.caption("Proč ne 100 %: o přijetí rozhoduje i pořadí přihlášek a obsazenost míst, "
                   "konkurence se každý rok mění a škola může mít další kritéria (prospěch, "
                   "školní zkouška). Proto ukazujeme jen to, jak nízko může šance klesnout.")
    else:
        st.markdown(f"**Rozmezí šance: {lo * 100:.0f}–{min(hi * 100, 99):.0f} %** (šedý pruh).")
    if pd.notna(skola.get("PERC_P10_LAG1")):
        latka, l_dolu, l_nahoru = (float(v[0]) for v in latka_v_bodech(jedna, d))
        st.markdown(f"V roce 2026 byla na této škole **laťka přijetí {latka:.0f} bodů** (body, nad kterými "
                    f"bylo 9 z 10 přijatých). V roce 2027 může vyjít zhruba mezi {l_dolu:.0f} a "
                    f"{l_nahoru:.0f} body" + (", protože u této školy kolísá víc než obvykle"
                                             if kolisava else "")
                    + ("." if nad else ". Rozmezí šance ukazuje, jak by ti to vyšlo na obou koncích."))
    if uzke and not nad:
        kde = "vysoko nad laťkou" if p >= 0.5 else "hluboko pod laťkou"
        st.caption(f"Rozmezí je úzké, protože tvoje body ({tvoje}) jsou {kde}: posun laťky "
                   f"tvou šanci skoro nezmění.")
    elif kolisava and not uzke and not nad:
        st.warning("Laťka této školy se mezi lety 2025 a 2026 posunula víc než u většiny škol "
                   "a tvoje body jsou blízko ní. Ber bodový odhad s rezervou.")

    if p >= CIL:
        st.success(f"Šance je již nad {int(CIL * 100)} %.")
    else:
        t, cj2, ma2, p2 = kolik_bodu(skola, cj_b, ma_b)
        if t is not None:
            st.info(f"Pro šanci {int(CIL * 100)} % ti chybí zhruba **{body(t)} celkem** "
                    f"(např. ČJ {cj2} a matematika {ma2}). Model bere v úvahu součet i oba "
                    f"předměty zvlášť, takže jiné rozdělení bodů může dát trochu jinou šanci.")
        else:
            st.info(f"Model u této školy ani při plném počtu bodů nepřekročí {p2 * 100:.0f} %. "
                    f"Jde o opatrný odhad: u nejnáročnějších škol je skutečná šance při "
                    f"velmi vysokém skóre obvykle vyšší.")

    for rok, sfx in ((2026, "LAG1"), (2025, "LAG2")):
        st.caption(veta_historie(rok, skola.get(f"SKOR_PRUMER_{sfx}"), skola.get(f"SKOR_P10_{rok}"),
                                 skola.get(f"SKOR_MIN_{sfx}"), tvoje))
    st.caption(veta_historie(2024, skola.get("SKOR_PRUMER_2024"), skola.get("SKOR_P10_2024"),
                             skola.get("SKOR_MIN_2024"), tvoje))

    pr, p10 = skola.get("SKOR_PRUMER_LAG1"), skola.get("SKOR_P10_2026")
    if pd.notna(pr) and pd.notna(p10):
        if tvoje > pr * MAX_BODU / 100:
            st.success("Tvé skóre je nad loňským průměrem přijatých na tuto školu.")
        elif tvoje >= p10 * MAX_BODU / 100:
            st.info("Tvé skóre je mezi loňskou laťkou přijetí a průměrem přijatých.")
        else:
            st.warning("Tvé skóre je pod loňskou laťkou přijetí.")

    if skola["TYP_SKOLY"] in ("GY6", "GY8"):
        st.warning("Šestileté a osmileté gymnázium: body se převádějí na pořadí mezi uchazeči "
                   "stejného ročníku (páťáci, resp. sedmáci). Dat je méně, ber výsledek jako orientaci.")
    if skola["JE_TALENTOVKA"] == 1:
        st.warning("Obor s talentovou zkouškou: rozhoduje i zkouška, kterou model nezná.")
    if skola["MA_LAG2"] == 0:
        st.warning("Škola má jen jeden rok historie, odhad je méně spolehlivý.")
    pocet = skola.get("PRIJATI_LAG1")
    if pd.notna(pocet) and pocet < 10:
        st.warning(f"Škola loni přijala jen {int(pocet)} uchazečů, takže statistiky přijatých "
                   f"jsou nespolehlivé a odhad je jen orientační.")
    st.warning("U škol s talentovou nebo školní zkouškou a dalšími kritérii může být odhad "
               "zkreslený. Podmínky přijetí vždy ověř na webu konkrétní školy.")
