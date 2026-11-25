# =============================================================================
# 01_priprava_dat.py – příprava dat pro model přijetí na SŠ (CERMAT, 1. kolo)
#
# Vstup:  DATA/PZ{rok}_kolo1_*.xlsx            (přihlášky žáků a tabulky škol 2024–2026)
# Výstup: V5/data/model_data.parquet            1 řádek = 1 přihláška (roky 2025, 2026)
#         V5/data/skoly_2027.parquet            příznaky škol pro predikci na rok 2027
#         V5/data/tabulky_percentilu_2026.csv   převod bodů na percentil (pro aplikaci)
#         V5/data/skoly_2027_nazvy.csv          názvy a adresy škol (kopie DATA/skoly_2027.csv)
#
# Hlavní myšlenky:
#   * Žáka popisuje PERCENTIL (pořadí mezi uchazeči stejného roku a ročníku),
#     ne body. Percentil nezávisí na tom, jak těžký byl test.
#   * Školu popisuje její HISTORIE v percentilech (loni = LAG1, předloni = LAG2),
#     takže model nikdy nevidí výsledky roku, který predikuje (žádný data leakage).
# =============================================================================
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")          # čeština ve Windows konzoli

# Cesty: složka V5 je o úroveň výš než tento soubor (V5/github nebo V5/podrobne).
# Surová data CERMAT leží v Prijimacky/DATA (lze přepsat proměnnou PRIJIMACKY_ROOT).
V5_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PRIJIMACKY_ROOT", V5_DIR.parent))
DATA_DIR = ROOT / "DATA"
OUT_DIR = V5_DIR / "data"
OUT_DIR.mkdir(exist_ok=True)

ROKY = [2024, 2025, 2026]
SOUBORY_UCHAZECI = {r: f"PZ{r}_kolo1_uchazeci_prihlasky_vysledky.xlsx" for r in ROKY}
SOUBORY_SKOLY = {
    2024: dict(soubor="PZ2024_kolo1_skoly_obory_vysledky.xlsx"),
    2025: dict(soubor="PZ2025_kolo1_vysledky_skola_obor.xlsx", sheet_name="ČR celkem", header=1),
    2026: dict(soubor="PZ2026_kolo1_skolobory_vysledky.xlsx"),
}
FORMY_DENNI = ["den", "den2"]
NEZKRACENE = 2
PRIORITY = [1, 2, 3, 4, 5]
KLIC = ["ROK", "REDIZO", "KKOV", "ZRIZOVATEL_KOD"]     # jeden obor jedné školy v jednom roce
MIN_PRIJATYCH = 5                                     # méně přijatých = nespolehlivé statistiky


# -----------------------------------------------------------------------------
# 1) Tabulka škol: sjednocení názvů sloupců mezi roky, jen obory s povinnou JPZ
# -----------------------------------------------------------------------------
PREJMENOVANI = {
    2024: {"ORP": "ORP_KOD", "ORP_NAZEV": "ORP", "ZRIZ": "ZRIZOVATEL_KOD",
           "ZRIZ_RED": "ZRIZOVATEL", "KAPACITA_1_KOLO": "KAPACITA"},
    2025: {},
    2026: {"KRAJ": "KRAJ_KOD", "KRAJ_NAZEV": "KRAJ", "ORP": "ORP_KOD", "ORP_NAZEV": "ORP",
           "ZRIZOVATEL": "ZRIZOVATEL_KOD", "ZRIZOVATEL_NAZEV": "ZRIZOVATEL",
           "FORMA_VZDELAVANI": "FORMA"},
}
SLOUPCE_SKOLY = ["REDIZO", "KKOV", "ZRIZOVATEL_KOD", "FORMA", "KRAJ_KOD",
                 "TYP_SKOLY", "ZKRACENE_STUDIUM", "KAPACITA", "POVINNOST_JPZ"]


def normalizuj_nazev(nazev):
    """'KRAJ - NÁZEV' -> 'KRAJ_NAZEV' (bez diakritiky, velká písmena)."""
    t = unicodedata.normalize("NFKD", str(nazev))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").upper()


def modus(s):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else np.nan


def nacti_skoly(rok):
    p = dict(SOUBORY_SKOLY[rok])
    df = pd.read_excel(DATA_DIR / p.pop("soubor"), **p)
    df.columns = [normalizuj_nazev(c) for c in df.columns]
    df = df.rename(columns=PREJMENOVANI[rok])[SLOUPCE_SKOLY]
    df = df[df["REDIZO"].notna()].copy()
    df["ROK"] = rok
    for c in ["KAPACITA", "POVINNOST_JPZ"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["REDIZO", "ZRIZOVATEL_KOD"]:
        df[c] = df[c].astype("Int64")
    # denní, nezkrácené obory s povinnou jednotnou přijímací zkouškou, bez nástaveb
    df = df[(df["POVINNOST_JPZ"] == 1) & df["FORMA"].isin(FORMY_DENNI)
            & (df["ZKRACENE_STUDIUM"] == "ne") & ~df["TYP_SKOLY"].isin(["NAS", "celkem"])]

    out = df.groupby(KLIC).agg(KAPACITA=("KAPACITA", "sum"), TYP_SKOLY=("TYP_SKOLY", modus),
                               KRAJ_KOD=("KRAJ_KOD", modus),
                               N_KRAJU=("KRAJ_KOD", "nunique")).reset_index()
    out.loc[out["N_KRAJU"] > 1, "KRAJ_KOD"] = np.nan     # pobočky ve více krajích
    out["KAPACITA"] = out["KAPACITA"].replace(0, np.nan)
    return out.drop(columns="N_KRAJU")


# -----------------------------------------------------------------------------
# 2) Přihlášky: široký formát (5 priorit v jednom řádku) -> dlouhý formát
# -----------------------------------------------------------------------------
def nacti_prihlasky(rok):
    df = pd.read_excel(DATA_DIR / SOUBORY_UCHAZECI[rok]).reset_index(drop=True)
    df["UCHAZEC_ID"] = [f"{rok}_{i}" for i in range(len(df))]

    # Kohorta: jen osmiletá gymnázia (K/81) = 5. třída, jen šestiletá (K/61) = 7. třída,
    # ostatní 9. třída. Každá kohorta píše jiný test, percentil se proto počítá zvlášť.
    kkov = df[[f"ss{p}_kkov" for p in PRIORITY]]
    konc = kkov.apply(lambda s: s.astype("string").str[-4:])
    vyplneno = kkov.notna()
    jen81 = ((konc == "K/81") | ~vyplneno).all(axis=1) & vyplneno.any(axis=1)
    jen61 = ((konc == "K/61") | ~vyplneno).all(axis=1) & vyplneno.any(axis=1)
    df["KOHORTA"] = np.select([jen81, jen61], [5, 7], default=9)

    d = pd.concat([pd.DataFrame({
        "UCHAZEC_ID": df["UCHAZEC_ID"], "ROK": rok, "PRIORITA": p, "KOHORTA": df["KOHORTA"],
        "REDIZO": df[f"ss{p}_redizo"], "KKOV": df[f"ss{p}_kkov"],
        "ZRIZOVATEL_KOD": df[f"ss{p}_zrizovatel"], "FORMA": df[f"ss{p}_forma"],
        "ZKRACENO": df[f"ss{p}_zkraceno"], "PRIJAT": df[f"ss{p}_prijat"],
        "DUVOD": df[f"ss{p}_duvod_neprijeti"],
        "SKOR_CJ_MA": df["c_m_procentni_skor"], "SKOR_CJ": df["c_procentni_skor"],
        "SKOR_MA": df["m_procentni_skor"],
    }) for p in PRIORITY], ignore_index=True)

    d = d[d["REDIZO"].notna()].copy()
    for c in ["REDIZO", "ZRIZOVATEL_KOD"]:
        d[c] = d[c].astype("Int64")
    d = d[d["FORMA"].isin(FORMY_DENNI) & (d["ZKRACENO"] == NEZKRACENE)]
    d = d[~d["KKOV"].astype(str).str.endswith(("/51", "/52"))]       # bez nástaveb
    d = d[d["SKOR_CJ_MA"].notna()].copy()                             # psal ČJ i M

    d["STAV"] = np.select(
        [d["PRIJAT"] == 1, d["DUVOD"] == "prijat_na_vyssi_prioritu",
         d["DUVOD"].isin(["vzdal_se_prijeti", "vzdal_se_prijeti_po_terminu"]),
         d["DUVOD"].isin(["pro_nedostacujici_kapacitu", "pro_nesplneni_podminek"])],
        ["PRIJAT", "PRIJAT_JINDE", "VZDAL_SE", "NEPRIJAT"], default="NEZNAMY")
    d = d[d["STAV"] != "NEZNAMY"]
    # Cíl: 1 = žák na škole uspěl (přijat, nebo přijat na vyšší prioritu / vzdal se)
    d["SPLNIL_KRITERIA"] = (d["STAV"] != "NEPRIJAT").astype(int)
    return d.drop(columns=["FORMA", "ZKRACENO", "PRIJAT", "DUVOD"])


def pridej_percentily(d):
    """Percentil žáka (0–100) v rámci roku a kohorty: součet ČJ+M i předměty zvlášť."""
    zaci = d.drop_duplicates("UCHAZEC_ID")[["UCHAZEC_ID", "ROK", "KOHORTA",
                                            "SKOR_CJ_MA", "SKOR_CJ", "SKOR_MA"]].copy()
    g = zaci.groupby(["ROK", "KOHORTA"])
    zaci["PERC"] = g["SKOR_CJ_MA"].rank(pct=True) * 100
    zaci["PERC_CJ"] = g["SKOR_CJ"].rank(pct=True) * 100
    zaci["PERC_MA"] = g["SKOR_MA"].rank(pct=True) * 100
    return d.merge(zaci[["UCHAZEC_ID", "PERC", "PERC_CJ", "PERC_MA"]], on="UCHAZEC_ID")


def tabulky_percentilu(d, rok):
    """Převodní tabulky body -> percentil pro aplikaci (podle kohorty a předmětu)."""
    zaci = d[d["ROK"] == rok].drop_duplicates("UCHAZEC_ID")
    casti = []
    for koh, z in zaci.groupby("KOHORTA"):
        for skor, perc in [("SKOR_CJ_MA", "PERC"), ("SKOR_CJ", "PERC_CJ"), ("SKOR_MA", "PERC_MA")]:
            t = z.groupby(skor)[perc].mean().reset_index()
            t.columns = ["SKOR", "PERCENTIL"]
            t["KOHORTA"], t["PREDMET"] = koh, skor
            casti.append(t)
    return pd.concat(casti, ignore_index=True)


# -----------------------------------------------------------------------------
# 3) Statistiky oboru za rok a jejich posun do dalších let (lagy)
# -----------------------------------------------------------------------------
LAG_SLOUPCE = ["POCET_CELKEM", "PRIJATI", "PREVIS", "PODIL_P1", "PODIL_P345",
               "PERC_MIN", "PERC_P10", "PERC_MEDIAN", "PERC_MAX"]
ZOBRAZENI = ["SKOR_PRUMER", "SKOR_P10", "SKOR_MIN"]      # body, jen pro texty v aplikaci


def statistiky_skoly(d):
    pocty = (d.pivot_table(index=KLIC, columns="PRIORITA", values="UCHAZEC_ID",
                           aggfunc="count", fill_value=0).reindex(columns=PRIORITY, fill_value=0))
    pocty.columns = [f"N_P{p}" for p in PRIORITY]
    pocty["POCET_CELKEM"] = pocty.sum(axis=1)

    perc = d[d["STAV"] == "PRIJAT"].groupby(KLIC).agg(
        PRIJATI=("UCHAZEC_ID", "size"),
        PERC_MIN=("PERC", "min"),
        PERC_P10=("PERC", lambda s: s.quantile(0.10)),
        PERC_MEDIAN=("PERC", "median"),
        PERC_MAX=("PERC", "max"),
        SKOR_PRUMER=("SKOR_CJ_MA", "mean"),
        SKOR_P10=("SKOR_CJ_MA", lambda s: s.quantile(0.10)),
        SKOR_MIN=("SKOR_CJ_MA", "min"),
    )
    s = pocty.join(perc)
    s["PRIJATI"] = s["PRIJATI"].fillna(0).astype(int)
    s.loc[s["PRIJATI"] < MIN_PRIJATYCH, [c for c in perc.columns if c != "PRIJATI"]] = np.nan
    s["PREVIS"] = s["POCET_CELKEM"] / s["PRIJATI"].replace(0, np.nan)
    s["PODIL_P1"] = 100 * s["N_P1"] / s["POCET_CELKEM"]
    s["PODIL_P345"] = 100 * (s["N_P3"] + s["N_P4"] + s["N_P5"]) / s["POCET_CELKEM"]
    return s.reset_index()


def lagy(stat, posun):
    """Statistiky roku X se připojí k roku X + posun (LAG1 = loni, LAG2 = předloni)."""
    s = stat[KLIC + LAG_SLOUPCE + ZOBRAZENI].copy()
    s["ROK"] += posun
    return s.rename(columns={c: f"{c}_LAG{posun}" for c in LAG_SLOUPCE + ZOBRAZENI})


def odvozene_priznaky(df):
    """Příznaky z historie školy (stejná funkce pro trénink i rok 2027)."""
    df = df.copy()
    df["MA_HISTORII"] = df["POCET_CELKEM_LAG1"].notna().astype(int)
    # Průměr dostupných let místo samostatného LAG2 (v roce 2025 LAG2 chybí vždy
    # a jeho chybění by modelu prozradilo rok).
    for c in ["PERC_MIN", "PERC_P10", "PERC_MEDIAN", "PREVIS"]:
        df[f"{c}_AVG"] = df[[f"{c}_LAG1", f"{c}_LAG2"]].mean(axis=1)
    return df


def rozdily(df):
    """Rozdíl percentilu žáka a historické hranice školy (kladný = žák je nad hranicí)."""
    df = df.copy()
    for c in ["PERC_MIN", "PERC_P10", "PERC_MEDIAN"]:
        for s in ["LAG1", "AVG"]:
            df[f"ROZDIL_{c}_{s}"] = df["PERC"] - df[f"{c}_{s}"]
    return df


def je_talentovka(kkov):
    k = kkov.astype(str)
    return (k.str.startswith("82-") | k.isin(["68-42-M/01"])
            | k.str.startswith(("79-42-K/", "79-43-K/"))).astype(int)


def jako_text(df):
    """Kategorie jako text (CatBoost je tak bere jako kategorie, ne čísla)."""
    for c in ["KKOV", "KRAJ_KOD", "TYP_SKOLY"]:
        df[c] = df[c].astype("string").fillna("NA")
    for c in ["ZRIZOVATEL_KOD", "REDIZO"]:
        df[c] = df[c].astype("string")
    return df


# -----------------------------------------------------------------------------
# 4) Hlavní běh
# -----------------------------------------------------------------------------
def main():
    prihlasky = pridej_percentily(pd.concat([nacti_prihlasky(r) for r in ROKY], ignore_index=True))
    skoly = pd.concat([nacti_skoly(r) for r in ROKY], ignore_index=True)
    stat = statistiky_skoly(prihlasky)
    print("přihlášek:", len(prihlasky))

    # Trénovací a testovací data: roky 2025 a 2026 (rok 2024 slouží jen jako historie)
    data = (prihlasky[prihlasky["ROK"] >= 2025]
            .merge(lagy(stat, 1), on=KLIC, how="left")
            .merge(lagy(stat, 2), on=KLIC, how="left")
            .merge(skoly, on=KLIC, how="inner"))           # jen obory s povinnou JPZ
    assert not data.duplicated(["UCHAZEC_ID", "PRIORITA"]).any()
    data["JE_TALENTOVKA"] = je_talentovka(data["KKOV"])
    data = jako_text(rozdily(odvozene_priznaky(data)))
    data.to_parquet(OUT_DIR / "model_data.parquet", index=False)
    print(data.groupby("ROK")["SPLNIL_KRITERIA"].agg(["size", "mean"]).round(3))

    # Školy pro rok 2027: LAG1 = 2026, LAG2 = 2025 (rozdíly se dopočítají v aplikaci)
    sk27 = skoly[skoly["ROK"] == 2026].assign(ROK=2027)
    sk27 = sk27.merge(lagy(stat, 1), on=KLIC, how="left").merge(lagy(stat, 2), on=KLIC, how="left")
    sk27["JE_TALENTOVKA"] = je_talentovka(sk27["KKOV"])
    jako_text(odvozene_priznaky(sk27)).to_parquet(OUT_DIR / "skoly_2027.parquet", index=False)

    tabulky_percentilu(prihlasky, 2026).to_csv(OUT_DIR / "tabulky_percentilu_2026.csv", index=False)
    # Názvy a adresy škol pro aplikaci (tabulka vytvořená dřívějším notebookem)
    shutil.copy(DATA_DIR / "skoly_2027.csv", OUT_DIR / "skoly_2027_nazvy.csv")
    print("uloženo do", OUT_DIR)


if __name__ == "__main__":
    main()
