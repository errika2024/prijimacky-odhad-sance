# =============================================================================
# 02_model.py – trénink, vyhodnocení a rozmezí šance (model V5)
#
# Vstup:  V5/data/model_data.parquet, V5/data/skoly_2027.parquet
# Výstup: V5/vystup/  (modely .cbm, metriky .csv, metadata.json)
#
# Validační protokol (časové dělení, žádný únik informací z budoucnosti):
#   1) Výběr varianty a počtu iterací jen na roce 2025: 3násobná křížová validace
#      GroupKFold po OBORECH ŠKOL (REDIZO + KKOV), predikce mimo fold (OOF).
#   2) Rok 2026 se použije jednou: model natrénovaný jen na 2025, bez kalibrace.
#   3) Rozmezí šance: šířka nastavená z dat do roku 2025, pokrytí ověřené na 2026.
#   4) Produkční model = stejné nastavení natrénované na 2025 + 2026.
# Hodnotí se hlavně priorita 1 (tu ukazuje aplikace a má čistý cíl).
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding="utf-8")          # čeština ve Windows konzoli

V5_DIR = Path(__file__).resolve().parents[1]        # složka V5
DATA = V5_DIR / "data"
VYSTUP = V5_DIR / "vystup"
VYSTUP.mkdir(parents=True, exist_ok=True)
SEED = 42
KLIC = ["REDIZO", "KKOV", "ZRIZOVATEL_KOD"]

# -----------------------------------------------------------------------------
# Příznaky: percentily žáka, historie školy v percentilech, rozdíly žák – škola.
# Body nepoužíváme (závisí na obtížnosti testu), LAG2 jen v průměru *_AVG.
# -----------------------------------------------------------------------------
KAT = ["KKOV", "KRAJ_KOD", "ZRIZOVATEL_KOD", "TYP_SKOLY"]
ZAK = ["PRIORITA", "PERC", "PERC_CJ", "PERC_MA"]
SKOLA = ["KAPACITA", "POCET_CELKEM_LAG1", "PRIJATI_LAG1", "PREVIS_LAG1", "PODIL_P1_LAG1",
         "PODIL_P345_LAG1", "PERC_MIN_LAG1", "PERC_P10_LAG1", "PERC_MEDIAN_LAG1", "PERC_MAX_LAG1",
         "PERC_MIN_AVG", "PERC_P10_AVG", "PERC_MEDIAN_AVG", "PREVIS_AVG",
         "MA_HISTORII", "JE_TALENTOVKA"]
HRANICE = ["PERC_MIN", "PERC_P10", "PERC_MEDIAN"]
ROZDIL = [f"ROZDIL_{c}_{s}" for c in HRANICE for s in ["LAG1", "AVG"]]
FEATURE = KAT + ZAK + SKOLA + ROZDIL

# Monotónní omezení: víc bodů (vyšší percentil) nesmí šanci snížit.
MONOTONNI = {c: 1 for c in ["PERC", "PERC_CJ", "PERC_MA"] + ROZDIL}
ZAKLAD = dict(loss_function="Logloss", learning_rate=0.08, l2_leaf_reg=10, depth=4,
              random_seed=SEED, verbose=0, thread_count=-1)
VARIANTY = {"A_bez_omezeni": False, "B_monotonni": True}     # hodnota = monotónní?

KVANTIL_ROZMEZI = 0.80          # rozmezí pokrývá posun hranice u 8 z 10 oborů
MIN_UCHAZECU = 30               # obory s menším počtem se do pokrytí nepočítají


def ece(y, p, kosu=10):
    """Expected Calibration Error: vážený průměr |predikce − skutečnost| v 10 pásmech."""
    kos = np.minimum((p * kosu).astype(int), kosu - 1)
    return sum((kos == k).mean() * abs(p[kos == k].mean() - y[kos == k].mean())
               for k in range(kosu) if (kos == k).any())


def metriky(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(AUC=roc_auc_score(y, p), Brier=brier_score_loss(y, p), LogLoss=log_loss(y, p),
                ECE=ece(y, p), predikce=p.mean(), skutecnost=y.mean())


def model(monotonni, iterace=3000):
    par = dict(ZAKLAD, iterations=iterace)
    if monotonni:
        par["monotone_constraints"] = MONOTONNI
    return CatBoostClassifier(**par, cat_features=KAT)


# --- pomocné funkce pro rozmezí šance -----------------------------------------
def z(perc):
    """Percentil (0–100) -> z-skóre normálního rozdělení (probit)."""
    return norm.ppf(np.clip(np.asarray(perc, float) / 100, 5e-4, 1 - 5e-4))


def kolisani(df):
    """O kolik se mezi LAG2 a LAG1 posunula hranice (P10 přijatých), v z-skóre."""
    return np.abs(z(df["PERC_P10_LAG1"]) - z(df["PERC_P10_LAG2"]))


def posun_zaka(df, delta):
    """Posune pořadí žáka o delta (v z-skóre) a přepočítá rozdíly vůči škole."""
    df = df.copy()
    for c in ["PERC", "PERC_CJ", "PERC_MA"]:
        df[c] = 100 * norm.cdf(z(df[c]) + delta)
    for c in HRANICE:
        for s in ["LAG1", "AVG"]:
            df[f"ROZDIL_{c}_{s}"] = df["PERC"] - df[f"{c}_{s}"]
    return df


# =============================================================================
# 1) Data
# =============================================================================
md = pd.read_parquet(DATA / "model_data.parquet")
trn = md[md["ROK"] == 2025].reset_index(drop=True)
tst = md[md["ROK"] == 2026].reset_index(drop=True)
del md
y_trn = trn["SPLNIL_KRITERIA"].values

# =============================================================================
# 2) Výběr varianty na OOF 2025 (GroupKFold po oborech škol)
# Proč ne po žácích: přihlášky jiných žáků na stejný obor ve stejném roce prozradí
# letošní hranici. Při dělení po oborech ji model musí odhadnout z historie,
# stejně jako při skutečné predikci na další rok.
# =============================================================================
skupina = trn["REDIZO"] + "_" + trn["KKOV"]
foldy = list(GroupKFold(n_splits=3).split(trn, groups=skupina))
p1 = (trn["PRIORITA"] == 1).values
vyber, iterace = [], {}
for nazev, mono in VARIANTY.items():
    oof, best = np.zeros(len(trn)), []
    for i_uc, i_val in foldy:
        m = model(mono)
        m.fit(trn.loc[i_uc, FEATURE], y_trn[i_uc],
              eval_set=(trn.loc[i_val, FEATURE], y_trn[i_val]), early_stopping_rounds=100)
        oof[i_val] = m.predict_proba(trn.loc[i_val, FEATURE])[:, 1]
        best.append(m.get_best_iteration())
    iterace[nazev] = int(np.median(best) * 1.1) + 1       # +10 %: finální model má víc dat
    vyber.append(dict(varianta=nazev, iterace=iterace[nazev], **metriky(y_trn[p1], oof[p1])))
    print(vyber[-1], flush=True)

vyber = pd.DataFrame(vyber).set_index("varianta")
vyber.round(4).to_csv(VYSTUP / "vyber_oof_2025.csv")
VITEZ = vyber["LogLoss"].idxmin()
MONO = VARIANTY[VITEZ]
print("Vítěz podle OOF LogLoss (priorita 1, 2025):", VITEZ)

# =============================================================================
# 3) Jediné vyhodnocení na 2026 (model natrénovaný jen na 2025)
# =============================================================================
val_model = model(MONO, iterace[VITEZ])
val_model.fit(trn[FEATURE], y_trn)
val_model.save_model(str(VYSTUP / "validacni_2025.cbm"))
tst["P_MODEL"] = val_model.predict_proba(tst[FEATURE])[:, 1]

# Benchmark: rozdíl percentilu žáka a loňské P10 přijatých + logistická regrese
b_trn = trn[(trn["PRIORITA"] == 1) & trn["ROZDIL_PERC_P10_LAG1"].notna()]
bench = LogisticRegression().fit(b_trn[["ROZDIL_PERC_P10_LAG1"]], b_trn["SPLNIL_KRITERIA"])
ma_hist = tst["ROZDIL_PERC_P10_LAG1"].notna().values
tst["P_BENCH"] = np.nan
tst.loc[ma_hist, "P_BENCH"] = bench.predict_proba(tst.loc[ma_hist, ["ROZDIL_PERC_P10_LAG1"]])[:, 1]

y = tst["SPLNIL_KRITERIA"].values
p1 = (tst["PRIORITA"] == 1).values
segmenty = {
    "vsechny_priority": np.ones(len(tst), bool),
    "priorita_1": p1,
    "priorita_1_konkurencni (previs>2)": p1 & (tst["PREVIS_LAG1"] > 2).values,
    "priorita_1_s_historii": p1 & ma_hist,
}
radky = []
for seg, m in segmenty.items():
    radky.append(dict(segment=seg, model="V5", n=int(m.sum()), **metriky(y[m], tst["P_MODEL"].values[m])))
    mb = m & ma_hist
    radky.append(dict(segment=seg, model="benchmark_P10", n=int(mb.sum()),
                      **metriky(y[mb], tst["P_BENCH"].values[mb])))
test = pd.DataFrame(radky)
test.round(4).to_csv(VYSTUP / "test_2026.csv", index=False)
print(test.round(4).to_string())

# Kalibrační tabulka (priorita 1): průměrná predikce vs. skutečnost v pásmech po 10 %
k = pd.DataFrame({"P": tst["P_MODEL"].values[p1], "y": y[p1]})
k["pasmo"] = pd.cut(k["P"], np.linspace(0, 1, 11), include_lowest=True)
kal = k.groupby("pasmo", observed=True).agg(n=("y", "size"), predikce=("P", "mean"),
                                            skutecnost=("y", "mean"))
kal.round(3).to_csv(VYSTUP / "kalibrace_2026_priorita1.csv")
print(kal.round(3))

# =============================================================================
# 4) Rozmezí šance
# Nejistotu nedělá hlavně model, ale to, že se hranice školy mezi roky posouvá.
# Posun měříme v z-skóre percentilu (u elitních škol je 1 percentil hodně bodů).
# Rozmezí = šance, kdyby byl žák v pořadí o delta níž / výš (tj. hranice o delta
# výš / níž), kde delta = max(běžné kolísání, kolísání této školy).
# Běžné kolísání = 80% kvantil posunu 2024 -> 2025: známe ho dřív, než začal rok 2026.
# =============================================================================
skoly_26 = tst.drop_duplicates(KLIC)
DELTA_VAL = float(np.nanquantile(kolisani(skoly_26), KVANTIL_ROZMEZI))

t1 = tst[p1].reset_index(drop=True)
delta = np.fmax(DELTA_VAL, np.nan_to_num(kolisani(t1), nan=0.0))
t1["P_DOLNI"] = val_model.predict_proba(posun_zaka(t1, -delta)[FEATURE])[:, 1]
t1["P_HORNI"] = val_model.predict_proba(posun_zaka(t1, delta)[FEATURE])[:, 1]

# Pokrytí: leží skutečná úspěšnost oboru (priorita 1) mezi průměrnou dolní a horní mezí?
obory = (t1.assign(KOLISANI=kolisani(t1))
         .groupby(KLIC).agg(n=("SPLNIL_KRITERIA", "size"), skutecnost=("SPLNIL_KRITERIA", "mean"),
                            predikce=("P_MODEL", "mean"), dolni=("P_DOLNI", "mean"),
                            horni=("P_HORNI", "mean"), kolisani=("KOLISANI", "first")))
obory = obory[obory["n"] >= MIN_UCHAZECU].dropna(subset=["kolisani"])
obory["pokryto"] = obory["skutecnost"].between(obory["dolni"], obory["horni"])
obory["sirka"] = obory["horni"] - obory["dolni"]
obory["skupina"] = pd.qcut(obory["kolisani"], 4, labels=["1_stabilni", "2", "3", "4_kolisave"])
rozmezi = (obory.groupby("skupina", observed=True)[["pokryto", "sirka"]].mean()
           .assign(n_oboru=obory.groupby("skupina", observed=True).size()))
rozmezi.loc["celkem"] = [obory["pokryto"].mean(), obory["sirka"].mean(), len(obory)]
rozmezi.round(3).to_csv(VYSTUP / "rozmezi_pokryti_2026.csv")
print("\nRozmezí šance, pokrytí na 2026 (delta = %.3f):" % DELTA_VAL)
print(rozmezi.round(3))

# Pro produkci: běžné kolísání ze všech známých posunů (2024 -> 2025 i 2025 -> 2026)
sk27 = pd.read_parquet(DATA / "skoly_2027.parquet")
DELTA_PROD = float(np.nanquantile(np.r_[kolisani(skoly_26), kolisani(sk27)], KVANTIL_ROZMEZI))

# =============================================================================
# 5) Produkční model: 2025 + 2026, stejné nastavení
# =============================================================================
vse = pd.concat([trn, tst[trn.columns]], ignore_index=True)
prod = model(MONO, iterace[VITEZ])
prod.fit(vse[FEATURE], vse["SPLNIL_KRITERIA"].values)
prod.save_model(str(VYSTUP / "produkce_2025_2026.cbm"))
(pd.Series(prod.get_feature_importance(), index=FEATURE).sort_values(ascending=False)
 .round(3).to_csv(VYSTUP / "dulezitost_priznaku.csv"))

with open(VYSTUP / "metadata.json", "w", encoding="utf-8") as f:
    json.dump({
        "varianta": VITEZ, "monotonni_omezeni": MONO, "iterace": iterace[VITEZ],
        "parametry": ZAKLAD, "feature": FEATURE, "kategorie": KAT, "monotonni": list(MONOTONNI),
        "trenovaci_roky": [2025, 2026], "kalibrator": None,
        "rozmezi": {"metoda": "posun percentilu zaka v z-skore o max(delta_globalni, kolisani skoly)",
                    "kvantil": KVANTIL_ROZMEZI, "delta_globalni": round(DELTA_PROD, 4),
                    "delta_validace_2026": round(DELTA_VAL, 4),
                    "pokryti_2026": round(float(obory["pokryto"].mean()), 3)},
    }, f, indent=2, ensure_ascii=False)
print("hotovo:", VYSTUP)
