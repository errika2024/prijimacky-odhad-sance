# Model V5 – predikce přijetí na SŠ (finální verze)

Odhad šance na přijetí na střední školu z dat CERMAT (1. kolo, 2024–2026) pro rok 2027.
CatBoost s monotónními omezeními, žák popsaný percentilem v kohortě, škola historií
v percentilech (loni a předloni), rozmezí šance podle kolísání laťky přijetí.

## Obsah složky

| Složka / soubor | Obsah |
|---|---|
| `github/` | Stručná, čistá verze pro GitHub: `01_priprava_dat.py`, `02_model.py`, `app.py`, `test_app.py` |
| `podrobne/` | Studijní verze se stejnými výstupy: vysvětlení „proč“, teorie, otázky k pohovoru, ponechané experimenty |
| `data/` | Výstupy kroku 01: `model_data.parquet`, `skoly_2027.parquet`, `tabulky_percentilu_2026.csv`, `skoly_2027_nazvy.csv` |
| `vystup/` | Výstupy kroku 02: modely `.cbm`, `metadata.json`, metriky (`test_2026.csv`, `kalibrace_2026_priorita1.csv`, `vyber_oof_2025.csv`, `rozmezi_pokryti_2026.csv`, `dulezitost_priznaku.csv`) |
| `github/test_vysledky.csv` | Výsledek posledního testu aplikace |

Vstupní data CERMAT zůstávají ve složce projektu `Prijimacky/DATA` (`PZ{rok}_kolo1_*.xlsx`, `skoly_2027.csv`).

## Spuštění (ze složky `Prijimacky`)

```
python V5/github/01_priprava_dat.py     # DATA/*.xlsx -> V5/data/      (cca 2 min)
python V5/github/02_model.py            # V5/data/ -> V5/vystup/       (cca 15–30 min)
streamlit run V5/github/app.py          # aplikace
python V5/github/test_app.py            # test aplikace: 18 škol, 110 kombinací
```

Podrobné verze se spouštějí stejně (`V5/podrobne/..._podrobne.py`). V `02_model_podrobne.py`
přepni `EXPERIMENTY = True`, pokud chceš znovu spustit i zamítnuté varianty.

## Výsledky (test 2026, priorita 1, n = 104 637, model trénovaný jen na 2025)

| Model | AUC | Brier | LogLoss | ECE |
|---|---|---|---|---|
| Benchmark: percentil vs. loňská laťka + log. regrese | 0,9365 | 0,0986 | 0,3191 | 0,025 |
| **V5** (bez kalibrace) | **0,9501** | **0,0874** | **0,2798** | 0,015 |

- Výběr varianty na OOF 2025, GroupKFold po oborech škol: vyhrála monotónní varianta (hloubka 4, 611 iterací).
- OOF 2025 (AUC 0,942) odpovídá testu 2026 (0,950), validace je tedy realistická.
- Produkční model je natrénovaný na 2025 + 2026 se stejným nastavením.

## Laťka přijetí a rozmezí šance

**Laťka přijetí** = body (ČJ + M, max 100), nad kterými bylo 9 z 10 přijatých (10. percentil přijatých).

**Rozmezí šance**: šance žáka, kdyby se laťka posunula nahoru nebo dolů o `delta` (v z-skóre pořadí),
`delta = max(běžné kolísání, kolísání školy 2025 vs. 2026)`, běžné kolísání = 80% kvantil posunu laťky mezi roky.
Posun laťky se v aplikaci ukazuje v bodech jako jedno číslo pro školu („laťka 88, v roce 2027 asi 81–92 bodů“).

Ověření na roce 2026 (model jen z 2025, delta jen z posunu 2024 → 2025):

| Metoda | Pokrytí skutečné úspěšnosti oboru | Průměrná šířka |
|---|---|---|
| **Posun pořadí žáka (zvoleno)** | **82 %** (stabilní obory 84 %, kolísavé 80 %) | 36 p. b. |
| Posun laťky školy | 75 % | 30 p. b. |
| Scénáře let (historie jen loni / jen předloni) | 23 % | 7 p. b. |

**„Šance minimálně X %“**: místo rozmezí, když má žák víc bodů než laťka každého roku 2024–2026,
víc než horní odhad laťky 2027 a dolní mez šance je alespoň 80 %. Aplikace nikdy neukazuje 100 %
(rozhoduje i pořadí přihlášek, konkurence se mění, škola může mít další kritéria).

## Známá omezení

- Elitní gymnázia: laťka se mezi roky posouvá až o 10–20 bodů. Kepler (4leté) 2026 při 80–84 bodech:
  skutečnost 3 %, odhad 52 %, rozmezí 14–89 %. Takový skok je mimo běžných 8 z 10, dolní mez na riziko upozorní.
- Body se na percentil převádějí podle rozložení roku 2026 (předpoklad stejně těžkého testu v roce 2027).
- Cíl u priorit 2–5 je optimistický (přijetí na vyšší prioritu se počítá jako úspěch); aplikace ukazuje prioritu 1.
- Obory s talentovou zkouškou a školní kritéria model nezná.
