"""
Automatický regresní test Streamlit aplikace app.py (bez prohlížeče).

Spuštění (z libovolné složky):
    python V5/github/test_app.py

Výstup: test_vysledky.csv vedle tohoto souboru
        (jeden řádek = jedna kombinace škola + body, včetně rozmezí šance).
Potřebuje streamlit >= 1.28 (obsahuje streamlit.testing.v1.AppTest).
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

sys.stdout.reconfigure(encoding="utf-8")          # čeština ve Windows konzoli

TADY = Path(__file__).resolve().parent
APP = str(TADY / "app.py")
VYSTUP = TADY / "test_vysledky.csv"
TIMEOUT = 120  # s na jeden běh aplikace (načtení modelu napoprvé chvíli trvá)

# Stejné školy a body jako v testu z 11. 11. 2026
# (kraj, okres nebo None, typ školy, hledaný text v názvu, [(ČJ, MA), ...])
SCENARE = [
    ("Hlavní město Praha", None, "Čtyřleté gymnázium", "Christiana Dopplera",
     [(25, 25), (30, 30), (35, 35), (40, 40), (45, 45), (50, 50), (43, 42)]),
    ("Hlavní město Praha", None, "Čtyřleté gymnázium", "Jana Keplera",
     [(25, 25), (30, 30), (35, 35), (40, 40), (45, 45), (50, 50), (42, 42), (50, 34), (34, 50)]),
    ("Hlavní město Praha", None, "Čtyřleté gymnázium", "Gymnázium Opatov",
     [(25, 25), (30, 30), (35, 35), (40, 40), (45, 45), (50, 50), (39, 38)]),
    ("Hlavní město Praha", None, "Čtyřleté gymnázium", "1. Slovanské",
     [(10, 10), (12, 12), (20, 20), (25, 25), (35, 35), (45, 45)]),
    ("Jihomoravský", None, "Čtyřleté gymnázium", "Matyáše Lercha",
     [(25, 25), (35, 35), (40, 39), (45, 45), (50, 50)]),
    ("Jihomoravský", None, "Čtyřleté gymnázium", "Gymnázium – Gymnázium (Brno",
     [(35, 35), (40, 40)]),  # test nejednoznačných názvů – projde VŠECHNY shody
    ("Olomoucký", None, "Čtyřleté gymnázium", "Slovanské gymnázium",
     [(25, 25), (30, 30), (34, 33), (35, 35), (40, 40), (45, 45)]),
    ("Olomoucký", None, "učiliště", "Tiskař",
     [(10, 10), (20, 20), (30, 30), (40, 40)]),
    ("Moravskoslezský", None, "Čtyřleté gymnázium", "Hladnov",
     [(20, 20), (26, 25), (30, 30), (35, 35), (40, 40)]),
    ("Moravskoslezský", None, "učiliště", "Chemik oper",
     [(0, 0), (5, 5), (11, 11), (15, 15), (25, 25), (35, 35)]),
    ("Plzeňský", None, "Čtyřleté gymnázium", "Masarykovo",
     [(25, 25), (30, 30), (34, 33), (35, 35), (40, 40), (45, 45)]),
    ("Hlavní město Praha", None, "odborná škola", "Dušní – Informační",
     [(15, 15), (18, 18), (25, 25), (30, 30), (35, 35), (45, 45)]),
    ("Hlavní město Praha", None, "odborná škola", "Třebešíně – Stroj",
     [(10, 10), (13, 13), (20, 20), (25, 25), (30, 30), (40, 40)]),
    ("Hlavní město Praha", None, "Osmileté", "Jana Keplera",
     [(30, 30), (35, 35), (40, 40), (41, 40), (45, 45), (50, 50)]),
    ("Hlavní město Praha", None, "Čtyřleté gymnázium", "sportovní přípravou",
     [(20, 20), (25, 25), (30, 30), (40, 40)]),
    ("Středočeský", None, "odborná škola", "Rakovník, SOŠ – Agropod",
     [(10, 10), (16, 15), (20, 20), (30, 30), (40, 40)]),
    ("Jihočeský", "Písek", "učiliště", "Autotronik",
     [(5, 5), (9, 9), (15, 15), (25, 25), (35, 35)]),
    ("Zlínský", None, "odborná škola", "Vyšší polic",
     [(20, 20), (22, 22), (30, 30), (40, 40), (50, 50)]),
]


def najdi(widgets, label_start):
    for w in widgets:
        if str(w.label).startswith(label_start):
            return w
    raise KeyError(f"Widget začínající '{label_start}' nenalezen. "
                   f"K dispozici: {[w.label for w in widgets]}")


def vyber(at, label_start, hledany_text, vsechny=False):
    """Vybere položku v selectboxu podle podřetězce. Vrací seznam indexů shod."""
    sb = najdi(at.selectbox, label_start)
    opts = [str(o) for o in sb.options]
    shody = [i for i, o in enumerate(opts) if hledany_text.lower() in o.lower()]
    if not shody:
        raise KeyError(f"'{hledany_text}' není v '{label_start}' ({len(opts)} možností)")
    return sb, opts, shody


def sance_z_grafu(at):
    """Vytáhne číslo z plotly gauge (Indicator) nebo z metriky."""
    try:
        for el in at.get("plotly_chart"):
            spec = json.loads(el.proto.spec)
            for tr in spec.get("data", []):
                if "value" in tr:
                    return float(tr["value"])
    except Exception:
        pass
    try:
        for m in at.metric:
            v = re.search(r"([\d.,]+)\s*%", str(m.value))
            if v:
                return float(v.group(1).replace(",", "."))
    except Exception:
        pass
    return None


def rozmezi(at):
    """Rozmezí z textu „Rozmezí šance: 19–91 %“, nebo („minimálně X“, None), jinak (None, None)."""
    for e in at.markdown:
        r = re.search(r"Rozmezí šance: (\d+)–(\d+) %", str(e.value))
        if r:
            return int(r.group(1)), int(r.group(2))
        r = re.search(r"Šance minimálně (\d+) %", str(e.value))   # body jasně nad laťkou
        if r:
            return int(r.group(1)), None
    return None, None


def texty(at):
    out = {}
    for kind in ("info", "success", "warning", "error", "caption", "markdown"):
        try:
            out[kind] = " | ".join(str(e.value) for e in getattr(at, kind))
        except Exception:
            out[kind] = ""
    return out


def nastav_cislo(at, label_start, hodnota):
    najdi(at.number_input, label_start).set_value(hodnota)


def main():
    radky = []
    t0 = time.time()
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    if at.exception:
        print("Aplikace spadla už při startu:", at.exception)
        sys.exit(1)

    for kraj, okres, typ, hledat, body in SCENARE:
        # kraj → okres → typ (každá změna spustí přepočet aplikace)
        sb, opts, sh = vyber(at, "Kraj", kraj)
        sb.select_index(sh[0]).run()
        if okres:
            sb, opts, sh = vyber(at, "Okres", okres)
            sb.select_index(sh[0]).run()
        else:
            sb, opts, sh = vyber(at, "Okres", "Všechny")
            sb.select_index(sh[0]).run()
        sb, opts, sh = vyber(at, "Typ", typ)
        sb.select_index(sh[0]).run()

        _, opts_skol, shody = vyber(at, "Škola", hledat)
        # u běžných škol stačí první shoda; u nejednoznačného názvu projdi všechny
        cile = shody if "Gymnázium – Gymnázium (Brno" in hledat else shody[:1]

        for idx in cile:
            najdi(at.selectbox, "Škola").select_index(idx).run()
            for cj, ma in body:
                nastav_cislo(at, "ČJ", cj)
                nastav_cislo(at, "Matematika", ma)
                at.run()
                najdi(at.button, "Spočítat").click().run()
                tx = texty(at)
                p = sance_z_grafu(at)
                od, do = rozmezi(at)
                radky.append({
                    "kraj": kraj, "okres": okres or "", "typ": typ,
                    "skola_obor": opts_skol[idx], "index_v_nabidce": idx,
                    "pocet_shod_nazvu": len(shody),
                    "cj": cj, "ma": ma, "soucet": cj + ma,
                    "sance_pct": p, "rozmezi_od": od, "rozmezi_do": do,
                    "info": tx["info"], "success": tx["success"],
                    "warning": tx["warning"], "error": tx["error"],
                    "caption": tx["caption"],
                    "vyjimka": str(at.exception) if at.exception else "",
                })
                print(f"{opts_skol[idx][:55]:55s} {cj:>2}+{ma:<2} → {p} ({od}–{do})")

    df = pd.DataFrame(radky)
    df.to_csv(VYSTUP, index=False, encoding="utf-8-sig", sep=";")
    chyby = (df["vyjimka"] != "").sum()
    print(f"\nHotovo: {len(df)} řádků za {time.time()-t0:.0f} s, výjimek: {chyby} → {VYSTUP}")


if __name__ == "__main__":
    main()
