# -*- coding: utf-8 -*-
"""
Centralna konfiguracija za PvZ Whack-a-Zombie RL agenta.

Sve vrednosti koje zavise od rezolucije ekrana date su kao frakcije
(0.0 - 1.0) u odnosu na primarni monitor, tako da rade na bilo kojoj
rezoluciji dok god je igra u fullscreen režimu.
Za proveru kalibracije pokrenuti: python tools/calibrate_hud.py
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Putanje
# ----------------------------------------------------------------------------
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "best.pt")
POLICY_PATH = os.path.join(ROOT_DIR, "models", "policy.pt")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
STATS_DIR = os.path.join(ROOT_DIR, "stats")
RUNS_RL_DIR = os.path.join(ROOT_DIR, "runs", "rl")

# ----------------------------------------------------------------------------
# Geometrija table (5 redova x 9 kolona)
# ----------------------------------------------------------------------------
ROWS = 5
COLS = 9

# Pozicija travnjaka kao frakcije ekrana (kalibrisano za PvZ GOTY fullscreen,
# nativna rezolucija 800x600 skalirana na ceo ekran).
LAWN_LEFT_FRAC = 0.0475   # leva ivica prve kolone
LAWN_TOP_FRAC = 0.1300    # gornja ivica prvog reda
LAWN_RIGHT_FRAC = 0.9525  # desna ivica poslednje kolone
LAWN_BOTTOM_FRAC = 0.9583 # donja ivica poslednjeg reda

# ----------------------------------------------------------------------------
# HUD - karte sa semenima (3 slota u Whack-a-Zombie: Grave Buster,
# Cherry Bomb, Ice-shroom). ROI po slotu kao frakcije ekrana.
# ----------------------------------------------------------------------------
CARD_SLOTS_FRAC = [
    # (left, top, width, height) za svaki slot, sleva nadesno
    (0.1100, 0.0083, 0.0563, 0.0967),  # slot 0: Grave Buster
    (0.1700, 0.0083, 0.0563, 0.0967),  # slot 1: Cherry Bomb
    (0.2300, 0.0083, 0.0563, 0.0967),  # slot 2: Ice-shroom
]
# Redosled karata u slotovima (proveriti u igri i po potrebi zameniti!)
CARD_ORDER = ["gravebuster", "cherry", "ice"]

# Prag prosečne saturacije iznad kog se karta smatra spremnom
# (siva/tamna karta = puni se ili nema dovoljno sunca).
CARD_READY_SATURATION = 60
CARD_READY_VALUE = 90

# Brojač sunca (ROI, koristi se samo za debug/snimanje)
SUN_COUNTER_FRAC = (0.0250, 0.0083, 0.0750, 0.0500)

# ----------------------------------------------------------------------------
# Kosačice - uske trake na levoj ivici svake od 5 staza
# ----------------------------------------------------------------------------
MOWER_STRIP_LEFT_FRAC = 0.0050
MOWER_STRIP_WIDTH_FRAC = 0.0400
MOWER_CHANGE_THRESHOLD = 0.50  # korelacija sa baseline ispod ovoga = kosačica aktivirana

# ----------------------------------------------------------------------------
# Percepcija (YOLO + HSV)
# ----------------------------------------------------------------------------
# Prag poverenja po klasi - grobovi su teži za detekciju pa imaju niži prag,
# a lažne pozitive filtrira vremenska postojanost (vidi detector.py).
CLASS_CONF = {
    "zombie": 0.30,
    "conehead": 0.30,
    "buckethead": 0.30,
    "damaged_buckethead": 0.25,
    "grave": 0.12,
    "gravebuster": 0.20,
}
YOLO_IMGSZ = 640  # = veličina treninga; ako je detekcija slaba na visokoj
                  # rezoluciji ekrana, probati 1280 (v1.0 je tako radio, 4x sporije)
GRAVE_PERSISTENCE_FRAMES = 3   # grob mora biti viđen u bar N od poslednjih M frejmova
GRAVE_PERSISTENCE_WINDOW = 8

# HSV opseg za sunca (nasleđeno iz v1.0, provereno u igri)
SUN_HSV_LOWER = (16, 52, 172)
SUN_HSV_UPPER = (40, 255, 255)
SUN_AREA_MIN = 1700
SUN_AREA_MAX = 8000

# Šabloni za detekciju kraja partije
TEMPLATE_WIN = os.path.join(ASSETS_DIR, "moneybag.png")
TEMPLATE_LOSS = os.path.join(ASSETS_DIR, "gameover.png")
TEMPLATE_MATCH_THRESHOLD = 0.80

# ----------------------------------------------------------------------------
# Model ljudske motorike (deli ga simulator i humanizer da bi politika
# naučena u simulatoru bila izvodljiva istom brzinom u pravoj igri)
# ----------------------------------------------------------------------------
HUMAN_REACTION_MIN = 0.10   # s - kašnjenje odluke pre početka pokreta
HUMAN_REACTION_MAX = 0.22
FITTS_A = 0.08              # s - fiksni deo Fittsovog zakona
FITTS_B = 0.10              # s/bit - nagib Fittsovog zakona
FITTS_TARGET_W = 60.0       # px - efektivna širina mete
CLICK_HOLD_MIN = 0.05       # s - držanje klika
CLICK_HOLD_MAX = 0.09
MULTI_CLICK_GAP_MIN = 0.08  # s - pauza između uzastopnih udaraca
MULTI_CLICK_GAP_MAX = 0.15
OVERSHOOT_PROB = 0.12       # verovatnoća malog promašaja pa korekcije (samo pokret, ne klik)
NOMINAL_SCREEN_W = 1920.0   # za pretvaranje ćelija u px u simulatoru

# ----------------------------------------------------------------------------
# Tok partije / bezbednost
# ----------------------------------------------------------------------------
GAME_HARD_TIMEOUT = 420.0   # s - apsolutni limit; upisuje se kao TIMEOUT, ne kao pobeda
DEBUG_RECORD_VIDEO = False  # snimanje debug overlay videa u stats/ (za rad/analizu)
DEBUG_SHOW_WINDOW = False   # živi debug prozor (NE koristiti tokom prave partije -
                            # u v1.0 je pravio "slepu zonu" u koju bot nije smeo da klikne)
START_SUN = 150             # početno sunce u mini-igri (za interni registar)
SUN_VALUE = 25

PLANT_COSTS = {"gravebuster": 75, "cherry": 150, "ice": 75}
