# -*- coding: utf-8 -*-
"""
Centralna konfiguracija za PvZ Whack-a-Zombie RL agenta.

Sve vrednosti koje zavise od rezolucije date su kao frakcije (0.0 - 1.0)
u odnosu na AUTOMATSKI DETEKTOVAN OKVIR IGRE (svetla 4:3 oblast između
crnih traka u fullscreen režimu), ne na ceo monitor - vidi
bot/vision/game_region.py. Tako kalibracija radi na bilo kojoj rezoluciji.
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

# Pozicija travnjaka kao frakcije okvira igre. Vrednosti odgovaraju
# kanonskom PvZ rasporedu u 800x600: mreža od x=40 do x=760 (0.05-0.95),
# y=80 do y=580 (0.1333-0.9667) - potvrđeno kalibracionim snimkom.
LAWN_LEFT_FRAC = 0.0500   # leva ivica prve kolone
LAWN_TOP_FRAC = 0.1333    # gornja ivica prvog reda
LAWN_RIGHT_FRAC = 0.9500  # desna ivica poslednje kolone
LAWN_BOTTOM_FRAC = 0.9667 # donja ivica poslednjeg reda

# ----------------------------------------------------------------------------
# HUD - karte sa semenima (3 slota u Whack-a-Zombie: Grave Buster,
# Cherry Bomb, Ice-shroom). ROI po slotu kao frakcije ekrana.
# ----------------------------------------------------------------------------
CARD_SLOTS_FRAC = [
    # (left, top, width, height) za svaki slot, sleva nadesno
    # (fino doterano po drugom kalibracionom snimku, frakcije okvira igre)
    (0.1230, 0.0220, 0.0550, 0.1030),  # slot 0: Potato Mine
    (0.1960, 0.0220, 0.0550, 0.1030),  # slot 1: Grave Buster
    (0.2720, 0.0220, 0.0550, 0.1030),  # slot 2: Ice-shroom
]
# Redosled karata u slotovima (potvrđeno u igri: mina, buster, led)
CARD_ORDER = ["potato_mine", "gravebuster", "ice"]

# Karta je spremna kad je svetla; dok se puni / nema sunca prekrivena je
# tamnim slojem koji obara svetlinu ~50%. Pragovi po slotu su ~85% izmerene
# svetline SPREMNE karte (izmereno na korisnikovim frejmovima 05.07.2026:
# potato val=215, gravebuster val=185, ice val=204; saturacija se NE koristi
# jer je Grave Buster sivkasta biljka - spreman ima sat svega 52).
CARD_READY_VALUE_PER_SLOT = [180, 155, 170]

# Brojač sunca (ROI, koristi se samo za debug/snimanje)
SUN_COUNTER_FRAC = (0.0170, 0.0080, 0.0830, 0.1280)

# ----------------------------------------------------------------------------
# Kosačice - okviri oko kolica na levoj ivici svake od 5 staza.
# Kolica sede u DONJEM delu trake staze (60%-102% visine reda).
# ----------------------------------------------------------------------------
MOWER_STRIP_LEFT_FRAC = 0.0020
MOWER_STRIP_WIDTH_FRAC = 0.0530
MOWER_ROW_OFFSET_FRAC = 0.60   # početak okvira unutar reda (deo visine reda)
MOWER_ROW_HEIGHT_FRAC = 0.42   # visina okvira (deo visine reda)
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

# HSV opseg za sunca - donja nijansa 19 (čisto žuto): "POW!" efekat udarca
# je narandžast (hue ~10-18) i ne sme da upada u masku kao lažno sunce.
# Saturacija/svetlina spuštene: sunce pred istek BLEDI (treperi), pa bleđa
# sunca moraju i dalje da se hvataju.
SUN_HSV_LOWER = (19, 35, 150)
SUN_HSV_UPPER = (40, 255, 255)
# Površina konture sunca u NATIVNIM pikselima igre (800x600); u realnom
# vremenu se skalira sa (visina_okvira_igre / 600)^2.
# VAŽNO: prag svetline propušta samo sjajno JEZGRO sunca (ne i zrake/oreol),
# pa je blob pojedinačnog sunca mali (~250-450 nativno); zombiji ispuštaju
# i TROJKE sunaca čiji spojeni blob ume biti ogroman - zato širok opseg.
SUN_AREA_MIN_NATIVE = 175
SUN_AREA_MAX_NATIVE = 5200  # trojka sunaca + sjaj = ogroman spojeni blob

# Šabloni za detekciju kraja partije. Poraz se prepoznaje po kinematici
# "zombiji ulaze u kuću" (vrata kuće - isečeno sa stvarnog snimka poraza);
# kraj partije BEZ te kinematike = pobeda (partija ima samo ta dva ishoda).
TEMPLATE_WIN = os.path.join(ASSETS_DIR, "moneybag.png")
TEMPLATE_LOSS = os.path.join(ASSETS_DIR, "gameover.png")
TEMPLATE_LOSS_HOUSE = os.path.join(ASSETS_DIR, "loss_house.png")
TEMPLATE_MATCH_THRESHOLD = 0.80
# Kraj partije se detektuje i nestankom HUD-a (ikona sunca) - pobednički/
# gubitnički ekran prekriva HUD; posle ovoliko sekundi bez HUD-a partija
# se smatra završenom (frejm se snima u stats/ radi klasifikacije)
HUD_MISSING_END_SECS = 3.0

# ----------------------------------------------------------------------------
# Model ljudske motorike (deli ga simulator i humanizer da bi politika
# naučena u simulatoru bila izvodljiva istom brzinom u pravoj igri)
# ----------------------------------------------------------------------------
# (ubrzano posle pravih partija - "human-like" ne znači trom; dobar
# igrač whack-a-mole žanra je BRZ, prirodnost daju putanje i varijacija)
HUMAN_REACTION_MIN = 0.04   # s - kašnjenje odluke pre početka pokreta
HUMAN_REACTION_MAX = 0.10
FITTS_A = 0.04              # s - fiksni deo Fittsovog zakona
FITTS_B = 0.055             # s/bit - nagib Fittsovog zakona
FITTS_TARGET_W = 60.0       # px - efektivna širina mete
CLICK_HOLD_MIN = 0.03       # s - držanje klika
CLICK_HOLD_MAX = 0.05
MULTI_CLICK_GAP_MIN = 0.04  # s - pauza između uzastopnih udaraca
MULTI_CLICK_GAP_MAX = 0.08
OVERSHOOT_PROB = 0.10       # verovatnoća malog promašaja pa korekcije (samo pokret, ne klik)
NOMINAL_SCREEN_W = 1920.0   # za pretvaranje ćelija u px u simulatoru

# Tačka nišanjenja na zombiju: "oko vrata ili tik niže" - 0.28 je gađalo
# noge/zemlju (pogađalo biljke), 0.05 je u gornjem redu umelo da zakači HUD
# karte ("uzme biljku" -> zaglavljen paket na kursoru -> izgubljene kosačice)
AIM_Y_FRAC = 0.16

# Prioritet sunca: kupljenje ima prednost nad ostalim akcijama kada je
# bezbedno - "bezbedno" znači: najbliži zombi >= SAFE_COL, ILI (malo
# zombija na terenu i najbliži >= RELAXED_COL). Puna kasa = više biljaka.
SUN_PRIORITY = True
SUN_PRIORITY_SAFE_COL = 2.5
SUN_PRIORITY_RELAXED_COL = 1.6
SUN_PRIORITY_FEW_ZOMBIES = 4

# Sigurnosni sloj: zombi bliži kući od ovoliko kolona se udara BEZUSLOVNO,
# bez obzira šta politika kaže (garancija da se kosačice ne troše).
# Pod velikom navalom (>= 12 zombija) prag se širi za +0.7 kolona -
# odbrana dobija apsolutni prioritet baš pri kraju kad je najopasnije.
SAFETY_OVERRIDE = True
SAFETY_COL_THRESHOLD = 1.3
SAFETY_SWARM_BONUS = 0.7
SAFETY_SWARM_COUNT = 12

# Grobovi se u Whack-a-Zombie NE stvaraju u prve 3 kolone - svaka "grob"
# detekcija tamo je lažna (najčešće naša mina ili tragovi na travi)
GRAVE_MIN_COL = 3

# Ice-shroom se ne aktivira dok navala nije ozbiljna (politika je trenirana
# na prag ~22 zombija; ovo je dodatna runtime brava za svaki slučaj).
# Obrnuto: na >= ICE_FORCE_ZOMBIES led se aktivira PRISILNO ako je spreman -
# garancija da veliki finalni talas dočeka zamrzavanje.
# Led je za PRED-finalnu gomilu: kad se nakupi jako puno zombija,
# zamrzavanje spašava kosačice; ispod MIN se ne pušta (štedi se za vrhunac).
# (24: na 26-30 se u praksi nije aktivirao iako je navala bila ozbiljna -
# YOLO u gužvi potcenjuje broj zbog preklapanja)
ICE_MIN_ZOMBIES = 20
ICE_FORCE_ZOMBIES = 24

# Sunce u letu ka brojaču se ne juri: posle klika na sunca, kupljenje se
# pauzira ovoliko sekundi (let traje ~0.7 s)
SUN_BLIND_AFTER_COLLECT = 1.2

# ----------------------------------------------------------------------------
# Tok partije / bezbednost
# ----------------------------------------------------------------------------
GAME_HARD_TIMEOUT = 420.0   # s - apsolutni limit; upisuje se kao TIMEOUT, ne kao pobeda
DEBUG_RECORD_VIDEO = False  # snimanje debug overlay videa u stats/ (za rad/analizu)
DEBUG_SHOW_WINDOW = False   # živi debug prozor (NE koristiti tokom prave partije -
                            # u v1.0 je pravio "slepu zonu" u koju bot nije smeo da klikne)
START_SUN = 150             # početno sunce u mini-igri (za interni registar)
SUN_VALUE = 25

# Cene potvrđene u igri: Potato Mine 25, Grave Buster 75, Ice-shroom 75.
# Za pravu igru priuštivost se ionako čita vizuelno sa karata; ovo koristi
# simulator i interni registar sunca.
PLANT_COSTS = {"potato_mine": 25, "gravebuster": 75, "ice": 75}
