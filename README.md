# Razvoj autonomnog agenta za mini-igru "Whack-a-Zombie" u Plants vs Zombies-u korišćenjem računarskog vida i dubokog potkrepljujućeg učenja

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-magenta)
![PyTorch](https://img.shields.io/badge/RL-Double%20DQN-orange)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green)

## 📋 O projektu

Inteligentni softverski agent koji potpuno autonomno igra "Whack-a-Zombie"
mini-igru u *Plants vs. Zombies GOTY* — od glavnog menija do upisa statistike —
i to potezima koji se ne razlikuju od ljudskih.

Sistem kombinuje:
- **YOLOv8 Nano** za detekciju zombija (običan / čunj / kanta), grobova i Grave Buster-a,
- **HSV computer vision** za sakupljanje sunca,
- **Duboko potkrepljujuće učenje (Double DQN)** za donošenje odluka,
- **Model ljudske motorike** (Fittsov zakon, minimum-jerk putanje, Bezier krive)
  za izvršavanje poteza.

**Autor:** Igor Ludajić (RA 46/2022)

---

## 🧠 Arhitektura v2.0 (RL)

```
            SIMULATOR  ──── trening (DQN) ────►  POLITIKA (policy.pt)
   (mehanika igre + model                              │
    ljudske motorike +                                 │  isti format stanja
    domain randomization)                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ PRAVA IGRA:  kaptura → YOLO/HSV/HUD → Snapshot → politika →     │
  │              humanizovano izvršenje (1/2/3 udarca po tipu)      │
  └─────────────────────────────────────────────────────────────────┘
```

**Sim-to-real pristup:** RL trening direktno na igri je nepraktičan (jedna
partija = ~4 minuta realnog vremena). Zato se politika trenira u brzom
simulatoru koji modelira mehaniku igre i — ključno — **ugrađuje ljudska
motorička ograničenja**: svaka akcija u simulatoru traje onoliko koliko bi
trajala čoveku (vreme reakcije + Fittsov zakon + trajanje klikova). Naučena
strategija je zato izvodljiva ljudskom brzinom: u špicevima talasa čekić
fizički ne može sve da postigne, pa agent uči da štedi sunce i koristi
Cherry Bomb / Ice-shroom — kao što bi i čovek.

**Prostor akcija (138):** udari zombija u ćeliji (45), Grave Buster na grob
(45), Cherry Bomb na ćeliju (45), Ice-shroom, pokupi sunce, čekaj — uz
maskiranje nevalidnih akcija. Politika bira ŠTA, skriptirani sloj rešava KAKO
(tačan broj udaraca: 1 običan, 2 čunj, 3 kanta; putanja miša; tajming).

**Algoritam učenja — Double DQN + DQfD elementi:**
- *n-step povraćaji* sa SMDP diskontovanjem (akcije traju različito, pa je
  diskont `γ^trajanje` po tranziciji);
- *trajni demo bafer*: 30k tranzicija pro-heuristike (koristi i biljke) čini
  25% svakog batch-a i nikada se ne pregazi — bez toga se politika urušava
  kada sopstveni podaci istisnu demonstracije;
- *supervizovani margin gubitak* na demo uzorcima (DQfD): demo akcija mora
  imati Q-vrednost bar za marginu veću od ostalih, čime politika brzo
  dostiže nivo demonstratora, a TD komponenta je dalje popravlja;
- *balansirano istraživanje*: nasumična akcija se bira uniformno po tipu
  akcije (ne po ćeliji), da 45 cherry-ćelija ne dominira istraživanjem;
- *selekcija modela periodičnom greedy evaluacijom* (svake 20k koraka, bez
  istraživanja): čuva se politika sa najboljim win rate-om uz kaznu za
  iskorišćene kosačice — trening metrika je zagađena epsilon šumom, pa se
  najbolji checkpoint bira po stvarnoj snazi.

**Navigacija kroz meni je namerno ostala deterministička** (template
matching): meni je determinističko UI okruženje bez neizvesnosti, pa RL tu
ne može ništa da nauči što skripta ne radi pouzdanije.

### Šta je novo u odnosu na v1.0

| Problem v1.0 | Rešenje v2.0 |
| :--- | :--- |
| Koristi samo Grave Buster | RL politika uči kada se isplate sve 3 biljke |
| Brljavo kupljenje sunca | Sunce je akcija u politici + interni registar sunca |
| Slaba detekcija grobova | Niži prag za klasu `grave` + vremenska postojanost (N od M frejmova) |
| Dupli udarci na zombije u umiranju | Tracker sa stabilnim ID-jevima i "dying" periodom |
| Nadljudska brzina miša (10 ms potezi) | Humanizer: Fitts + minimum-jerk + Bezier + overshoot |
| "WIN" upisivan po tajmeru od 260 s | Pobeda/poraz isključivo detekcijom ekrana |
| Debug prozor pravio slepu zonu za klik | Opcioni video snimak umesto prozora |

---

## 📊 Statistika

Svaka partija se upisuje u `stats/game_stats.csv` (zbirno) i
`stats/games/*.json` (detaljno):
rezultat, trajanje, ubistva po tipu zombija, broj klikova, sakupljeno /
potrošeno sunce, **upotreba svake biljke**, **broj iskorišćenih kosačica**
(cilj: 0), broj zombija u kritičnoj zoni, najbliži prilaz kući i raspodela
akcija politike (analiza naučene strategije).

---

## ⚙️ Instalacija i pokretanje

```bash
pip install ultralytics opencv-python mss pyautogui numpy torch matplotlib
```

**1. Trening politike u simulatoru** (jednokratno, na CPU). Dva komplementarna
pristupa — vidi `REZULTATI.md` za poređenje:
```bash
python train_rl.py --steps 400000 --demo-steps 40000  # DQN (RL iz nagrade), ~stabilno do 94%
python train_bc.py --samples 150000 --epochs 10       # behavior cloning, ~100% (izabrano)
```
Rezultati (kriva učenja, statistika epizoda) idu u `runs/rl/`.

**2. Finalni izbor i evaluacija** (presuđuje nezavisni test set):
```bash
python tools/select_policy.py --episodes 150 --seed 555  # bira najbolju -> models/policy.pt
python evaluate_policy.py --episodes 200 --seed 2024            # potvrda izabrane politike
python evaluate_policy.py --episodes 150 --baseline v1   # heuristika v1.0 (bez biljaka)
python evaluate_policy.py --episodes 150 --baseline pro  # pro-heuristika
```

**3. Provera kalibracije** (jednokratno po mašini/rezoluciji): pokrenuti
Whack-a-Zombie, pa:
```bash
python tools/calibrate_hud.py
```
i proveriti da se mreža na `calibration_preview.png` poklapa sa igrom.
Ako ne, podesiti frakcije u `config.py`.

**4. Igranje:** otvoriti PvZ na glavnom meniju i:
```bash
python main.py
```
Agent sam bira mini-igru iz menija, prolazi popup-e, igra do kraja, upisuje
statistiku i gasi se. Hitno gašenje: gurnuti miš u gornji levi ugao ekrana
(pyautogui failsafe) ili `Ctrl+C` u terminalu.

---

## 🗂️ Struktura projekta

```
config.py             # sva podešavanja (ROI frakcije, pragovi, motorika)
main.py               # ceo tok: meni → partija → statistika → izlaz
train_rl.py           # DQN trening u simulatoru (RL iz nagrade + DQfD)
train_bc.py           # behavior cloning (imitacija pro-heuristike) — izabrani pristup
evaluate_policy.py    # evaluacija politike / heuristike u simulatoru
navigator.py          # deterministička navigacija kroz meni
bot/
  actions.py          # diskretni prostor akcija (138)
  state.py            # Snapshot → opservacija (278) + maska akcija
  sim/simulator.py    # simulator igre sa modelom ljudske motorike
  agent/              # CNN Q-mreža, Double DQN+DQfD, politika, heuristike
  vision/             # kaptura, YOLO detektor, sunca, HUD, kosačice, tracker
  control/            # humanizer (Fitts/Bezier) i izvršilac akcija
  game/               # mapiranje na mrežu 5×9 i petlja prave partije
  stats/              # logger statistike (CSV + JSON)
tools/calibrate_hud.py   # provera kalibracije ekrana
tools/select_policy.py   # finalni izbor politike na nezavisnom setu
tools/analyze_stats.py   # analiza odigranih partija (grafici)
gameplay.py           # v1.0 heuristički bot (zadržan radi poređenja)
```

---

## ⚠️ Rešavanje čestih problema

**Miš klikće van prozora igre / "ludi" po ekranu**
Windows DPI skaliranje: Display settings → Scale = 100%, pa restart skripte.

**Agent ne sadi biljke**
Proveriti `tools/calibrate_hud.py` — ROI slotova karata mora da pokriva karte.
Po potrebi podesiti `CARD_SLOTS_FRAC` i pragove `CARD_READY_*` u `config.py`.

**Politika ne postoji**
`models/policy.pt` nastaje tek posle `python train_rl.py`.
