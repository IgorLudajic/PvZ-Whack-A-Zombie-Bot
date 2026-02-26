# Razvoj autonomnog agenta za mini-igru “Whack-a-Zombie” u Plants vs Zombies-u korišćenjem računarskog vida i heurističkih algoritama


![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-magenta)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green)

## 📋 O Projektu
Ovaj projekat predstavlja razvoj inteligentnog softverskog agenta sposobnog za potpuno autonomno igranje "Whack-a-Zombie" mini-igre unutar *Plants vs. Zombies* u realnom vremenu.

Sistem kombinuje **Deep Learning (YOLOv8)** za detekciju neprijatelja i prepreka, **Computer Vision (HSV filtriranje)** za sakupljanje resursa, i **heurističke algoritme** za donošenje odluka u deliću sekunde.

**Autor:** Igor Ludajić (RA 46/2022)  

---

## 🚀 Ključne Funkcionalnosti

### 1. Percepcija (Computer Vision)
* **YOLOv8 Nano Model:** Treniran na prilagođenom skupu podataka za detekciju 6 klasa:
    * `zombie`, `conehead`, `buckethead`, `damaged_buckethead`
    * `grave`, `gravebuster`
* **HSV Filtriranje:** Algoritam za detekciju Sunca na osnovu specifičnog opsega boja i veličine kontura (kako bi se izbegli false positives).

### 2. Logika Odlučivanja (Decision Making)
* **Striktno Levi Prioritet:** Agent eliminiše pretnje koje su najbliže levoj strani ekrana (kući), ignorišući blizinu miša u kritičnim situacijama.
* **Dinamički Batching:** Sistem automatski prilagođava veličinu grupe meta i brzinu kliktanja u zavisnosti od broja zombija na ekranu (od "ljudske" brzine do "swarm" režima).
* **Upravljanje Resursima:** Pametno korišćenje *Gravebuster* kartica samo kada je broj neprijatelja nizak.

### 3. Kontrola (Action)
* **Prediktivno Nišanjenje:** Uračunava se offset i predviđanje kretanja meta kako bi se kompenzovala latencija obrade slike.
* **Sigurnosni Mehanizmi:** Implementirane "Blind Zone" i limiti kretanja kursora kako bi se sprečilo kliktanje van prozora igre.

---

## 📊 Rezultati i Performanse

Agent je testiran u realnom okruženju i postigao je **100% Win Rate** u finalnoj verziji.

| Metrika | Vrednost | Opis |
| :--- | :--- | :--- |
| **Precision** | **0.991** | Pouzdanost detekcije objekata |
| **Recall** | **0.959** | Pokrivenost (procenat detektovanih objekata) |
| **mAP@50** | **0.991** | Opšta tačnost modela |
| **FPS** | **30+** | Real-time obrada na laptop GPU/CPU |

---

## 🛠️ Tehnologije

* **Jezik:** Python
* **Obrada Slike:** OpenCV (`cv2`)
* **AI/ML:** Ultralytics YOLOv8
* **Screen Capture:** MSS
* **Input Simulacija:** PyAutoGUI

---

## ⚙️ Instalacija i Pokretanje

1. **Klonirajte repozitorijum:**
```bash
git clone https://github.com/IgorLudajic/pvz-whack-a-zombie-bot.git
cd pvz-whack-a-zombie-bot
```

2. **Instalacija biblioteka**
Instalirajte sve potrebne zavisnosti jednom komandom:
```bash
pip install ultralytics opencv-python mss pyautogui numpy
```

3. **Priprema modela**
Postavite vaš trenirani YOLO model `best.pt` u root folder projekta.

> **Napomena:** Ako nemate sopstveni model, skripta će zahtevati izmenu da koristi standardni `yolov8n.pt`, ali rezultati neće biti optimalni za ovu igru.

4. **Pokretanje**
Pokrenite igru **Plants vs. Zombies** i uđite na glavni meni, zatim pokrenite bota iz terminala
```bash
python main.py
```

---

## ⚠️ Rešavanje Čestih Problema (Troubleshooting)

**Problem:** Miš klikće van prozora igre

**Simptomi:** Kursor se nekontrolisano kreće ("ludi") po ekranu.

**Uzrok:** Razlika u rezoluciji monitora i Windows DPI skaliranju.

**Rešenje:**
1. Kliknite desnim tasterom na Desktop -> **Display settings**.
2. Pod sekcijom "Scale and layout", promenite vrednost sa **125%** (ili 150%) na **100%**.
3. Ponovo pokrenite skriptu.
