# Rezultati i metodologija (RL agent v2.0)

Dokument konsoliduje eksperimentalne rezultate za diplomski rad. Svi brojevi
su iz **simulatora** (sim-to-real trening); rezultati u pravoj igri beleže se
zasebno u `stats/game_stats.csv`.

## Metodologija evaluacije

Trening i selekcija checkpointa razdvojeni su od finalne ocene da bi se
izbegao overfitting na selekcioni set:

- **tokom treninga**: greedy evaluacija (bez istraživanja) na malom skupu
  (24-30 epizoda, fiksni seed) — dovoljno da vodi izbor najboljeg checkpointa;
- **finalna selekcija** (`tools/select_policy.py`): svi kandidati se mere na
  **nezavisnom test setu** (150 epizoda, seed 555 koji nije korišćen u
  treningu); pobednik po kombinovanoj oceni `win% − 5·kosačice` ide u
  `models/policy.pt`.

Simulator je **namerno teži od prave mini-igre** (agresivniji talasi, brži
zombiji, modelovan percepcijski šum YOLO-a i promašaji klika, domain
randomization po epizodi). Rezultati u simulatoru su zato donja granica
očekivanja u pravoj igri.

## Finalna tabela (test set: 150 epizoda, seed 555)

| Politika | Pristup | Win rate | Kosačice/partiji | Grave Buster | Cherry Bomb | Ice-shroom |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Heuristika v1.0 | ručna pravila | 100.0% | 0.19 | 7.9 | **0.0** | **0.0** |
| Pro-heuristika (demonstrator) | ručna pravila | 100.0% | 0.19 | 8.1 | 2.4 | 3.2 |
| DQN (najbolji, v10) | RL iz nagrade | 94.0% | 0.61 | — | — | — |
| **Behavior cloning (izabrana)** | **imitaciono učenje** | **99.3%** | **0.17** | 7.6 | 2.1 | 3.2 |

Potvrda izabrane politike na **drugom** nezavisnom setu (200 epizoda, seed 2024):
**100.0% (200/200)**, 0.23 kosačice/partiji, **87% partija bez ijedne kosačice**.

> **Tri paradigme, jedan zaključak.** Ručna heuristika postiže 100% jer je
> „savršen reaktor" sa instant prioritetom i potpunom informacijom — ali je
> ručno napisano pravilo i (u verziji v1.0) ne koristi biljke. DQN **uči iz
> signala nagrade**, bez ijednog eksplicitnog pravila, i samostalno otkriva
> upotrebu sve tri biljke; dostiže 94%, ali pati od nestabilnosti TD
> bootstrappinga (win rate osciluje i degradira posle vrhunca). Behavior
> cloning sa istom CNN arhitekturom uči da preslika stanje u ekspertovu akciju
> bez tog churn-a i dostiže nivo demonstratora (≈100%) — pa je izabran za
> izvršavanje. Raspodela akcija (`stats/games/*.json`, polje
> `action_distribution`) je direktan uvid u strategiju.

## Put razvoja modela (sažetak eksperimenata)

| Verzija | Ključna izmena | Ishod (greedy, tokom treninga) |
| :--- | :--- | :--- |
| v1–v2 | MLP, čist DQN | stagnacija na 0% — prostor od 138 akcija pretežak za slepo istraživanje |
| v3 | + seedovanje demonstracijama | prvi pobedi, ali urušavanje (demo pregažen u baferu) |
| v4 | + trajni demo bafer | stabilnije, ali i dalje ~10% |
| v5 | + DQfD margin gubitak | skok na 70% pa pad (MLP uči svaku ćeliju posebno) |
| v6 | + balansirano istraživanje, greedy selekcija | ~66% vrh, degradacija posle eps-minimuma |
| **v7** | **CNN nad mrežom 5×9 (deljenje težina)** | **91.7% već na 20k koraka** — arhitektura je bila usko grlo |
| v8–v10 | fino doterivanje (mali lr, jako demo sidro) | konsolidacija; nezavisni set 90–94% |
| **BC** | **behavior cloning (čista imitacija)** | **99.3% / 100%** — bez TD churn-a; izabrana politika |

Glavne pouke:
1. Zamena MLP-a konvolucionom mrežom koja deli težine preko table (pravilo
   „zombi u ćeliji → udari tu ćeliju" uči se jednom, ne 45 puta) bila je
   najveća pojedinačna poluga za DQN.
2. Za zadatak gde postoji kvalitetan demonstrator, čisto imitaciono učenje
   nadmašuje DQN po stabilnosti i krajnjem učinku — TD bootstrapping unosi
   oscilacije koje kod supervizovanog kloniranja ne postoje.

## Reprodukcija

```bash
python train_rl.py --steps 400000 --demo-steps 40000    # DQN trening (RL iz nagrade)
python train_bc.py --samples 150000 --epochs 10         # behavior cloning (izabrani pristup)
python tools/select_policy.py --episodes 150 --seed 555 # finalni izbor (presuđuje nezavisni set)
python evaluate_policy.py --episodes 200 --seed 2024            # potvrda izabrane politike
python evaluate_policy.py --episodes 150 --seed 555 --baseline v1
python evaluate_policy.py --episodes 150 --seed 555 --baseline pro
python tools/analyze_stats.py                           # analiza pravih partija
```
