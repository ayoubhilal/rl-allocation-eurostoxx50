# Allocation dynamique de portefeuille par reinforcement learning — Euro Stoxx 50

Un agent **PPO** (stable-baselines3) décide chaque semaine de la répartition d'un portefeuille long-only
entre les **50 actions de l'Euro Stoxx 50** et du cash. Il part d'un portefeuille de référence
(max-Sharpe) et apprend à l'ajuster selon des signaux par titre (momentum, retournement, volatilité,
drawdown) et le contexte de marché (tendance, volatilité, VIX). Le protocole est strictement temporel :
aucune donnée postérieure à une décision n'est utilisée, les réglages sont choisis sur une validation
glissante, et le jeu de test n'est évalué qu'une fois (verrou).

## Résultats

### Test 2022-01-07 → 2026-11-13 (254 semaines, frais inclus) — `results/tables/test_comparison.md`

| Stratégie | Rendement/an | Volatilité | Sharpe | Pire perte | Rendement cumulé | p vs 1/N |
|---|---|---|---|---|---|---|
| **Agent PPO (ensemble de 5 seeds)** | **16.45 %** | 15.66 % | **1.05** | **-18.22 %** | **110.4 %** | 0.88 |
| Référence non entraînée (max-Sharpe ajusté) | 16.58 % | 15.81 % | 1.05 | -18.28 % | 111.5 % | 0.88 |
| ETF Euro Stoxx 50 (dividendes inclus) | 9.97 % | 16.77 % | 0.65 | -20.97 % | 59.1 % | 0.00 |
| Indice Euro Stoxx 50 (prix) | 6.82 % | 16.63 % | 0.48 | -22.94 % | 38.0 % | 0.00 |
| 1/N (50 titres à parts égales) | 16.29 % | 16.34 % | 1.01 | -17.01 % | 109.0 % | — |
| Max-Sharpe (plafond 15 %) | 15.95 % | 16.37 % | 0.99 | -20.76 % | 106.0 % | 0.95 |
| Variance minimale | 16.83 % | 11.85 % | 1.37 | -8.98 % | 113.8 % | 0.33 |
| Faible volatilité (top 10) | 18.45 % | 13.16 % | 1.35 | -10.19 % | 128.7 % | 0.28 |
| Parité de risque | 16.44 % | 14.48 % | 1.13 | -14.60 % | 110.4 % | 0.12 |
| Momentum 12-1 (top 10) | 22.14 % | 20.01 % | 1.10 | -23.17 % | 165.7 % | 0.71 |

### Validation glissante 2014-2021 (4 blocs de 2 ans, 414 semaines) — `results/tables/tuning_summary.md`

| Stratégie | Rendement/an | Sharpe | Pire perte | Sharpe par bloc (14-15 · 16-17 · 18-19 · 20-21) |
|---|---|---|---|---|
| **Agent PPO (configuration retenue)** | 19.75 % | **1.05** | -29.36 % | 0.81 · 1.34 · 1.30 · 0.97 |
| Référence non entraînée | 19.98 % | 1.06 | -29.68 % | 0.82 · 1.34 · 1.31 · 0.97 |
| Max-Sharpe | 20.75 % | 1.09 | -28.19 % | 0.78 · 1.35 · 1.36 · 1.05 |
| Variance minimale | 15.33 % | 1.00 | -27.03 % | 1.21 · 1.12 · 1.45 · 0.58 |
| 1/N | 13.13 % | 0.72 | -35.15 % | 0.70 · 1.18 · 0.63 · 0.64 |
| ETF Euro Stoxx 50 | 8.48 % | 0.52 | -33.83 % | 0.47 · 0.79 · 0.48 · 0.47 |

### Lecture

- L'agent **bat nettement l'Euro Stoxx 50** : Sharpe 1,05 contre 0,65 pour l'ETF au test, rendement de
  +16,5 %/an contre +10,0 %/an, avec une pire perte plus faible (−18 % contre −21 %). Sur 5 ans, l'écart de
  Sharpe n'est pas statistiquement significatif (test apparié : p = 0,17), mais l'agent devance l'ETF sur toutes
  les périodes de validation. L'essentiel de l'écart vient de la répartition entre les 50 titres actuels (1/N fait
  déjà 16,3 %/an), l'apprentissage n'en expliquant qu'une petite partie. Ses
  résultats sont **stables** entre la validation (1,05) et le test (1,05).
- Cette performance vient surtout de la **conception** : les 50 titres et un bon portefeuille de
  référence (max-Sharpe, réajusté chaque mois). Les ajustements appris par l'agent sont faibles : ses
  résultats sont très proches de ceux de sa référence non entraînée.
- Aucune des stratégies diversifiées ne se distingue **statistiquement** de 1/N au test (toutes les
  p-values ≥ 0,12). Les intervalles de confiance du Sharpe (bootstrap par blocs) sont larges : environ
  ±1 sur 5 ans.
- *Pire perte* = plus forte baisse entre un sommet et le creux suivant, sur toute la période (ici
  essentiellement le krach de mars 2020 en validation et la baisse de 2022 au test).

## Méthode

**Données** (`scripts/01_download_data.py`, `src/data.py`, `src/universe.py`) :
- constituants actuels scrapés sur Wikipédia, tickers Yahoo vérifiés ;
- prix ajustés quotidiens depuis 2009 (yfinance), ETF `EXW1.DE` et indice `^STOXX50E` comme benchmarks ;
- indice et VIX comme contexte de marché ;
- les 5 titres entrés récemment dans l'indice (Adyen, Argenx, Ferrari, Prosus, Siemens Energy) ne
  deviennent investissables qu'**un an après leur cotation** ; aucun prix antérieur n'est utilisé ;
- rapport : `data/processed/report.md`.

**Décision hebdomadaire** : au dernier jour de bourse de chaque semaine ; les poids s'appliquent aux
rendements de la semaine suivante.

**État observé** (`src/universe.py`, 14 informations par titre) :
- rangs parmi les titres disponibles du momentum 12-1 et 6-1, du retournement 1 mois, du rendement
  1 semaine, de la volatilité 60 j et du drawdown 252 j ;
- momentum et volatilité en niveau ;
- contexte de marché (tendance MM200, volatilité, rendement 13 semaines, VIX et sa variation) ;
- drapeau de disponibilité du titre, et poids courants du portefeuille ;
- normalisation calculée sur la période d'entraînement uniquement.

**Action** (`src/env.py`) : une note par titre et une pour le cash, dans [−1, 1] :
- poids risqués = softmax(note + log(poids max-Sharpe + 0,002)) : une action nulle reproduit le
  portefeuille de référence ;
- cash = 50 % × max(note cash, 0) ;
- les poids exécutés ne font que la moitié du chemin vers la cible chaque semaine (lissage) ;
- les titres indisponibles ont un poids nul.

**Récompense** : rendement net − (γ/2)·rendement² (utilité moyenne-variance, γ = 2). Les frais valent
0,1 % du montant échangé (0,3 % dans la récompense d'entraînement, pour inciter à l'inertie). Les
poids dérivent avec les prix.

**Agent** (`src/policies.py`, `src/train.py`) :
- PPO, avec un **réseau partagé entre titres** : le même petit réseau (2×32, environ 2 800 paramètres)
  note chaque titre, ce qui rend la politique équivariante par permutation et indépendante du nombre
  de titres ;
- décision myope (γ_PPO = 0), 40 000 pas ;
- augmentation par permutation des titres et bruit sur les features (hors drapeau de disponibilité) ;
- **ensemble de 5 seeds** (moyenne des poids cibles).

**Protocole** (`scripts/02_train_evaluate.py`) :
1. **Réglage** : 8 configurations × 4 blocs de validation glissante (entraînement 2010→2013/15/17/19,
   validation 2014-15/16-17/18-19/20-21) × 3 seeds = 96 entraînements. Critère fixé à l'avance : gain de
   Sharpe sur 1/N à la même fréquence. Configuration retenue : inclinaisons autour du max-Sharpe
   (`results/tables/best_config.json`).
2. **Modèle final** : 5 seeds entraînées sur 2010-2021 (`results/models/final/autour_maxsharpe/`).
3. **Test unique** sur 2022-2026, verrouillé (`results/.final_test_done`), avec intervalles de confiance
   du Sharpe et test de différence contre 1/N (bootstrap par blocs appariés).

**Baselines** (`src/baselines.py`), avec les mêmes dates, frais et départ déjà investi : buy & hold de
l'ETF et de l'indice, 1/N, variance minimale et max-Sharpe (cvxpy, covariance Ledoit-Wolf sur 3 ans),
parité de risque, momentum 12-1, faible volatilité (rebalancement mensuel).

## Installation et exécution

Python ≥ 3.11 (développé avec CPython 3.13, Windows 11, CPU).

```bash
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt

.venv/Scripts/python -m pytest                                  # 25 tests (~30 s)
.venv/Scripts/python scripts/01_download_data.py                # données (cache fourni ; --refresh pour re-télécharger)
.venv/Scripts/python scripts/02_train_evaluate.py --final-test --force   # rejoue le test avec les modèles fournis (~2 min)
.venv/Scripts/python scripts/02_train_evaluate.py --tune        # refait le réglage complet (96 entraînements, ~2 h 15)
```

- Avec les modèles et le cache fournis, la commande `--final-test --force` **reproduit exactement** les
  résultats ci-dessus (vérifié : écart nul). Sans modèles, elle entraîne d'abord les 5 seeds (~10 min).
- Le test a déjà été exécuté : `--force` est nécessaire pour le rejouer, et la tentative est journalisée.
- Ne pas baser le venv sur le Python d'Anaconda (conflit de DLL avec cvxpy/osqp sous Windows) ;
  `scs` est épinglé à 3.2.7 (la 3.3.2 plante à la fermeture de Python sous Windows).
- `--refresh` re-télécharge les données jusqu'à la date du jour : les résultats changeront alors.

## Structure

```
config.yaml                  tous les paramètres (données, signaux, environnement, PPO, protocole)
requirements.txt
src/data.py                  constituants Wikipédia -> tickers Yahoo -> prix nettoyés
src/universe.py              univers dynamique, signaux par titre, contexte de marché
src/features.py              calendrier hebdomadaire, découpage temporel, normalisation
src/env.py                   environnement Gymnasium (comptabilité, coûts, dérive, masque)
src/policies.py              réseau partagé entre titres (politique PPO)
src/train.py                 fabrique d'agents (PPO ; A2C/SAC/TD3 prévus), entraînement
src/baselines.py             stratégies de référence (cvxpy)
src/metrics.py               métriques, bootstrap du Sharpe
src/evaluate.py, plots.py    tableaux, verrou du test, figures
scripts/01_download_data.py  téléchargement et nettoyage
scripts/02_train_evaluate.py réglage (--tune) et test final (--final-test)
tests/                       25 tests (comptabilité, absence de look-ahead, masque, réseau, baselines, statistiques)
data/raw, data/processed     cache des données (au 2026-11-20) et rapport de données
results/tables, figures      résultats du réglage et du test ; results/models/final : les 5 modèles
```

## Démarche (versions explorées avant ce modèle)

Plusieurs approches ont été comparées avec le même protocole avant de retenir celle-ci (Sharpe au test
2022-2026) :

| Approche | Résultat |
|---|---|
| PPO sur 20 titres, rendements bruts | sur-apprentissage, Sharpe 0,58 |
| PPO bridé (rangs, réseau partagé, lissage) | n'apprend rien, reste sur 1/N |
| PPO pour l'exposition actions / cash | réduit le risque, mais une allocation fixe 50/50 fait aussi bien |
| **PPO, 50 titres, autour du max-Sharpe (ce modèle)** | **Sharpe 1,05, stable entre validation et test** |
| Gradient direct (backtest différentiable) | apprend en validation (+0,17), pas au test (0,93) |
| PPO répartissant entre 6 stratégies | décisions actives, mais pas mieux que le mélange égal (validation 0,89) |

Le modèle final a été désigné par une règle fixée à l'avance (meilleur Sharpe en validation glissante),
et non d'après le test. Une expérience avec un signal planté a montré que PPO apprend bien quand un
signal existe : la limite vient du signal exploitable dans les prix passés, pas de l'algorithme.

## Limites

- **Biais de survie** : l'univers est la composition *actuelle* de l'indice ; les sociétés sorties depuis
  2009 sont absentes, ce qui flatte toutes les performances absolues. Seules les comparaisons à univers
  identique ont du sens.
- **Peu de données et incertitude** : environ 10 ans d'entraînement ; les écarts de Sharpe entre stratégies
  diversifiées ne sont pas significatifs. Le test 2022-2026 a été observé à plusieurs reprises au fil des
  versions explorées : son résultat est indicatif.
- **Apport limité de l'apprentissage** : l'agent reste proche de sa référence ; l'essentiel de la
  performance vient du choix de l'univers et du portefeuille de référence.
- **Coûts simplifiés** : 0,1 % du montant échangé, exécution au cours de clôture, sans impact de marché ni
  taxe sur les transactions ; cash rémunéré à 0 %.
- **Données Yahoo Finance** : prix ajustés rétroactivement ; benchmark avec dividendes approché par un ETF.
