# Rapport de données

- Généré le : 2026-11-23
- Source des constituants : **wikipedia** (https://en.wikipedia.org/wiki/EURO_STOXX_50)
- Téléchargement Yahoo Finance (auto_adjust=True) : 2009-01-01 -> 2026-11-20
- Benchmarks : ^STOXX50E (indice prix, dividendes non réinvestis) et EXW1.DE (ETF Euro Stoxx 50, prix ajustés = dividendes inclus, frais ~0,1 %/an)
- Période couverte après nettoyage : 2009-01-05 -> 2026-11-20
- Lignes : prix = 4583, rendements = 4582 ; colonnes = 46 (45 titres + CASH)
- Jours supprimés (valeurs manquantes après ffill <= 3 j) : 1
- Actif CASH : rendement quotidien constant = 0.0

## Résolution des tickers

| Société | Ticker Yahoo | Candidats | Échecs |
|---|---|---|---|
| Adidas | ADS.DE | ADS.DE |  |
| Adyen | ADYEN.AS | ADYEN.AS |  |
| Ahold Delhaize | AD.AS | AD.AS |  |
| Air Liquide | AI.PA | AI.PA |  |
| Airbus | AIR.PA | AIR.PA |  |
| Allianz | ALV.DE | ALV.DE |  |
| Anheuser-Busch InBev | ABI.BR | ABI.BR |  |
| Argenx | ARGX.BR | ARGX.BR |  |
| ASML Holding | ASML.AS | ASML.AS |  |
| Axa | CS.PA | CS.PA |  |
| BASF | BAS.DE | BAS.DE |  |
| Bayer | BAYN.DE | BAYN.DE |  |
| BBVA | BBVA.MC | BBVA.MC |  |
| Banco Santander | SAN.MC | SAN.MC |  |
| BMW | BMW.DE | BMW.DE |  |
| BNP Paribas | BNP.PA | BNP.PA |  |
| Danone | BN.PA | BN.PA |  |
| Deutsche Bank | DBK.DE | DBK.DE |  |
| Deutsche Börse | DB1.DE | DB1.DE |  |
| Deutsche Post | DHL.DE | DHL.DE |  |
| Deutsche Telekom | DTE.DE | DTE.DE |  |
| Enel | ENEL.MI | ENEL.MI |  |
| Engie | ENGI.PA | ENGI.PA |  |
| Eni | ENI.MI | ENI.MI |  |
| EssilorLuxottica | EL.PA | EL.PA |  |
| Ferrari | RACE.MI | RACE.MI |  |
| Hermès | RMS.PA | RMS.PA |  |
| Iberdrola | IBE.MC | IBE.MC |  |
| Inditex | ITX.MC | ITX.MC |  |
| Infineon Technologies | IFX.DE | IFX.DE |  |
| ING Group | INGA.AS | INGA.AS |  |
| Intesa Sanpaolo | ISP.MI | ISP.MI |  |
| L'Oréal | OR.PA | OR.PA |  |
| LVMH Moët Hennessy Louis Vuitton | MC.PA | MC.PA |  |
| Mercedes-Benz Group | MBG.DE | MBG.DE |  |
| Munich Re | MUV2.DE | MUV2.DE |  |
| Nokia | NOKIA.HE | NOKIA.HE |  |
| Nordea | NDA-FI.HE | NDA-FI.HE |  |
| Prosus | PRX.AS | PRX.AS |  |
| Rheinmetall | RHM.DE | RHM.DE |  |
| Safran | SAF.PA | SAF.PA |  |
| Saint-Gobain | SGO.PA | SGO.PA |  |
| Sanofi | SAN.PA | SAN.PA |  |
| SAP | SAP.DE | SAP.DE |  |
| Schneider Electric | SU.PA | SU.PA |  |
| Siemens | SIE.DE | SIE.DE |  |
| Siemens Energy | ENR.DE | ENR.DE |  |
| TotalEnergies | TTE.PA | TTE.PA |  |
| Vinci SA | DG.PA | DG.PA |  |
| UniCredit | UCG.MI | UCG.MI |  |

## Univers retenu (top 50 par valeur échangée moyenne, train 2010-01-01 -> 2019-12-31)

| Rang | Ticker | Société | Valeur échangée moyenne (M EUR/jour) |
|---|---|---|---|
| 1 | SAN.MC | Banco Santander | 319.7 |
| 2 | UCG.MI | UniCredit | 250.2 |
| 3 | BBVA.MC | BBVA | 184.2 |
| 4 | DBK.DE | Deutsche Bank | 176.4 |
| 5 | SIE.DE | Siemens | 168.1 |
| 6 | SAP.DE | SAP | 159.6 |
| 7 | BAYN.DE | Bayer | 157.6 |
| 8 | ISP.MI | Intesa Sanpaolo | 134.6 |
| 9 | TTE.PA | TotalEnergies | 133.9 |
| 10 | ALV.DE | Allianz | 128.2 |
| 11 | ITX.MC | Inditex | 127.3 |
| 12 | SAN.PA | Sanofi | 126.5 |
| 13 | MBG.DE | Mercedes-Benz Group | 119.9 |
| 14 | BNP.PA | BNP Paribas | 118.8 |
| 15 | BAS.DE | BASF | 116.7 |
| 16 | ENI.MI | Eni | 115.0 |
| 17 | MC.PA | LVMH Moët Hennessy Louis Vuitton | 111.2 |
| 18 | ASML.AS | ASML Holding | 110.0 |
| 19 | INGA.AS | ING Group | 107.7 |
| 20 | ABI.BR | Anheuser-Busch InBev | 107.0 |
| 21 | IBE.MC | Iberdrola | 102.3 |
| 22 | DTE.DE | Deutsche Telekom | 100.3 |
| 23 | AIR.PA | Airbus | 95.5 |
| 24 | ENEL.MI | Enel | 84.1 |
| 25 | ADS.DE | Adidas | 82.9 |
| 26 | BMW.DE | BMW | 80.7 |
| 27 | NOKIA.HE | Nokia | 79.5 |
| 28 | OR.PA | L'Oréal | 77.2 |
| 29 | SU.PA | Schneider Electric | 76.9 |
| 30 | CS.PA | Axa | 75.7 |
| 31 | BN.PA | Danone | 71.3 |
| 32 | IFX.DE | Infineon Technologies | 71.2 |
| 33 | AI.PA | Air Liquide | 67.8 |
| 34 | MUV2.DE | Munich Re | 67.6 |
| 35 | DG.PA | Vinci SA | 60.9 |
| 36 | DHL.DE | Deutsche Post | 56.6 |
| 37 | SGO.PA | Saint-Gobain | 55.7 |
| 38 | SAF.PA | Safran | 50.5 |
| 39 | EL.PA | EssilorLuxottica | 45.0 |
| 40 | ENGI.PA | Engie | 44.1 |
| 41 | AD.AS | Ahold Delhaize | 41.5 |
| 42 | DB1.DE | Deutsche Börse | 35.8 |
| 43 | RMS.PA | Hermès | 17.3 |
| 44 | RHM.DE | Rheinmetall | 9.7 |
| 45 | NDA-FI.HE | Nordea | 6.1 |

## Titres retirés au nettoyage

| Ticker | Société | Raison |
|---|---|---|
| ADYEN.AS | Adyen | historique incomplet : 1re cotation le 2018-06-13 (> 2009-01-11) |
| ARGX.BR | Argenx | historique incomplet : 1re cotation le 2014-07-10 (> 2009-01-11) |
| RACE.MI | Ferrari | historique incomplet : 1re cotation le 2016-01-04 (> 2009-01-11) |
| PRX.AS | Prosus | historique incomplet : 1re cotation le 2019-09-11 (> 2009-01-11) |
| ENR.DE | Siemens Energy | historique incomplet : 1re cotation le 2020-09-29 (> 2009-01-11) |

## Titres valides mais hors univers (liquidité inférieure)

Aucun.

## Sociétés sans ticker Yahoo valide

Aucune.

## Avertissement : biais de survie

L'univers part de la composition **actuelle** de l'indice : les sociétés sorties de l'indice depuis 2009 sont absentes, ce qui biaise les performances historiques à la hausse.
