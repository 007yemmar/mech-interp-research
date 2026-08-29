# Concordance validation — complete results

All cells: **metric %** with **(n)**. Bands are DISJOINT |r| intervals.
Judges: Claude Sonnet 4.6 (direct API) and DeepSeek-V3 (OpenRouter). Identical prompts and features; only the judge changes.
Sonnet concordance comes from the auto-interp stage (the verbatim Table-2 prompt); DeepSeek from arm0_eval.
All SAE/source evaluations are on HELD-OUT notes (shards 281-311).

## Table 1 — exact-YES (%)

The strict verdict: the explanation describes the ICD concept.

| Source | Type | Judge | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6+ |
|---|---|---|---|---|---|---|---|---|
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | Sonnet | 0 (80) | 0 (20) | – | 18 (136) | 30 (84) | 60 (60) |
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | DeepSeek | 0 (80) | 0 (20) | – | 21 (136) | 30 (84) | 60 (60) |
| SAE jumprelu (strat 200) | SAE (domain, stratified) | Sonnet | – | 0 (40) | 8 (40) | 30 (40) | 30 (40) | 68 (40) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | Sonnet | 0 (86) | 0 (14) | – | 15 (137) | 33 (70) | 60 (73) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | DeepSeek | 0 (86) | 0 (14) | – | 18 (137) | 41 (70) | 64 (73) |
| SAE gemmascope (113) | SAE (general-purpose) | Sonnet | 0 (97) | 0 (3) | – | 11 (9) | 0 (4) | – |
| SAE gemmascope (113) | SAE (general-purpose) | DeepSeek | 0 (97) | 0 (3) | – | 33 (9) | 0 (4) | – |
| keyword (39) | lexical | Sonnet | 50 (2) | 75 (4) | 85 (13) | 85 (13) | 100 (3) | 100 (3) |
| keyword (39) | lexical | DeepSeek | 0 (2) | 75 (4) | 69 (13) | 85 (13) | 67 (3) | 100 (3) |
| random (300) | random directions | Sonnet | 2 (54) | 2 (230) | 0 (15) | 0 (1) | – | – |
| random (300) | random directions | DeepSeek | 0 (54) | 2 (230) | 0 (15) | 0 (1) | – | – |
| diff-in-means (46) | supervised | Sonnet | 0 (4) | 7 (14) | 10 (10) | 27 (11) | 20 (5) | 0 (2) |
| diff-in-means (46) | supervised | DeepSeek | 0 (4) | 14 (14) | 20 (10) | 36 (11) | 20 (5) | 50 (2) |
| probe LR (46) | supervised | Sonnet | 0 (2) | 13 (15) | 7 (14) | 20 (10) | 33 (3) | 0 (2) |
| probe LR (46) | supervised | DeepSeek | 0 (2) | 13 (15) | 7 (14) | 20 (10) | 33 (3) | 0 (2) |
| PCA (35) | unsupervised | Sonnet | 0 (26) | 0 (8) | – | 100 (1) | – | – |
| PCA (35) | unsupervised | DeepSeek | 4 (26) | 0 (8) | – | 100 (1) | – | – |

## Table 2 — YES+PARTIAL (%)

The pooled verdict as reported in the original Table 2.

| Source | Type | Judge | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6+ |
|---|---|---|---|---|---|---|---|---|
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | Sonnet | 52 (80) | 80 (20) | – | 90 (136) | 98 (84) | 100 (60) |
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | DeepSeek | 86 (80) | 100 (20) | – | 98 (136) | 100 (84) | 100 (60) |
| SAE jumprelu (strat 200) | SAE (domain, stratified) | Sonnet | – | 78 (40) | 95 (40) | 88 (40) | 98 (40) | 100 (40) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | Sonnet | 52 (86) | 71 (14) | – | 96 (137) | 96 (70) | 100 (73) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | DeepSeek | 87 (86) | 93 (14) | – | 100 (137) | 100 (70) | 100 (73) |
| SAE gemmascope (113) | SAE (general-purpose) | Sonnet | 20 (97) | 100 (3) | – | 67 (9) | 75 (4) | – |
| SAE gemmascope (113) | SAE (general-purpose) | DeepSeek | 71 (97) | 100 (3) | – | 78 (9) | 100 (4) | – |
| keyword (39) | lexical | Sonnet | 100 (2) | 100 (4) | 100 (13) | 100 (13) | 100 (3) | 100 (3) |
| keyword (39) | lexical | DeepSeek | 100 (2) | 100 (4) | 100 (13) | 100 (13) | 100 (3) | 100 (3) |
| random (300) | random directions | Sonnet | 78 (54) | 80 (230) | 100 (15) | 100 (1) | – | – |
| random (300) | random directions | DeepSeek | 93 (54) | 93 (230) | 100 (15) | 100 (1) | – | – |
| diff-in-means (46) | supervised | Sonnet | 25 (4) | 36 (14) | 60 (10) | 45 (11) | 60 (5) | 50 (2) |
| diff-in-means (46) | supervised | DeepSeek | 50 (4) | 57 (14) | 70 (10) | 64 (11) | 80 (5) | 100 (2) |
| probe LR (46) | supervised | Sonnet | 0 (2) | 27 (15) | 36 (14) | 40 (10) | 100 (3) | 100 (2) |
| probe LR (46) | supervised | DeepSeek | 100 (2) | 67 (15) | 50 (14) | 40 (10) | 100 (3) | 100 (2) |
| PCA (35) | unsupervised | Sonnet | 23 (26) | 62 (8) | – | 100 (1) | – | – |
| PCA (35) | unsupervised | DeepSeek | 88 (26) | 100 (8) | – | 100 (1) | – | – |

## Table 3 — NO (%)

Explicit rejection rate.

| Source | Type | Judge | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6+ |
|---|---|---|---|---|---|---|---|---|
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | Sonnet | 44 (80) | 20 (20) | – | 4 (136) | 1 (84) | 0 (60) |
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | DeepSeek | 14 (80) | 0 (20) | – | 2 (136) | 0 (84) | 0 (60) |
| SAE jumprelu (strat 200) | SAE (domain, stratified) | Sonnet | – | 22 (40) | 2 (40) | 5 (40) | 0 (40) | 0 (40) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | Sonnet | 37 (86) | 29 (14) | – | 3 (137) | 0 (70) | 0 (73) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | DeepSeek | 13 (86) | 7 (14) | – | 0 (137) | 0 (70) | 0 (73) |
| SAE gemmascope (113) | SAE (general-purpose) | Sonnet | 76 (97) | 0 (3) | – | 22 (9) | 25 (4) | – |
| SAE gemmascope (113) | SAE (general-purpose) | DeepSeek | 28 (97) | 0 (3) | – | 22 (9) | 0 (4) | – |
| keyword (39) | lexical | Sonnet | 0 (2) | 0 (4) | 0 (13) | 0 (13) | 0 (3) | 0 (3) |
| keyword (39) | lexical | DeepSeek | 0 (2) | 0 (4) | 0 (13) | 0 (13) | 0 (3) | 0 (3) |
| random (300) | random directions | Sonnet | 22 (54) | 20 (230) | 0 (15) | 0 (1) | – | – |
| random (300) | random directions | DeepSeek | 7 (54) | 7 (230) | 0 (15) | 0 (1) | – | – |
| diff-in-means (46) | supervised | Sonnet | 75 (4) | 64 (14) | 40 (10) | 45 (11) | 40 (5) | 50 (2) |
| diff-in-means (46) | supervised | DeepSeek | 50 (4) | 43 (14) | 30 (10) | 36 (11) | 20 (5) | 0 (2) |
| probe LR (46) | supervised | Sonnet | 100 (2) | 73 (15) | 64 (14) | 50 (10) | 0 (3) | 0 (2) |
| probe LR (46) | supervised | DeepSeek | 0 (2) | 33 (15) | 50 (14) | 60 (10) | 0 (3) | 0 (2) |
| PCA (35) | unsupervised | Sonnet | 69 (26) | 25 (8) | – | 0 (1) | – | – |
| PCA (35) | unsupervised | DeepSeek | 12 (26) | 0 (8) | – | 0 (1) | – | – |

## Table 4 — hit@1 (%)

Blind forced choice: 1 grounded code + 7 cross-system distractors + 'none'. Chance = 11.1%.

| Source | Type | Judge | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6+ |
|---|---|---|---|---|---|---|---|---|
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | Sonnet | 15 (80) | 25 (20) | – | 85 (136) | 93 (84) | 100 (60) |
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | DeepSeek | 11 (80) | 20 (20) | – | 86 (136) | 93 (84) | 97 (60) |
| SAE jumprelu (strat 200) | SAE (domain, stratified) | Sonnet | – | 28 (40) | 75 (40) | 75 (40) | 90 (40) | 98 (40) |
| SAE jumprelu (strat 200) | SAE (domain, stratified) | DeepSeek | – | 35 (40) | 88 (40) | 85 (40) | 95 (40) | 100 (40) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | Sonnet | 10 (86) | 43 (14) | – | 88 (137) | 96 (70) | 100 (73) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | DeepSeek | 19 (86) | 50 (14) | – | 90 (137) | 96 (70) | 99 (73) |
| SAE gemmascope (113) | SAE (general-purpose) | Sonnet | 2 (97) | 0 (3) | – | 67 (9) | 75 (4) | – |
| SAE gemmascope (113) | SAE (general-purpose) | DeepSeek | 3 (97) | 0 (3) | – | 67 (9) | 100 (4) | – |
| keyword (39) | lexical | Sonnet | 100 (2) | 100 (4) | 92 (13) | 100 (13) | 100 (3) | 100 (3) |
| keyword (39) | lexical | DeepSeek | 100 (2) | 100 (4) | 100 (13) | 85 (13) | 100 (3) | 100 (3) |
| random (300) | random directions | Sonnet | 28 (54) | 36 (230) | 67 (15) | 100 (1) | – | – |
| random (300) | random directions | DeepSeek | 30 (54) | 37 (230) | 80 (15) | 100 (1) | – | – |
| diff-in-means (46) | supervised | Sonnet | 25 (4) | 29 (14) | 30 (10) | 27 (11) | 40 (5) | 50 (2) |
| diff-in-means (46) | supervised | DeepSeek | 0 (4) | 36 (14) | 40 (10) | 36 (11) | 40 (5) | 50 (2) |
| probe LR (46) | supervised | Sonnet | 0 (2) | 13 (15) | 14 (14) | 30 (10) | 33 (3) | 0 (2) |
| probe LR (46) | supervised | DeepSeek | 0 (2) | 20 (15) | 14 (14) | 30 (10) | 67 (3) | 50 (2) |
| PCA (35) | unsupervised | Sonnet | 8 (26) | 38 (8) | – | 100 (1) | – | – |
| PCA (35) | unsupervised | DeepSeek | 12 (26) | 12 (8) | – | 100 (1) | – | – |

## Table 5 — 'none' rate (%)

How often the judge declines to pick any code — a direct read on describability.

| Source | Type | Judge | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6+ |
|---|---|---|---|---|---|---|---|---|
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | Sonnet | 79 (80) | 65 (20) | – | 15 (136) | 7 (84) | 0 (60) |
| SAE jumprelu (pub 380) | SAE (domain, JumpReLU) | DeepSeek | 79 (80) | 70 (20) | – | 11 (136) | 5 (84) | 2 (60) |
| SAE jumprelu (strat 200) | SAE (domain, stratified) | Sonnet | – | 72 (40) | 25 (40) | 25 (40) | 10 (40) | 2 (40) |
| SAE jumprelu (strat 200) | SAE (domain, stratified) | DeepSeek | – | 60 (40) | 10 (40) | 5 (40) | 2 (40) | 0 (40) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | Sonnet | 86 (86) | 57 (14) | – | 12 (137) | 4 (70) | 0 (73) |
| SAE vanilla (380) | SAE (domain, ReLU+L1) | DeepSeek | 74 (86) | 50 (14) | – | 7 (137) | 0 (70) | 0 (73) |
| SAE gemmascope (113) | SAE (general-purpose) | Sonnet | 95 (97) | 100 (3) | – | 33 (9) | 25 (4) | – |
| SAE gemmascope (113) | SAE (general-purpose) | DeepSeek | 89 (97) | 67 (3) | – | 33 (9) | 0 (4) | – |
| keyword (39) | lexical | Sonnet | 0 (2) | 0 (4) | 8 (13) | 0 (13) | 0 (3) | 0 (3) |
| keyword (39) | lexical | DeepSeek | 0 (2) | 0 (4) | 0 (13) | 0 (13) | 0 (3) | 0 (3) |
| random (300) | random directions | Sonnet | 67 (54) | 59 (230) | 27 (15) | 0 (1) | – | – |
| random (300) | random directions | DeepSeek | 54 (54) | 53 (230) | 13 (15) | 0 (1) | – | – |
| diff-in-means (46) | supervised | Sonnet | 75 (4) | 71 (14) | 70 (10) | 73 (11) | 40 (5) | 50 (2) |
| diff-in-means (46) | supervised | DeepSeek | 50 (4) | 64 (14) | 60 (10) | 55 (11) | 40 (5) | 50 (2) |
| probe LR (46) | supervised | Sonnet | 100 (2) | 80 (15) | 79 (14) | 70 (10) | 67 (3) | 100 (2) |
| probe LR (46) | supervised | DeepSeek | 100 (2) | 73 (15) | 79 (14) | 60 (10) | 33 (3) | 0 (2) |
| PCA (35) | unsupervised | Sonnet | 88 (26) | 62 (8) | – | 0 (1) | – | – |
| PCA (35) | unsupervised | DeepSeek | 88 (26) | 62 (8) | – | 0 (1) | – | – |

## Table 6 — inter-judge divergence |Sonnet − DeepSeek|, percentage points

| Source | Metric | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6+ |
|---|---|---|---|---|---|---|---|
| SAE jumprelu (pub 380) | exact-YES | 0 | 0 | – | 3 | 0 | 0 |
| SAE jumprelu (pub 380) | YES+PARTIAL | 34 | 20 | – | 8 | 2 | 0 |
| SAE jumprelu (pub 380) | hit@1 | 4 | 5 | – | 1 | 0 | 3 |
| SAE jumprelu (strat 200) | hit@1 | – | 7 | 13 | 10 | 5 | 2 |
| SAE vanilla (380) | exact-YES | 0 | 0 | – | 3 | 8 | 4 |
| SAE vanilla (380) | YES+PARTIAL | 35 | 22 | – | 4 | 4 | 0 |
| SAE vanilla (380) | hit@1 | 9 | 7 | – | 2 | 0 | 1 |
| SAE gemmascope (113) | exact-YES | 0 | 0 | – | 22 | 0 | – |
| SAE gemmascope (113) | YES+PARTIAL | 51 | 0 | – | 11 | 25 | – |
| SAE gemmascope (113) | hit@1 | 1 | 0 | – | 0 | 25 | – |
| keyword (39) | exact-YES | 50 | 0 | 16 | 0 | 33 | 0 |
| keyword (39) | YES+PARTIAL | 0 | 0 | 0 | 0 | 0 | 0 |
| keyword (39) | hit@1 | 0 | 0 | 8 | 15 | 0 | 0 |
| random (300) | exact-YES | 2 | 0 | 0 | 0 | – | – |
| random (300) | YES+PARTIAL | 15 | 13 | 0 | 0 | – | – |
| random (300) | hit@1 | 2 | 1 | 13 | 0 | – | – |
| diff-in-means (46) | exact-YES | 0 | 7 | 10 | 9 | 0 | 50 |
| diff-in-means (46) | YES+PARTIAL | 25 | 21 | 10 | 19 | 20 | 50 |
| diff-in-means (46) | hit@1 | 25 | 7 | 10 | 9 | 0 | 0 |
| probe LR (46) | exact-YES | 0 | 0 | 0 | 0 | 0 | 0 |
| probe LR (46) | YES+PARTIAL | 100 | 40 | 14 | 0 | 0 | 0 |
| probe LR (46) | hit@1 | 0 | 7 | 0 | 0 | 34 | 50 |
| PCA (35) | exact-YES | 4 | 0 | – | 0 | – | – |
| PCA (35) | YES+PARTIAL | 65 | 38 | – | 0 | – | – |
| PCA (35) | hit@1 | 4 | 26 | – | 0 | – | – |
