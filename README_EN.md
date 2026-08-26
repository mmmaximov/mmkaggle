<div align="center">

# mmkaggle

![Competitions](https://img.shields.io/badge/competitions-3-blue.svg)
![Best result](https://img.shields.io/badge/best_result-top_10%25-brightgreen.svg)
![Tracks](https://img.shields.io/badge/tracks-4-lightgrey.svg)

*A repository of notebooks and solutions from the ML competitions I've taken part in.*

</div>

---

## Contents

- [Results at a glance](#results-at-a-glance)
- [Yandex ML Challenge 2026. Long Tour](#yandex-ml-challenge-2026-long-tour)
  - [A. Adaptive Puzzle Solving Challenge](#a-adaptive-puzzle-solving-challenge)
  - [B. Novel View Synthesis](#b-novel-view-synthesis)
  - [C. Effective Inference On School Questions](#c-effective-inference-on-school-questions)
- [RWB WildHack](#rwb-wildhack)
  - [Uninterrupted Shipments (Solo track)](#uninterrupted-shipments-solo-track)
  - [Warehouse Logistics Automation (Team track)](#warehouse-logistics-automation-team-track)
- [Repository structure](#repository-structure)

---

## Results at a glance

| Competition | Track / Task | Rank | Percentile |
|---|---|---|---|
| **Yandex ML Challenge 2026** · [Description](https://github.com/mmmaximov/mmkaggle/blob/main/Yandex%20ML%20Challenge%202026.%20LongTour/description%20of%20tasks.md) | [Long Tour (overall standings)](https://github.com/mmmaximov/mmkaggle/tree/main/Yandex%20ML%20Challenge%202026.%20LongTour) | 121 / 630 | Top 20% |
| **RWB WildHack** · [Description](https://github.com/mmmaximov/mmkaggle/blob/main/RWB%20Wildhack/description%20of%20tasks%20RWB.md) | [Uninterrupted Shipments (Solo)](https://github.com/mmmaximov/mmkaggle/blob/main/RWB%20Wildhack/WB_SOLO.ipynb) | 125 / 678 | Top 20% |
| **RWB WildHack** · [Description](https://github.com/mmmaximov/mmkaggle/blob/main/RWB%20Wildhack/description%20of%20tasks%20RWB.md) | [Warehouse Logistics Automation (Team)](https://github.com/mmmaximov/mmkaggle/blob/main/RWB%20Wildhack/WB_TEAM.ipynb) | 35 / 443 | Top 10% |

---

## Yandex ML Challenge 2026. Long Tour

**Final rank: 121 / 630 (top 20%)**

### A. Adaptive Puzzle Solving Challenge

$$\text{score} = \frac{\text{baseline length}}{\text{moves}}$$

**My score: 79.4**

### B. Novel View Synthesis

$$\text{PSNR} = 20 \cdot \log_{10}\!\left(\frac{255}{\sqrt{\text{MSE}}}\right)$$

$$\text{score} = \frac{\text{clamp}(\text{PSNR},\, 10,\, 30) - 10}{20} \times 100$$

**My score: 51.3**

### C. Effective Inference On School Questions

Judge-model evaluation (average score across answers).

**My score: 59.6**

---

## RWB WildHack

$$\text{score} = \text{WAPE} + |\text{Relative Bias}|, \qquad \text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i}, \qquad \text{Relative Bias} = \left|\frac{\sum \hat{y}_i}{\sum y_i} - 1\right|$$

### Uninterrupted Shipments (Solo track)

**Rank: 125 / 678 (top 20%)**

### Warehouse Logistics Automation (Team track)

**Rank: 35 / 443 (top 10%)**

---

## Repository structure

```
mmkaggle/
├── Yandex ML Challenge 2026. LongTour/
│   ├── description of tasks.md
│   ├── A_puzzle_solving.ipynb
│   ├── B_novel_view_synthesis.ipynb
│   └── C_school_questions_inference.ipynb
└── RWB Wildhack/
    ├── description of tasks RWB.md
    ├── WB_SOLO.ipynb          # Uninterrupted Shipments (solo)
    └── WB_TEAM.ipynb          # Warehouse Logistics Automation (team)
```
