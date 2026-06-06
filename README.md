# mmkaggle

Репозиторий с ноутбуками и решениями ML-соревнований, в которых я участвовал.

| Соревнование | Трек / Задача | Место | Топ |
|---|---|---|---|
| **Yandex ML Challenge 2026. Long Tour** | [Long Tour (общий зачёт)](https://github.com/mmmaximov/mmkaggle/tree/main/Yandex%20ML%20Challenge%202026.%20LongTour) | 121 / 630 | 20% |
| **RWB WildHack** | [Отгрузки без простоев (Solo)](https://github.com/mmmaximov/mmkaggle/blob/main/RWB%20Wildhack/WB_SOLO.ipynb) | 125 / 678 | 20% |
| **RWB WildHack** | [Автоматизация логистики склада (Team)](https://github.com/mmmaximov/mmkaggle/blob/main/RWB%20Wildhack/WB_TEAM.ipynb) | 35 / 443 | 10% |

---

## Yandex ML Challenge 2026. Long Tour

**Итоговое место: 121 / 630 (топ-20%)**

### A. Adaptive Puzzle Solving Challenge

$$\text{score} = \frac{\text{baseline length}}{\text{moves}}$$

**Мой скор: 79.4**

### B. Novel View Synthesis

$$\text{PSNR} = 20 \cdot \log_{10}\!\left(\frac{255}{\sqrt{\text{MSE}}}\right)$$

$$\text{score} = \frac{\text{clamp}(\text{PSNR},\, 10,\, 30) - 10}{20} \times 100$$

**Мой скор: 51.3**

### C. Effective Inference On School Questions

Оценка модели-судьёй (средний балл по ответам).

**Мой балл: 59.6**

---

## RWB WildHack

$$\text{score} = \text{WAPE} + |\text{Relative Bias}|, \qquad \text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i}, \qquad \text{Relative Bias} = \left|\frac{\sum \hat{y}_i}{\sum y_i} - 1\right|$$

### Отгрузки без простоев (Solo трек)

**Место: 125 / 678 (топ-20%)**

### Автоматизация логистики склада (Team трек)

**Место: 35 / 443 (топ-10%)**
