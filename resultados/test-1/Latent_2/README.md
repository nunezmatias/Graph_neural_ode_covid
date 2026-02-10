# Resultados: Latent Dimension = 2 (Re-run)

**Resumen:**
Experimento con 2 variables latentes y entrenamiento extendido. 
Este experimento fue re-ejecutado para confirmar resultados y recuperar parámetros.

**Métricas Globales:**
*   **Final Training Loss (MSE):** 0.0814
*   **Final Test Loss (MSE):** 0.8102

**Tabla de Error por Estado:**
| State | Train MSE (0-180) | Test MSE (180-401) | Train MAE | Test MAE |
|---|---|---|---|---|
| **NY** | 0.02646 | 1.35859 | 0.13325 | 1.02273 |
| **OH** | 0.12900 | 0.90585 | 0.31159 | 0.85808 |
| **NJ** | 0.04140 | 0.51629 | 0.16845 | 0.62548 |
| **GA** | 0.07838 | 1.52483 | 0.22732 | 0.98262 |
| **IL** | 0.07718 | 0.62084 | 0.23376 | 0.69266 |
| **FL** | 0.10424 | 0.33961 | 0.28904 | 0.46955 |
| **CA** | 0.07436 | 0.46386 | 0.22106 | 0.58708 |
| **VA** | 0.03379 | 1.06638 | 0.15090 | 0.92420 |
| **TX** | 0.20723 | 0.99019 | 0.36138 | 0.86591 |
| **NC** | 0.04166 | 0.31550 | 0.16714 | 0.49768 |

**Comparación y Conclusión:**
*   **Performance:** Muy similar a Latent 3 (Test MSE ~0.806 vs 0.810).
*   **Robustez:** Muestra excelente generalización en NC y FL, aunque sufre en NY y GA.
*   **Parsimonia:** Dado que Latent 3 no ofrece una mejora significativa en Test MSE, Latent 2 sigue siendo una opción muy competitiva por tener menos parámetros (mejor parsimonia).

**Archivos:**
*   `plots/`: Predicciones visuales.
*   `params.jld2`: Parámetros del modelo (Latent 2).
