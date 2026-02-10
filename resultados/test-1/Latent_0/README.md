# Resultados: Latent Dimension = 0 (Baseline)

**Resumen:**
Experimento "Baseline" sin variables latentes (`latent_dim = 0`).
Sirve como punto de referencia para cuantificar la mejora aportada por las variables latentes.

**Métricas Globales:**
*   **Final Training Loss (MSE):** 0.0878
*   **Final Test Loss (MSE):** 0.9227

**Tabla de Error por Estado:**
| State | Train MSE (0-180) | Test MSE (180-401) | Train MAE | Test MAE |
|---|---|---|---|---|
| **NY** | 0.01879 | 2.09251 | 0.09632 | 1.31353 |
| **OH** | 0.13872 | 0.41433 | 0.32189 | 0.56569 |
| **NJ** | 0.04528 | 0.46053 | 0.17233 | 0.60677 |
| **GA** | 0.14762 | 1.03788 | 0.29222 | 0.86602 |
| **IL** | 0.10658 | 0.91164 | 0.27015 | 0.79816 |
| **FL** | 0.08280 | 0.56580 | 0.25417 | 0.61092 |
| **CA** | 0.04853 | 0.69525 | 0.19291 | 0.63234 |
| **VA** | 0.07912 | 1.79362 | 0.23570 | 1.17289 |
| **TX** | 0.18403 | 1.10450 | 0.36852 | 0.80207 |
| **NC** | 0.02611 | 0.15092 | 0.13880 | 0.31974 |

**Conclusión:**
*   **Desempeño:** Tiene el peor desempeño de todos en entrenamiento (MSE 0.088) y su Test MSE (0.92) indica que le cuesta generalizar la dinámica compleja.
*   **NY Failure:** El error en NY (2.09) es masivo, lo que confirma que sin variables latentes, el modelo no puede capturar la heterogeneidad de la primera ola de NY.

**Archivos:**
*   `plots/`: Predicciones visuales.
*   `params.jld2`: Parámetros del modelo (Baseline).
