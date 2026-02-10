# Resultados: Latent Dimension = 3 (Extended Epochs)

**Resumen:**
Experimento con 3 variables latentes y entrenamiento extendido (~2800 épocas totales).
El objetivo fue mejorar la convergencia y comparar la capacidad de generalización vs Latent 2.

**Métricas Globales:**
*   **Final Training Loss (MSE):** 0.0813
*   **Final Test Loss (MSE):** 0.8057

**Tabla de Error por Estado:**
| State | Train MSE (0-180) | Test MSE (180-401) | Train MAE | Test MAE |
|---|---|---|---|---|
| **NY** | 0.05585 | 0.83614 | 0.17214 | 0.78719 |
| **OH** | 0.12723 | 1.02983 | 0.29985 | 0.92342 |
| **NJ** | 0.03980 | 0.58666 | 0.15268 | 0.68184 |
| **GA** | 0.07199 | 0.73172 | 0.19900 | 0.61489 |
| **IL** | 0.10944 | 1.27361 | 0.26593 | 1.00350 |
| **FL** | 0.11945 | 0.38492 | 0.29122 | 0.51321 |
| **CA** | 0.03798 | 0.93407 | 0.15719 | 0.80178 |
| **VA** | 0.04835 | 0.53555 | 0.17678 | 0.63740 |
| **TX** | 0.15667 | 0.81856 | 0.31490 | 0.76553 |
| **NC** | 0.04640 | 0.92638 | 0.18975 | 0.84533 |

**Observaciones:**
*   **Generalización:** El Test MSE (0.80) es significativamente mayor que el Train MSE (0.08), lo que sugiere que aunque el modelo aprende bien la historia (0-180), tiene dificultades para proyectar la magnitud exacta de las olas futuras en todos los estados, aunque captura la forma general.
*   **Comparación:** Se requiere contrastar con Latent 4 para ver si mayor capacidad reduce el error de test.

**Archivos:**
*   `plots/`: Predicciones visuales.
*   `params.jld2`: Parámetros del modelo (Latent 3).
