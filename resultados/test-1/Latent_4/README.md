# Resultados: Latent Dimension = 4 (Extended Epochs)

**Resumen:**
Experimento con 4 variables latentes y entrenamiento extendido.
Se buscó observar si una mayor dimensionalidad latente permitía capturar mejor la complejidad de la dinámica a largo plazo.

**Métricas Globales:**
*   **Final Training Loss (MSE):** 0.0775
*   **Final Test Loss (MSE):** 0.9288

**Tabla de Error por Estado:**
| State | Train MSE (0-180) | Test MSE (180-401) | Train MAE | Test MAE |
|---|---|---|---|---|
| **NY** | 0.06683 | 0.68628 | 0.21777 | 0.71200 |
| **OH** | 0.11178 | 1.92521 | 0.28235 | 1.22394 |
| **NJ** | 0.04421 | 0.63345 | 0.17074 | 0.67452 |
| **GA** | 0.07488 | 1.85239 | 0.22560 | 1.16762 |
| **IL** | 0.06840 | 0.40182 | 0.22181 | 0.51607 |
| **FL** | 0.12324 | 1.11939 | 0.31103 | 0.88729 |
| **CA** | 0.07003 | 1.06319 | 0.21863 | 0.93317 |
| **VA** | 0.03144 | 0.73008 | 0.14806 | 0.78512 |
| **TX** | 0.15319 | 0.74800 | 0.32181 | 0.75054 |
| **NC** | 0.03126 | 0.12818 | 0.14367 | 0.31915 |

**Observaciones:**
*   **Overfitting:** El modelo tiene el menor error de entrenamiento (0.0775) de todos los experimentos latentes, pero el **mayor error de test (0.9288)**. Esto indica un claro overfitting: con 4 dimensiones latentes, el modelo tiene suficiente capacidad para "memorizar" el ruido o particularidades del entrenamiento, perdiendo generalización.
*   **Comparación:**
    *   Latent 0: Test MSE ~1.5 (Baseline)
    *   Latent 2: Test MSE ~0.6-0.7 (Mejor generalización, sujeto a re-confirmación)
    *   Latent 3: Test MSE ~0.80
    *   Latent 4: Test MSE ~0.93
*   **Conclusión:** Aumentar la dimensión latente más allá de 2 o 3 es contraproducente para la generalización en este dataset limitado.

**Archivos:**
*   `plots/`: Predicciones visuales.
*   `params.jld2`: Parámetros del modelo (Latent 4).
