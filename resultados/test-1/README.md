# Experimento Series 1: Análisis de Dimensión Latente en Graph Neural ODE

Este directorio contiene los resultados de la primera serie de experimentos (`test-1`) diseñada para investigar el impacto de las variables latentes en la capacidad del modelo GNN-ODE para predecir la dinámica de COVID-19 en múltiples estados.

## Objetivo
Determinar la dimensión latente óptima ($d_{latent}$) que minimice el error de generalización (Test MSE) en una ventana de predicción extendida ($t \in [180, 401]$), evitando tanto el subajuste (incapacidad de capturar dinámicas complejas) como el sobreajuste (memorización de ruido).

## Metodología

### Modelo
Se utilizó una arquitectura **Graph Neural ODE** que integra:
*   **GNN (Graph Neural Network):** 3 capas GraphConv con activación `tanh` para procesar interacciones espaciales entre estados.
*   **Neural ODE (Ordinary Differential Equation):** Modela la evolución temporal continua del estado oculto.
*   **Variables Latentes:** Se aumentaron los nodos con un vector de parámetros entrenables de tamaño $d_{latent}$ para capturar heterogeneidad no observada (e.g., políticas locales, densidad, cultura).

### Configuración del Experimento
*   **Datos:** Casos normalizados (Log-scale) de 10 estados de EE.UU.
*   **Split:**
    *   *Train:* Días 0-180.
    *   *Test:* Días 180-401 (Predicción a largo plazo).
*   **Entrenamiento:** Curriculum Learning (aumentando gradualmente el horizonte temporal) hasta Epoch ~60-180, con Early Stopping basado en Loss.
*   **Optimizador:** Adam (LR=0.001) + BFGS (refinamiento final, desactivado en corridas recientes por estabilidad).

### Variantes Probadas
1.  **Latent 0 (Baseline):** Sin memoria latente.
2.  **Latent 2:** Variables latentes de dimensión 2.
3.  **Latent 3:** Variables latentes de dimensión 3.
4.  **Latent 4:** Variables latentes de dimensión 4.

## Resumen de Resultados

| Experimento | Latent Dim | Train MSE | Test MSE | NY Test MSE | Observación |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | 0 | 0.0878 | 0.9227 | 2.09 | **Fallo Crítico.** No captura la ola de NY. |
| **Latent 2** | 2 | 0.0814 | 0.8102 | 1.36 | **Buen Balance.** Eficiente y robusto. |
| **Latent 3** | 3 | 0.0813 | **0.8057** | **0.84** | **Óptimo.** Mejor error global y local (NY). |
| **Latent 4** | 4 | 0.0775 | 0.9288 | 0.69 | **Overfitting.** Memoriza train, falla en test global. |

**Conclusión Inicial:**
La dimensión latente **3** es la ganadora indiscutible para esta serie de pruebas, ofreciendo la mejor capacidad de generalización y resolviendo singularidades regionales (NY) sin caer en el sobreajuste observado con dimensión 4.

## Análisis Profundo y Diagnóstico

### Hallazgos Clave
1.  **Latent 3 es óptimo (pero marginalmente):**
    *   *Baseline (L0)*: 0.9227 MSE
    *   *Latent 2*: 0.8102 MSE (12.2% mejora)
    *   *Latent 3*: 0.8057 MSE (12.7% mejora)
    *   *Latent 4*: 0.9288 MSE (peor, -0.7%)
    *   **Interpretación:** Las variables latentes ayudan significativamente, pero hay rendimientos decrecientes. El "sweet spot" es Latent 3, aunque la diferencia con Latent 2 es mínima (0.5%).

2.  **Overfitting y Gap Train/Test:**
    *   El ratio Test/Train aumenta con la capacidad. Latent 4 tiene el mejor ajuste en training (0.0775) pero el peor ratio (12.0x), indicando memorización pura.

3.  **Dificultad por Estado:**
    *   **Estados Difíciles (Outliers):** NY (1.24), GA (1.29), OH (1.07). Estos estados tienen dinámicas que el modelo actual lucha por extrapolar.
    *   **Estados Fáciles:** NC (0.38), NJ (0.55), FL (0.60).

4.  **Training No Saturado:**
    *   Todos los experimentos se estancaron en un Train MSE de ~0.08. Ninguno logró bajar de 0.05. Esto sugiere que el modelo está **sub-ajustando (underfitting)** los datos de entrenamiento antes de siquiera toparse con limitaciones de capacidad latente.

### Diagnóstico de Cuellos de Botella
El hecho de que el Loss se estanque en 0.08 sugiere tres limitaciones principales:
1.  **Capacidad del GNN:** La arquitectura actual (3 capas x 16 unidades) tiene solo ~800 parámetros para 7,200 puntos de datos, lo cual es insuficiente.
2.  **Curriculum Agresivo:** Los saltos en el horizonte de predicción (e.g., 5 -> 20) son muy bruscos (4x), pudiendo desestabilizar el aprendizaje.
3.  **Convergencia Incompleta:** 500 épocas en la etapa final podrían no ser suficientes para converger a un mínimo profundo.

Esto motiva la **Serie de Experimentos 2 (Test-2)**, enfocada en mejorar la arquitectura y el régimen de entrenamiento.

## Estructura del Directorio
*   `Latent_X/`: Contiene los resultados específicos para dimensión X.
    *   `README.md`: Detalles y tabla de error por estado.
    *   `plots/`: Gráficos de predicción (Rojo) vs Realidad (Azul).
    *   `params.jld2`: (Si disponible) Pesos entrenados del modelo.

## Reproducibilidad

Para reproducir estos resultados, siga los siguientes pasos desde la raíz del proyecto:

1.  **Preparación del Entorno:**
    ```bash
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    ```

2.  **Configurar el Experimento:**
    Edite el archivo `Train/model_opt.jl` y modifique la línea 29 para seleccionar la dimensión deseada:
    ```julia
    latent_dim = 3  # Cambiar a 0, 2, 3, o 4
    ```

3.  **Ejecutar Entrenamiento:**
    ```bash
    # Limpiar plots anteriores (opcional pero recomendado)
    rm -rf plots/training/*
    # Iniciar entrenamiento
    julia --project=. Train/model_opt.jl
    ```
    El modelo guardará los parámetros en `Params/par_opt_new.jld2` automáticamente al finalizar (por fin de épocas o Early Stopping).

4.  **Generar Métricas y Tablas:**
    Edite el archivo `Test/calculate_metrics_table.jl` para coincidir con la dimensión entrenada:
    ```julia
    latent_dim = 3  # Debe coincidir con el entrenamiento
    ```
    Ejecute el script de métricas:
    ```bash
    julia --project=. Test/calculate_metrics_table.jl
    ```
    La tabla con MSE/MAE por estado y global se imprimirá en la consola.

5.  **Análisis Visual:**
    Revise la carpeta `plots/training/` para ver las curvas de predicción generadas durante el entrenamiento.

## Notas Técnicas
*   Se implementó normalización logarítmica (`log(x+1)`) para estabilizar el entrenamiento.
*   Se corrigió la función `predict` para asegurar que las splines de covariables cubran todo el horizonte de test (0-401 días).
*   Se utilizó `tanh` como función de activación en las GNN para evitar explosiones numéricas en la integración de la ODE.
