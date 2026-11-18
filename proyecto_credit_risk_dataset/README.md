# Modelado IFRS 9 / NIIF 9: Estimación de ECL con Credit Risk Dataset (Kaggle)

Este repositorio contiene una implementación integral en Python de los componentes de riesgo de crédito requeridos por la norma **IFRS 9** (Pérdida Crediticia Esperada - *Expected Credit Loss*, ECL), utilizando el dataset público **Credit Risk Dataset** de Kaggle.

---

## Estructura y Flujo de los Scripts

El desarrollo metodológico está dividido en módulos secuenciales para cubrir la estimación de los parámetros regulatorios:

### 1. Probabilidad de Default (PD)
* **`IFRS9_CRD_PD_PASO_1_regresion_logistica.py`**: Carga de datos, transformación de variables financieras y demográficas, y ajuste de un modelo de Regresión Logística para la PD a 12 meses (*Through-The-Cycle*). Incluye validación estadística (ROC AUC, Gini, KS) y calibración por *Intercept Shifting*.
* **`IFRS9_CRD_PD_PASO_2_forward_looking.py`**: Incorporación de escenarios macroeconómicos (Base, Adverso, Optimista) condicionados por proyecciones de desempleo para cumplir con el enfoque *Forward-Looking*.
* **`IFRS9_CRD_PD_PASO_3_matriz_de_transición.py`**: Modelado de la **PD *Lifetime*** (a 3 y 5 años) mediante Cadenas de Markov (Matrices de Transición por calificación de riesgo `loan_grade`), calibradas consistentemente frente a la PD a 12 meses.

### 2. Pérdida Dada la Morosidad (LGD)
* **`IFRS9_CRD_LGD_1.py`**: Simulación de flujos de recuperación descontados y costos de cobranza sobre la cartera en default, implementando un modelo estadístico de regresión logística fraccional (GLM) segmentado por tipo de vivienda.
* **`IFRS9_CRD_LGD_2_calibracion.py`**: Calibración de la LGD a la media observada y exportación de artefactos de modelo (`.pkl`) para la cartera general.

### 3. Integración y Pruebas de Tensión (Stress Testing)
* **`IFRS9_CRD_ECL_intregrado.py`**: Script integrador principal. Asigna la Exposición al Incumplimiento (`EAD`), clasifica la cartera según los **Stages de IFRS 9** (criterios de SICR) y calcula la provisión final de ECL a 12 meses y *Lifetime*.
* **`IFRS9_CRD_STRESS_TESTING.py`**: Simulación de escenarios de estrés macroeconómico (shocks de PIB y desempleo) para evaluar el impacto sobre la migración de etapas (*Staging*) y el incremento porcentual de las provisiones.

---

## Requisitos y Dependencias

Para ejecutar los scripts de manera local, asegúrate de tener instalado Python 3.x junto con las siguientes librerías:

```bash
pip install numpy pandas scikit-learn statsmodels matplotlib scipy
