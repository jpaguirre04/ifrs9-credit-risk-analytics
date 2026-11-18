# Modelado IFRS 9 / NIIF 9: Estimación de ECL (EAD, PD, LGD) con Datos de Lending Club

Este repositorio contiene una implementación completa en Python de los modelos econométricos y metodologías de riesgo de crédito requeridos por la norma **IFRS 9** (Pérdida Crediticia Esperada - *Expected Credit Loss*, ECL), utilizando datos históricos de préstamos de la plataforma Lending Club (disponibles en Kaggle).

---

## Estructura y Flujo de los Scripts

El proyecto está estructurado de manera modular secuencial para cubrir todo el ciclo de estimación de provisiones:

### 1. Probabilidad de Default (PD)
* **`IFRS9_LCLD_PD_Paso_1_regresion_logistica.py`**: Carga los datos de préstamos aceptados, limpia variables clave (ingresos, FICO, DTI, sub-grados) y entrena un modelo de Regresión Logística para la PD a 12 meses (*Through-The-Cycle*). Incluye validación discriminatoria (ROC AUC, Gini, KS) y calibración por *Intercept Shifting*.
* **`IFRS9_LCLD_PD_Paso_2_forward_looking.py`**: Incorpora escenarios macroeconómicos (Base, Adverso, Optimista) condicionados por proyecciones de desempleo para cumplir con el enfoque *Forward-Looking* de la norma.
* **`IFRS9_LCLD_PD_Paso_3_matriz_de_transición.py`**: Modela la **PD *Lifetime*** (a 3 y 5 años) utilizando Cadenas de Markov (Matrices de Transición por calificación crediticia) calibradas consistentemente con la PD a 12 meses.

### 2. Pérdida Dada la Morosidad (LGD)
* **`IFRS9_LCLD_LGD_Paso_1.py`**: Analiza la cartera en default, calcula la tasa de recupero neta descontando costos de cobranza y ajusta una regresión logística fraccional (GLM) segmentada por tipo de vivienda (`home_ownership`).
* **`IFRS9_LCLD_LGD_Paso_2_calibración.py`**: Aplica factores de calibración a la media y exporta el modelo (`.pkl`) para asegurar la consistencia de la LGD en toda la cartera.

### 3. Integración y Pruebas de Tensión (Stress Testing)
* **`IFRS9_LCLD_ECL_integrado.py`**: Script integrador principal. Asigna la Exposición al Incumplimiento (`EAD`), clasifica los créditos por **Stages de IFRS 9** (Stage 1 de 12m y Stage 2/3 *Lifetime*) y calcula la provisión monetaria final de ECL bajo escenarios macroeconómicos.
* **`IFRS9_LCLD_STRESS_TESTING.py`**: Simula choques macroeconómicos adversos (PIB y desempleo) para evaluar el impacto en la migración de carteras hacia el Stage 2 y el incremento porcentual de las reservas de capital.

---

## Requisitos y Dependencias

Para ejecutar los scripts de manera local, asegúrate de tener instalado Python 3.x junto con las siguientes librerías científicas y de machine learning:

```bash
pip install numpy pandas scikit-learn statsmodels matplotlib scipy
