# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:56:14 2026

@author: Juan Pablo Aguirre
"""

# https://www.kaggle.com/datasets/laotse/credit-risk-dataset

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# =============================================================================
# PASO 1: CARGA DE DATOS Y DEFINICIÓN DE LA LGD HISTÓRICA OBSERVADA (IFRS 9)
# =============================================================================

# 1. Carga de datos
df = pd.read_csv('credit_risk_dataset.csv')

# 2. Selección de columnas clave
columnas_clave = [
    'person_income',              # Ingreso anual
    'person_home_ownership',      # Tipo de vivienda (Proxy de colateral)
    'person_emp_length',          # Antigüedad laboral
    'loan_intent',                # Propósito del crédito
    'loan_grade',                 # Calificación interna (A a G)
    'loan_amnt',                  # Monto del crédito (EAD Base)
    'loan_int_rate',              # Tasa de interés (EIR Base)
    'loan_percent_income',        # Capacidad de pago / DTI
    'cb_person_default_on_file',  # Historial previo de default (Y/N)
    'cb_person_cred_hist_length', # Antigüedad crediticia
    'loan_status'                 # Target de Default (0 = Good, 1 = Bad/Default)
]

df = df[columnas_clave].copy()
    
# 3. Filtrar únicamente la cartera en incumplimiento (loan_status == 1)    
df_lgd = df[df['loan_status'] == 1].copy().reset_index(drop=True)

# 4. Asignación de EAD y Tasa Efectiva de Interés (EIR)
df_lgd['EAD'] = df_lgd['loan_amnt']
df_lgd['EIR'] = df_lgd['loan_int_rate'] / 100.0
df_lgd['EIR'] = df_lgd['EIR'].fillna(df_lgd['EIR'].mean())

# 5. Simulación de flujos de recuperación y costos de cobranza (al no estar en el dataset)
np.random.seed(42)

# Tasa base de recuperación según tipo de vivienda (colateral)
mapa_recuperacion = {'OWN': 0.55, 'MORTGAGE': 0.42, 'RENT': 0.25, 'OTHER': 0.18}
tasa_base = df_lgd['person_home_ownership'].map(mapa_recuperacion).fillna(0.20)

# Factor aleatorio individual
ruido_individual = np.random.uniform(-0.10, 0.10, len(df_lgd))
tasa_recuperacion_est = np.clip(tasa_base + ruido_individual, 0.05, 0.85)

# Proyección de flujos (65% año 1, 35% año 2) y costos de cobranza
df_lgd['recuperacion_año1'] = df_lgd['EAD'] * tasa_recuperacion_est * 0.65
df_lgd['recuperacion_año2'] = df_lgd['EAD'] * tasa_recuperacion_est * 0.35
df_lgd['costos_cobranza'] = df_lgd['EAD'] * np.random.uniform(0.015, 0.035, len(df_lgd))

# 6. Cálculo del Valor Presente Descontado a la EIR (Descuento IFRS 9)
flujos_descontados = (
    (df_lgd['recuperacion_año1'] - df_lgd['costos_cobranza']) / ((1 + df_lgd['EIR'])**1) +
    (df_lgd['recuperacion_año2']) / ((1 + df_lgd['EIR'])**2)
)

# 7. DEFINICIÓN DEL TARGET / VARIABLE DEPENDIENTE PARA LGD
# LGD Observada = 1 - (VP Flujos / EAD)
df_lgd['lgd_observada'] = 1 - (flujos_descontados / df_lgd['EAD'])
df_lgd['lgd_observada'] = df_lgd['lgd_observada'].clip(0.001, 0.999)


# =============================================================================
# PASO 2: MODELO ESTADÍSTICO (FRACTIONAL LOGIT / GLM)
# =============================================================================

# 1. Tratamiento rápido de nulos para las variables explicativas si los hubiese
df_lgd['person_emp_length'] = df_lgd['person_emp_length'].fillna(df_lgd['person_emp_length'].median())
df_lgd['loan_int_rate'] = df_lgd['loan_int_rate'].fillna(df_lgd['loan_int_rate'].median())

# 2. Especificación de la fórmula optimizada (solo variables significativas)
model_formula = (
    "lgd_observada ~ "
    "C(person_home_ownership)"
)

# 3. Ajuste del Modelo Fractional Logit (GLM con familia Binomial y enlace Logit)
glm_model = smf.glm(
    formula=model_formula,
    data=df_lgd,
    family=sm.families.Binomial(link=sm.families.links.Logit())
).fit()

# 4. Predicción de la LGD estimada dentro de la muestra (Point-In-Time)
df_lgd['LGD_predicha'] = glm_model.predict(df_lgd)

print("=============================================================")
print("--- RESUMEN DEL MODELO LGD (FRACTIONAL LOGIT) ---")
print("=============================================================")
print(glm_model.summary())

print("\n--- MUESTRA DE LGD ESTIMADA VS OBSERVADA ---")
print(df_lgd[['lgd_observada', 'LGD_predicha', 'person_home_ownership', 'loan_grade']].head())


# =============================================================================
# PASO 3: PROYECCIÓN IFRS 9 FORWARD-LOOKING Y ESCENARIOS MACROECONÓMICOS
# =============================================================================

# 1. Selección de una muestra de cartera a evaluar (Primeros 5 registros)
cartera_evaluar = df_lgd[['loan_amnt', 'person_home_ownership', 'LGD_predicha']].head(5).copy()

# 2. Configuración de Escenarios Macroeconómicos NIIF 9 y Ponderaciones
escenarios = {
    'Base':      {'peso': 0.50, 'factor_macro': 1.00},  # Escenario central
    'Pesimista': {'peso': 0.30, 'factor_macro': 1.15},  # Deterioro macro (+15% severidad)
    'Optimista': {'peso': 0.20, 'factor_macro': 0.85}   # Mejora macro (-15% severidad)
}

# 3. Estimación por Escenario aplicando los factores prospectivos
resultados_escenarios = pd.DataFrame({
    'id_credito': cartera_evaluar.index + 1,
    'loan_amnt': cartera_evaluar['loan_amnt'],
    'person_home_ownership': cartera_evaluar['person_home_ownership'],
    'LGD_PIT_Base': cartera_evaluar['LGD_predicha']
})

for nombre_escenario, config in escenarios.items():
    lgd_escenario = (cartera_evaluar['LGD_predicha'] * config['factor_macro']).clip(0.001, 0.999)
    resultados_escenarios[f'LGD_{nombre_escenario}'] = lgd_escenario

# 4. Cálculo de la LGD Ponderada NIIF 9 Final
resultados_escenarios['LGD_IFRS9_Ponderada'] = (
    resultados_escenarios['LGD_Base']      * escenarios['Base']['peso'] +
    resultados_escenarios['LGD_Pesimista'] * escenarios['Pesimista']['peso'] +
    resultados_escenarios['LGD_Optimista']  * escenarios['Optimista']['peso']
)

# =============================================================================
# REPORTE Y CONSOLIDACIÓN DE RESULTADOS
# =============================================================================

print("\n=============================================================")
print("--- LGD PROYECTADA BAJO ESCENARIOS MACROECONÓMICOS IFRS 9 ---")
print("=============================================================")
print(resultados_escenarios.to_string(index=False))

print("\n=============================================================")
print("--- RESUMEN PROMEDIO DE CARTERA (NIIF 9) ---")
print("=============================================================")
print(f"LGD Promedio Escenario Base:       {resultados_escenarios['LGD_Base'].mean():.2%}")
print(f"LGD Promedio Escenario Pesimista:  {resultados_escenarios['LGD_Pesimista'].mean():.2%}")
print(f"LGD Promedio Escenario Optimista:  {resultados_escenarios['LGD_Optimista'].mean():.2%}")
print(f"-------------------------------------------------------------")
print(f"--> LGD PONDERADA NIIF 9 FINAL:   {resultados_escenarios['LGD_IFRS9_Ponderada'].mean():.2%}")