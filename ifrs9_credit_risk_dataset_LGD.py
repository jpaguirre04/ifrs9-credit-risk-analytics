# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:56:08 2026

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
df_lgd = df[df['loan_status']== 1]    .copy().reset_index(drop= True)
    

# 4. Asignación de EAD y Tasa Efectiva de Interés (EIR)
df_lgd['EAD'] = df_lgd['loan_amnt']
df_lgd['EIR'] = df_lgd['loan_int_rate'] / 100.0
df_lgd['EIR'] = df_lgd['EIR'].fillna(df_lgd['EIR'].mean())

# 5. Simulación de flujos de recuperación y costos de cobranza 
# (al no estar en el dataset)
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

flujos_descontados =(
    (df_lgd['recuperacion_año1'] - df_lgd['costos_cobranza']) /((1 + df_lgd['EIR'])**1)+
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

# 2. Especificación de la fórmula usando las características del dataset de Kaggle
# Se usa C() para variables categóricas para que statsmodels cree las variables dummy

"""
model_formula = (
    "lgd_observada ~ "
    "loan_percent_income + "
    "cb_person_cred_hist_length + "
    "C(person_home_ownership) + "
    "C(loan_intent) + "
    "C(loan_grade) + "
    "C(cb_person_default_on_file)"
)
"""
model_formula = (
    "lgd_observada ~ "
    "C(person_home_ownership)"
)

# 3. Ajuste del Modelo Fractional Logit (GLM con familia Binomial 
# y enlace Logit)
# Este método es ideal bajo IFRS 9 ya que asegura que las predicciones se 
# mantengan en (0, 1)

glm_model = smf.glm(
    formula = model_formula,
    data = df_lgd,
    family = sm.families.Binomial(link = sm.families.links.Logit())
).fit()

# 4. Predicción de la LGD estimada (LGD_PIT / LGD_base)
df_lgd['LGD_predicha'] = glm_model.predict(df_lgd)


print("\n--- RESUMEN DEL MODELO LGD (FRACTIONAL LOGIT) ---")
print(glm_model.summary())

print("\n--- MUESTRA DE LGD ESTIMADA VS OBSERVADA ---")
print(df_lgd[['lgd_observada', 'LGD_predicha', 'person_home_ownership', 'loan_grade']].head())








