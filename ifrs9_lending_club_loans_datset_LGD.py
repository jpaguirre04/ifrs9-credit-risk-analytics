# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:39:25 2026

@author: Juan Pablo Aguirre
"""


# https://www.kaggle.com/datasets/denychaen/lending-club-loans-rejects-data


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================================================
# PASO 1: Carga, Filtrado y Cálculo de LGD Observada
# =========================================================

# 1. Cargar la base de datos
df = pd.read_parquet('lending_club_lgd_2.parquet')

# 2. Filtrar únicamente créditos en Default o Castigados (Target = 1)
df_default = df[df['target'] == 1].copy()

# 3. Definir Exposición al Default (EAD)
df_default['EAD'] = df_default['funded_amnt']

# 4. Calcular Recuperaciones Totales Netas
df_default['recuperaciones_totales'] = (
    df_default['total_rec_prncp'] + 
    df_default['total_rec_int'] + 
    df_default['total_rec_late_fee'] + 
    df_default['recoveries']
)

df_default['recuperacion_neta'] = (
    df_default['recuperaciones_totales'] - df_default['collection_recovery_fee']
)

# 5. Cálculo de Tasa de Recuperación (RR) y LGD Observada
df_default['Recovery_Rate'] = np.where(
    df_default['EAD'] > 0, 
    df_default['recuperacion_neta'] / df_default['EAD'], 
    0
)

df_default['LGD_observada'] = (1 - df_default['Recovery_Rate']).clip(0.001, 0.999)


# =========================================================
# PASO 2: Modelo Econométrico GLM Fractional Logit
# =========================================================

# 1. Transformaciones logarítmicas de variables continuas
df_default['log_funded_amnt'] = np.log(df_default['funded_amnt'])

if 'annual_inc' in df_default.columns:
    df_default['annual_inc'] = pd.to_numeric(df_default['annual_inc'], errors='coerce')
    mediana_inc = df_default['annual_inc'].median()
    df_default['log_annual_inc'] = np.log1p(df_default['annual_inc'].fillna(mediana_inc))

# 2. Limpieza e imputación de variables numéricas
num_cols = ['int_rate', 'dti', 'revol_util', 'open_acc', 'total_acc']
for col in num_cols:
    if col in df_default.columns:
        if df_default[col].dtypes == 'object' or isinstance(df_default[col].dtype, pd.CategoricalDtype):
            df_default[col] = df_default[col].astype(str).str.replace('%', '', regex=False).str.strip()
        df_default[col] = pd.to_numeric(df_default[col], errors='coerce')
        df_default[col] = df_default[col].fillna(df_default[col].median())

# 3. Definición de la fórmula multivariable
variables_candidatas = [
    'log_funded_amnt', 'int_rate', 'dti', 'log_annual_inc',
    'revol_util', 'total_acc', 'C(term)', 'C(grade)',
    'C(home_ownership)', 'C(verification_status)'
]

variables_validas = [
    var for var in variables_candidatas 
    if var.replace('C(', '').replace(')', '') in df_default.columns
]

model_formula = "LGD_observada ~ " + " + ".join(variables_validas)

# 4. Ajuste del modelo Fractional Logit (GLM Binomial)
glm_model = smf.glm(
    formula=model_formula,
    data=df_default,
    family=sm.families.Binomial(link=sm.families.links.Logit())
).fit()

# Predicción dentro de la muestra
df_default['LGD_predicha'] = glm_model.predict(df_default)


# =========================================================
# PASO 3: Proyección IFRS 9 Forward-Looking & Escenarios
# =========================================================

# 1. Cartera a evaluar (Muestra)
cartera_evaluar = df_default.head(5).copy()

# 2. Configuración de Escenarios y Ponderaciones NIIF 9
escenarios = {
    'Base': {'peso': 0.50, 'factor_macro': 1.00},
    'Pesimista': {'peso': 0.30, 'factor_macro': 1.15},
    'Optimista': {'peso': 0.20, 'factor_macro': 0.85}
}

# 3. Estimación por Escenario
lgd_base = glm_model.predict(cartera_evaluar)

resultados_escenarios = pd.DataFrame({
    'id': cartera_evaluar['id'], 
    'funded_amnt': cartera_evaluar['funded_amnt']
})

for nombre_escenario, config in escenarios.items():
    lgd_escenario = (lgd_base * config['factor_macro']).clip(0.001, 0.999)
    resultados_escenarios[f'LGD_{nombre_escenario}'] = lgd_escenario

# 4. LGD Ponderada NIIF 9
resultados_escenarios['LGD_IFRS9_Ponderada'] = (
    resultados_escenarios['LGD_Base'] * escenarios['Base']['peso'] +
    resultados_escenarios['LGD_Pesimista'] * escenarios['Pesimista']['peso'] +
    resultados_escenarios['LGD_Optimista'] * escenarios['Optimista']['peso']
)

# Impresión de Resultados Consolidados
print("--- LGD Proyectada bajo Escenarios Macroeconómicos IFRS 9 ---")
print(resultados_escenarios.to_string(index=False))

print("\n--- Resumen Promedio de Cartera ---")
print(f"LGD Promedio Escenario Base: {resultados_escenarios['LGD_Base'].mean():.2%}")
print(f"LGD Promedio Escenario Pesimista: {resultados_escenarios['LGD_Pesimista'].mean():.2%}")
print(f"LGD Promedio Escenario Optimista: {resultados_escenarios['LGD_Optimista'].mean():.2%}")
print(f"--> LGD Ponderada NIIF 9 Final: {resultados_escenarios['LGD_IFRS9_Ponderada'].mean():.2%}")