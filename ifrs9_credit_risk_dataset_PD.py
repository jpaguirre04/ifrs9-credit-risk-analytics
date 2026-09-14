# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:22:38 2026

@author: Juan Pablo Aguirre
"""


# https://www.kaggle.com/datasets/laotse/credit-risk-dataset

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# =============================================================================
# PASO 1: CARGA DE DATOS Y DEFINICIÓN DEL TARGET
# =============================================================================
# Carga de datos

df = pd.read_csv('credit_risk_dataset.csv')


# 2. Selección de columnas equivalentes
columnas_clave = [
    'person_income',  # Equiv. annual_inc
    'person_home_ownership',  # Equiv. home_ownership
    'person_emp_length',  # Equiv. emp_length
    'loan_intent',  # Propósito del crédito
    'loan_grade',  # Calificación interna (A, B, C, D, E, F, G)
    'loan_amnt',  # Equiv. loan_amnt
    'loan_int_rate',  # Equiv. int_rate
    'loan_percent_income',  # Equiv. a una métrica de capacidad de pago/DTI
    'cb_person_default_on_file',  # Historial previo de default (Y/N)
    'cb_person_cred_hist_length',  # Antigüedad en buró de crédito
    'loan_status',  # Variable TARGET ya viene codificada (0 = Good, 1 = Bad/Default)
]

df = df[columnas_clave].copy()


# 3. DEFINICIÓN DEL TARGET
# En este dataset 'loan_status' ya viene como binario:
# 0 = Fully Paid / Good
# 1 = Default / Bad
df['target_default'] = df['loan_status'].astype(int)


# =============================================================================
# TRANSFORMACIÓN Y LIMPIEZA INICIAL DE VARIABLES
# =============================================================================

# A. Limpieza de Tasa de Interés (loan_int_rate)
# Si viene como objeto/texto con '%', lo limpiamos. Si ya es numérico, 
# se asegura el tipo float.

if df['loan_int_rate'].dtype == 'object':
    df['loan_int_rate']= (
        df['loan_int_rate'].str.rstrip('%').astype(float)
        )
else:
        df['loan_int_rate'] = df['loan_int_rate'].astype(float)


# B. Antigüedad laboral (person_emp_length) a valor numérico
# Rellenamos nulos con 0 o la mediana (en IFRS 9 el tratamiento 
# de NAs es crítico)

df['emp_length_years']=(
    df['person_emp_length'].fillna(0).astype(float)
)


# C. Mapeo ordinal de Calificación Interna (loan_grade: A=1 a G=7)
# Adaptado ya que este dataset no usa sub-grades (A1..G5) sino 
# grades principales (A..G)


# df['loan_grade'].unique().tolist() , abajo lo ordenó alfabéticamente

grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
grade_map = {grade: idx + 1 for idx, grade in enumerate(grades)}
df['grade_num'] = df['loan_grade'].map(grade_map)



# D. Tratamiento de Categorías Previas de Buró (cb_person_default_on_file)
# Mapeo binario: Y=1, N=0

df['historical_default_flag'] = (
      df['cb_person_default_on_file'].map({'Y': 1, 'N': 0}).fillna(0)
    )



# =============================================================================
# PASO 2: ESTIMACIÓN DE PD 12M TTC CON REGRESIÓN LOGÍSTICA (ENFOQUE DIRECTO)
# =============================================================================

# TTC significa Through-The-Cycle (A través del ciclo).

# 1. Definición de variables explicativas (X) y variable objetivo (y)


features = ['cb_person_cred_hist_length', 'loan_percent_income',
            'emp_length_years', 'person_income', 'loan_amnt',
            'loan_int_rate', 'grade_num',]

X = df[features].fillna(df[features].median())
y = df['target_default']

# 2. Ajuste del modelo de Regresión Logística sobre toda la muestra
modelo = LogisticRegression(max_iter =1000)
modelo.fit(X, y)

# 3. Cálculo e incorporación de la Probabilidad de Default TTC (12 Meses)

df['pd_12m_ttc'] =  modelo.predict_proba(X)[:,1]



# =============================================================================
# PASO 3: FORWARD-LOOKING CON 3 ESCENARIOS (NIIF 9)
# =============================================================================

# 1. Definición de escenarios macroeconómicos (ej. Proyección Tasa de Desempleo)
# Tomando como desempleo base histórico / proyectado: 8.0%

escenarios = pd.DataFrame({
                    'escenario': ['Base', 'Adverso', 'Optimista'],
                    'prob': [0.5, 0.3, 0.2],
                    'desempleo': [8.0, 10.5, 6.5] # Proyecciones macroeconómicas a 12 meses
              })

# 2. Función de ajuste por sensibilidad / elasticidad macroeconómica


def ajustar_pd(pd_ttc, desemplo_esc, desempleo_base= 0.8, elasticidad= 0.15):
    """Ajusta la PD TTC mediante un factor de escala según el escenario macroeconómico.

    - elasticidad: Cambio % en la PD por cada punto de variación en el desempleo.
    """
    factor = 1 + elasticidad * (desemplo_esc - desempleo_base)
    # Clip para asegurar que las probabilidades se mantengan en el rango [0, 1]
    return (pd_ttc * factor).clip(0, 1)
    
    

# 3. Cálculo de la PD bajo cada escenario
for esc in escenarios['escenario']:
     desempleo_val = escenarios.loc[escenarios['escenario']== esc, 'desempleo'].values[0]
     df[f'pd_12m_fl_{esc.lower()}'] = ajustar_pd(df['pd_12m_ttc'], desempleo_val)


# 4. PD Final Forward-Looking Ponderada por Probabilidad de Ocurrencia
# de los Escenarios (MÉTRICA FINAL NIIF 9)

df['pd_12m_fl'] = 0
for i, row in escenarios.iterrows():
    # print(i, row)
    esc_name = row['escenario'].lower()
    df['pd_12m_fl'] += row['prob'] * df[f'pd_12m_fl_{esc_name}']



# =============================================================================
# PASO 4: PD LIFETIME CON MATRIZ DE TRANSICIÓN (INTEGRADO)
# =============================================================================

# 1. Matriz Anual de Transición de Estados (Estados: A, B, C, D, E, F, G, Default)
# La última columna/fila representa el Estado Absorbente (Default)
M_1ano = np.array([
    # A     B     C     D     E     F     G    Default
    [0.85, 0.08, 0.03, 0.01, 0.01, 0.00, 0.00, 0.02],  # A
    [0.04, 0.78, 0.10, 0.03, 0.01, 0.01, 0.00, 0.03],  # B
    [0.01, 0.05, 0.72, 0.12, 0.04, 0.01, 0.00, 0.05],  # C
    [0.00, 0.02, 0.06, 0.68, 0.12, 0.03, 0.01, 0.08],  # D
    [0.00, 0.01, 0.02, 0.07, 0.62, 0.12, 0.03, 0.13],  # E
    [0.00, 0.00, 0.01, 0.03, 0.08, 0.58, 0.10, 0.20],  # F
    [0.00, 0.00, 0.00, 0.01, 0.04, 0.08, 0.55, 0.32],  # G
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # Default
])



# 2. Potencias de matriz para 36 meses (3 años) y 60 meses (5 años)
M_3anos = np.linalg.matrix_power(M_1ano, 3)
M_5anos = np.linalg.matrix_power(M_1ano, 5)


# 3. Homologación de la letra de calificación (loan_grade)
df['grade'] = df['loan_grade'].astype(str).str.strip().str.upper()


# 4. Mapeos de PD acumulada a 3 y 5 años según la letra

mapa_pd_3y = {
    'A': M_3anos[0, -1], 'B': M_3anos[1, -1],
    'C': M_3anos[2, -1], 'D': M_3anos[3, -1],
    'E': M_3anos[4, -1], 'F': M_3anos[5, -1],
    'G': M_3anos[6, -1],
}

mapa_pd_5y = {
    'A': M_5anos[0, -1], 'B': M_5anos[1, -1],
    'C': M_5anos[2, -1], 'D': M_5anos[5, -1],
    'E': M_5anos[4, -1], 'F': M_5anos[5, -1],
    'G': M_5anos[6, -1],
}


# 5. Asignación de plazo y PD Lifetime
# Como este dataset no incluye 'term', asignamos 36 meses por defecto
# (o usamos la variable 'term_months' si la definiste previamente)

if 'term_months' not in df.columns:
    df['term_months'] = 36

pd_3y = df['grade'].map(mapa_pd_3y)
pd_5y = df['grade'].map(mapa_pd_5y)

df['pd_lifetime'] = np.where(df['term_months']<= 36, pd_3y, pd_5y)



# =============================================================================
# PASO 5: ETAPAS NIIF 9 (STAGING) Y CÁLCULO DE PÉRDIDA ESPERADA (ECL)
# =============================================================================

# 1. Definición de EAD y LGD
# Usamos 'loan_amnt' como proxy de EAD (Exposición al Incumplimiento)
df['ead'] = df['loan_amnt']

# Asignamos LGD normativa del 45%
df['lgd'] = 0.45


# 2. Estimación de PD al Origen (PD Origination)
# Simulamos la PD al momento del desembolso (ej. 80% de la PD TTC actual)
df['pd_origen'] = df['pd_12m_ttc'] * 0.80

# Ratio de deterioro relativo para evaluar SICR
df['ratio_deterioro'] = df['pd_12m_fl'] / df['pd_origen']



# 3. Asignación de Etapas NIIF 9 (Staging)
# - Etapa 1 (Stage 1): Sin deterioro significativo de riesgo.
# - Etapa 2 (Stage 2): Deterioro significativo de riesgo (SICR).
# - Etapa 3 (Stage 3): Default / Incumplimiento.


# Criterios para Stage 2 (SICR)
cond_ratio = df['ratio_deterioro'] > 2.0  # El riesgo relativo se duplicó
cond_calif = (
        df['grade_num'] >= 5
      )# Calificación baja: E (5), F (6) o G (7) [Adaptado a la escala 1-7]

# Asignación a Etapa 2 si cumple algún criterio de deterioro (SICR)
df.loc[cond_ratio | cond_calif, 'etapa_niif9'] = 2

# Asignación a Etapa 3 (Créditos en default real)
df.loc[df['target_default'] == 1 , 'etapa_niif9'] = 3



# 4. Selección de PD para la Provisión según Etapa
# - Etapa 1: PD 12 Meses Forward-Looking (pd_12m_fl)
# - Etapa 2: PD Lifetime (pd_lifetime a 3 o 5 años según corresponda)
# - Etapa 3: 100% (1.0) ya que el default ocurrió

df['pd_provision'] = np.where(
        df['etapa_niif9'] == 1,
        df['pd_12m_fl'],
        np.where(df['etapa_niif9'] == 2, df['pd_lifetime'], 1.0),
    
    )

# 5. Cálculo de la Pérdida Esperada (ECL / Provisión)
# ECL = PD * LGD * EAD

df['ecl_provision'] = df['pd_provision'] * df['lgd'] * df['ead']




# ===================================================================
# 8. VALIDACIÓN Y PRUEBAS DEL MODELO
# ===================================================================
from sklearn.metrics import roc_auc_score

print('\n=== PRUEBAS DE VALIDACIÓN DEL MODELO ===')

# 1. Discriminación: AUC-ROC y Gini
auc = roc_auc_score(df['target_default'], df['pd_12m_ttc'])
gini = 2 * auc - 1

# 2. Discriminación: Estadística KS (Kolmogorov-Smirnov)
df_ks = df.sort_values(by='pd_12m_ttc', ascending=False).reset_index(
    drop=True
)
total_registros = len(df)
df_ks['bad_cum'] = (
    df_ks['target_default'].cumsum() / df_ks['target_default'].sum()
)
df_ks['good_cum'] = (1 - df_ks['target_default']).cumsum() / (
    total_registros - df_ks['target_default'].sum()
)
ks_stat = max(abs(df_ks['bad_cum'] - df_ks['good_cum']))

# 3. Calibración / Backtesting
pd_promedio = df['pd_12m_ttc'].mean()
default_real = df['target_default'].mean()

print(f'ROC-AUC Score   : {auc:.4f}')
print(f'Coeficiente Gini : {gini:.4f}')
print(f'Estadístico KS   : {ks_stat:.4f}')
print(f'PD Promedio Est. : {pd_promedio:.2%}')
print(f'Default Real Obs.: {default_real:.2%}')
print(f'Diferencia Abs.  : {abs(pd_promedio - default_real):.2%}')







