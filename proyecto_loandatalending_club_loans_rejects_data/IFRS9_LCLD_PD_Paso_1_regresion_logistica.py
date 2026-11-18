# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:06:56 2026

@author: Juan Pablo Aguirre
"""

# https://www.kaggle.com/datasets/denychaen/lending-club-loans-rejects-data


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# =============================================================================
# PASO 1: CARGA DE DATOS Y DEFINICIÓN DEL TARGET
# =============================================================================
# Carga de datos desde parquet
df = pd.read_parquet('accepted_loans_final.parquet')

# 1. Las 12 columnas estratégicas
columnas_clave = [
    'issue_d',          # Fecha de originación (Vintage / Orden cronológico para Train/Test)
    'loan_status',      # Estado del crédito: Fully Paid, Charged Off (Target de Default)
    'addr_state',       # Estado de EE. UU. del deudor (Segmentación geográfica / Riesgo regional)
    'annual_inc',       # Ingreso anual del deudor (Capacidad de pago)
    'dti',              # Debt-to-Income ratio (Nivel de endeudamiento / Apalancamiento)
    'fico_range_low',   # Límite inferior del puntaje FICO (Predictora clave de historial crediticio)
    'emp_length',       # Antigüedad laboral en años (Estabilidad / Probabilidad de cura)
    'home_ownership',   # Tipo de vivienda: RENT, OWN, MORTGAGE (Proxy de estabilidad patrimonial)
    'loan_amnt',        # Monto del préstamo solicitado/aprobado (Exposición al riesgo / EAD inicial)
    'term',             # Plazo del crédito: 36 o 60 meses (Horizonte temporal de riesgo)
    'int_rate',         # Tasa de interés anual (Precio del riesgo / Endogeneidad potencial)
    'sub_grade',        # Sub-grado de riesgo A1-G5 (Score sintético interno de la plataforma)
]

df = df[columnas_clave]

# df.columns


# df['home_ownership'].unique()

# 2. DEFINICIÓN DEL TARGET (VARIABLE DEPENDIENTE Y)

bad_status = ['Charged Off', 'Default']
good_status = ['Fully Paid']

df = df[df['loan_status'].isin(bad_status + good_status)].copy()

# Crear la variable objetivo binaria Y (1 = Default / Incumplimiento, 0 = No Default)
df['target_default'] = df['loan_status'].apply(lambda x: 1 if x in bad_status else 0)


# 3. TRANSFORMACIÓN Y LIMPIEZA INICIAL DE VARIABLES EN df


# A:  Parseo de fechas (issue_id)
# A. Control Robusto de Fechas (Garantiza preservación de datos)
if not pd.api.types.is_datetime64_any_dtype(df['issue_d']):
  # Si es número/timestamp:
  if pd.api.types.is_numeric_dtype(df['issue_d']):
    df['issue_d'] = pd.to_datetime(
        df['issue_d'], unit='ms', errors='coerce'
    ).fillna(pd.to_datetime(df['issue_d'], unit='D', origin='1970-01-01'))
  else:
    # Si es texto:
    df['issue_d'] = pd.to_datetime(
        df['issue_d'].astype(str).str.strip(), errors='coerce'
    )

# B. Limpieza de Tasa de Interés (int_rate)
if df['int_rate'].dtype == 'object':
    df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)


# C. Limpieza de Plazo (termo)

df['term_months'] = df['term'].str.extract('(\\d+)').astype(float)


# D. Antiguedad laboral (emp_lenght) a valor númerico

df['emp_length_years'] = (df['emp_length'].str.extract('(\\d+)').astype(float).fillna(0))


# E. mapeo ordinal de Calificación Interna (sub_grade: A1=1 a G5=35)

sub_grades = [
        f'{g}{i}' for g in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] for i in range(1,6)
    ] 
sub_grade_map = {sg: idx + 1 for idx, sg in enumerate(sub_grades)}
df['sub_grade_num'] = df['sub_grade'].map(sub_grade_map)


# =============================================================================
# PASO 2: ESTIMACIÓN DE PD 12M TTC CON REGRESIÓN LOGÍSTICA (ENFOQUE DIRECTO)
# =============================================================================

# TTC significa Through-The-Cycle (A través del ciclo).

# 1. Definición de variables explicativas (X) y variable objetivo (y)


features = ['fico_range_low', 'dti', 'emp_length_years',
             'annual_inc', 'loan_amnt', 'int_rate','sub_grade_num',
            ]

# Imputación rápida de nulos si existieran para evitar fallas en el modelo
X = df[features]. fillna(df[features].median())
y = df['target_default']


# 2. Ajuste del modelo de Regresión Logística sobre toda la muestra
modelo = LogisticRegression(max_iter =1000)
modelo.fit(X, y)

# 3. Cálculo e incorporación de la Probabilidad de Default TTC (12 Meses)
df['pd_12m_ttc'] = modelo.predict_proba(X)[:, 1]



# =============================================================================
# PASO 2B: VALIDACIÓN DISCRIMINATORIA (AUC, KS, GINI)
# =============================================================================

from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp

# 1. Split 70/30 estratificado
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state= 42,
                                                    stratify=y)
modelo = LogisticRegression(max_iter= 1000)
modelo.fit(X_train, y_train)

# Probabilidades
y_pred_test  = modelo.predict_proba(X_test)[:, 1]

# Cálculo métricas de validación:

# A. ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_test)

#B. coeficiente de gini
gini = (2 * roc_auc) - 1

# C. Kolmogorov-Smirnov (KS)
# Separamos las probabilidades predichas para los Buenos (0) y Malos (1)

proba_buenos = y_pred_test[y_test == 0]
proba_malos = y_pred_test[y_test == 1]


# Calculamos el estadístico KS entre ambas distribuciones
ks_stat, _ = ks_2samp(proba_buenos, proba_malos)


# =============================================================================
# REPORTE DE VALIDACIÓN
# =============================================================================
print("=" * 45)
print("     MÉTRICAS DE DISCRIMINACIÓN (TEST SET)")
print("=" * 45)
print(f"ROC AUC Score : {roc_auc:.4f}  ({roc_auc*100:.2f}%)")
print(f"Gini Index : {gini:.4f} ({gini*100:.2f}%)")
print(f"KS Statistic: {ks_stat:.4f} ({ks_stat*100:.2f}%)")
print("=" * 45)

# 6. Guardar la PD predicha en el dataframe principal
# df['pd_12m_ttc'] = modelo.predict_proba(X)[:,1]




# =============================================================================
# PASO 3: CALIBRACIÓN IFRS 9 - ESTÁNDAR BANCARIO (INTERCEPT SHIFTING)
# =============================================================================
# Definir la Tasa de Largo Plazo (LRA) histórica de la cartera 
# (Ejemplo: 0.045 para 4.5%, o reemplazar por el ODR histórico real)

# En lugar de un valor fijo (0.045), calculas el ODR histórico real de tu base de datos
# (idealmente filtrando por la cohorte o ventana temporal de ciclo completo)

# TARGET_LRA = df['target_default'].mean() # O la media de tu periodo histórico de largo plazo
TARGET_LRA = 0.045

coefs = modelo.coef_[0]
intercepto_original = modelo.intercept_[0]
linear_combination = np.dot(X, coefs)

# Función objetivo para alinear exactamente la media predicha con el LRA
def funcion_costo(delta_intercept):
    logit = linear_combination + intercepto_original + delta_intercept
    pd_pred = 1 / (1+ np.exp(-logit))
    return (pd_pred.mean() - TARGET_LRA) ** 2


from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
# Optimización numérica del intercepto
resultado =  minimize_scalar(funcion_costo)
delta_optimo = resultado.x


# Aplicar el nuevo intercepto al modelo
modelo.intercept_[0] += delta_optimo


# Generar la PD 12M TTC Calibrada final
df['pd_12m_calibrated'] = modelo.predict_proba(X)[:, 1]


print(f"AUC del Modelo (Inalterado): {roc_auc_score(y, df['pd_12m_calibrated']):.4f}")
print(f"PD Promedio Calibrada: {df['pd_12m_calibrated'].mean():.4f}")
print(f"Target LRA Objetivo: {TARGET_LRA:.4f}")



# =============================================================================
# PASO 4: VALIDACIÓN REGULATORIA (RELIABILITY DIAGRAM)
# =============================================================================
prob_true, prob_pred = calibration_curve(y, df['pd_12m_calibrated'], n_bins = 10, strategy= 'quantile')


plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Modelo Calibrado (Intercept Shifting)')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectamente Calibrado')
plt.xlabel('PD Predicha Promedio por Bucket')
plt.ylabel('Tasa de Default Real')
plt.title('Calibration Plot - IFRS 9')
plt.legend()
plt.grid(True)
plt.show()



