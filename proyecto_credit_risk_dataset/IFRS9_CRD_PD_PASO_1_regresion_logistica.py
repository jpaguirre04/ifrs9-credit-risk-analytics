# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:58:09 2026

@author: Juan Pablo Aguirre
"""


# https://www.kaggle.com/datasets/laotse/credit-risk-dataset

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp
from scipy.optimize import minimize_scalar

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


# imputar valores faltantes missing values con mediana, se imputa 
# con la mediana por grado 

df['loan_int_rate'] = df.groupby('loan_grade')['loan_int_rate'].transform(
       lambda x: x.fillna(x.median())
    )

 
# B. Antigüedad laboral (person_emp_length) a valor numérico
# Rellenamos nulos con la mediana (en IFRS 9 el tratamiento 
# de NAs es crítico)



df['emp_length_years']=(
    df['person_emp_length'].fillna(df['person_emp_length'].median())
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

# E. Como person_home_ownership y loan_intent son textos aun, hay que 
# convertirlas a números 

df.columns

df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], 
                    drop_first= True, dtype= int)


# =============================================================================
# PASO 2: ESTIMACIÓN DE PD 12M TTC CON REGRESIÓN LOGÍSTICA (ENFOQUE DIRECTO)
# Y VALIDACIÓN
# =============================================================================

# TTC significa Through-The-Cycle (A través del ciclo).

# 1. Definición de variables explicativas (X) y variable objetivo (y)


features = ['cb_person_cred_hist_length', 'loan_percent_income',
            'emp_length_years', 'person_income', 'loan_amnt',
            'loan_int_rate', 'grade_num',  'historical_default_flag',
            'person_home_ownership_OTHER', 'person_home_ownership_OWN',
            'person_home_ownership_RENT', 'loan_intent_EDUCATION',
            'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
            'loan_intent_PERSONAL', 'loan_intent_VENTURE']

X = df[features].fillna(df[features].median())
y = df['target_default']


# 2.- EL SPLIT, para validar de verdad


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.30,
                                                    random_state = 42,
                                                    stratify = y)

# 3.- Pipeline con scaler + modelo

modelo = Pipeline([
         ('scaler', StandardScaler()),    
         ('logreg', LogisticRegression(max_iter=1000))
    ])
modelo.fit(X_train, y_train)

y_pred_test = modelo.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_test)
gini = (2 * roc_auc) - 1 

# C. Kolmogorov-Smirnov (KS)
# Separamos las probabilidades predichas para los Buenos (0) y Malos (1)


# Calculamos el estadístico KS entre ambas distribuciones
proba_buenos = y_pred_test[y_test == 0]
proba_malos = y_pred_test[y_test == 1]

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

# 4. Re-entreno con toda la muestra para la PD final

modelo_final = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter= 1000))
    ])

modelo_final.fit(X, y)


# 5. Guardar la PD predicha en el dataframe principal
df['pd_12m_ttc'] =  modelo_final.predict_proba(X)[:,1]




# =============================================================================
# PASO 3: CALIBRACIÓN IFRS 9 - ESTÁNDAR BANCARIO (INTERCEPT SHIFTING)
# =============================================================================
# Objetivo: Ajustar el nivel promedio de la PD para que coincida
# con la Tasa de Default Observada (ODR) de largo plazo de la cartera,
# sin alterar la capacidad de discriminación del modelo (AUC).

# Usamos modelo_final, entrenado con el 100% de la muestra.
# TARGET_LRA = ODR histórico de la cartera (en este dataset: 21.7%).
# En un banco, este valor vendría de la serie histórica de ciclo completo.
logreg_final = modelo_final.named_steps['logreg']
scaler_final = modelo_final.named_steps['scaler']

# TARGET_LRA = 0.045 # Tu LRA de largo plazo, no la media de este año
TARGET_LRA = df['target_default'].mean() # te va a dar como 0.217



# Hay que escalar X antes de hacer el dot, porque el modelo lo hace así
X_scaled = scaler_final.transform(X)
coefs = logreg_final.coef_[0]
intercepto_original = logreg_final.intercept_[0]
linear_combination = np.dot(X_scaled, coefs)


def funcion_costo(delta_intercept):
    logit = linear_combination + intercepto_original + delta_intercept
    pd_pred = 1 / (1 + np.exp(-logit))
    return (pd_pred.mean() - TARGET_LRA) ** 2

resultado = minimize_scalar(funcion_costo)
delta_optimo = resultado.x

# Aplicar al modelo final
logreg_final.intercept_[0] += delta_optimo

df['pd_12m_calibrated'] = modelo_final.predict_proba(X)[:, 1]


print(f"AUC del Modelo (Inalterado): {roc_auc_score(y, df['pd_12m_calibrated']):.4f}")
print(f"PD Promedio Calibrada: {df['pd_12m_calibrated'].mean():.4f}")
print(f"Target LRA Objetivo: {TARGET_LRA:.4f}")



from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve

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


