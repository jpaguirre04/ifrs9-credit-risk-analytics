# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:02:43 2026

@author: Juan Pablo Aguirre
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

np.random.seed(42)
n = 10000
df = pd.DataFrame({
    'ingreso': np.random.normal(800000, 250000, n),
    'edad': np.random.randint(20, 70, n),
    'mora_max_12m': np.random.randint(0, 90, n),
    'deuda_total': np.random.normal(3000000, 1500000, n)
})
prob_real = 1 / (1 + np.exp(-(-4 + 0.05*df['mora_max_12m'] - 0.000001*df['ingreso'])))
df['default'] = np.random.binomial(1, prob_real)

X = df[['ingreso', 'edad', 'mora_max_12m', 'deuda_total']]
y = df['default']

# 50% train, 30% cal, 20% test -> más datos para calibrar
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

modelo_raw = LogisticRegression()
modelo_raw.fit(X_train, y_train)

# 1. RAW
pd_raw = modelo_raw.predict_proba(X_test)[:,1]

# 2. ISOTONIC
cal_iso = CalibratedClassifierCV(modelo_raw, method='isotonic', cv='prefit')
cal_iso.fit(X_cal, y_cal)
pd_iso = cal_iso.predict_proba(X_test)[:,1]

# 3. SIGMOID (Platt) - mejor con pocos datos
cal_sig = CalibratedClassifierCV(modelo_raw, method='sigmoid', cv='prefit')
cal_sig.fit(X_cal, y_cal)
pd_sig = cal_sig.predict_proba(X_test)[:,1]

# Métrica PRO: Brier Score (más bajo = más cercano a la realidad)
print(f"Brier RAW:      {brier_score_loss(y_test, pd_raw):.5f}")
print(f"Brier ISOTONIC: {brier_score_loss(y_test, pd_iso):.5f}")
print(f"Brier SIGMOID:  {brier_score_loss(y_test, pd_sig):.5f}")


# Tabla comparativa
df_test = pd.DataFrame({'default': y_test.values, 'raw': pd_raw, 
                        'iso': pd_iso, 'sig': pd_sig})
df_test['decil'] = pd.qcut(df_test['iso'], 8, duplicates= 'drop')

tabla = df_test.groupby('decil', observed= True).agg(
    pd_raw = ('raw', 'mean'),
    pd_iso= ('iso', 'mean'),
    pd_sig= ('sig', 'mean'),
    tasa_real= ('default', 'mean')
    )
print("\n--- COMPARATIVA REAL ---")
print(tabla)



print("=" * 45)
print("     MÉTRICAS DE VALIDACIÓN MÁS SIMPLE")
print("=" * 45)







