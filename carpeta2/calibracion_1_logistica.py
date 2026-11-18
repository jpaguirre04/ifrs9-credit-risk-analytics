# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:17:07 2026

@author: Juan Pablo Aguirre
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# PASO 1: Datos que ya tienes
"""
y = 1 si defaulteó en 12 meses, 0 si no
X = tus variables (ingreso, mora, etc)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
"""


# PASO 2: Modelo crudo (sin calibrar)
"""
modelo_raw = LogisticRegression()
modelo_raw.fit(X_train, y_train)
pd_raw = modelo_raw.predict_proba(X_test)[:,1] 
pd_raw ej: [0.02, 0.05, 0.80] -> pero no calza con realidad
"""


# PASO 3: Calibración IFRS9 - Platt Scaling
# Aquí ajustamos la PD raw a la tasa real observada
"""
calibrador = LogisticRegression()
calibrador.fit(pd_raw.reshape(-1,1), y_test)

pd_calibrada = calibrador.predict_proba(pd_raw.reshape(-1,1))[:,1]
Ahora pd_calibrada si calza con la realidad
"""


# PASO 4: Validación que pide CMF / auditoría IFRS9
# 4a. Creas deciles
"""
df = pd.DataFrame({'pd_cal': pd_calibrada, 'real': y_test})
df['decil'] = pd.qcut(df['pd_cal'], 10)

# 4b. Comparas promedio PD vs tasa real por decil
tabla_calib = df.groupby('decil').agg(
    pd_promedio=('pd_cal','mean'),
    tasa_real=('real','mean')
)
print(tabla_calib)
# Para que esté bien calibrado, pd_promedio ≈ tasa_real en cada decil
"""

"""
# PASO 5: Ajuste final a TTC para IFRS9
# Si tu tasa real de largo plazo es 5% y tu test tiene 3%, corriges:
ajuste = logit(0.05) - logit(0.03)
"""

"""
*¿Qué hace cada paso en simple?*

1. Entrenas tu logística normal.
2. Ves que tu PD cruda no pega con lo real.
3. Le pasas otra logística encima que solo hace el ajuste (ese es el truco IFRS9).
4. Haces la tablita que te va a pedir cualquier validador: "muéstrame que tu PD 
por decil es igual al default real".
5. Listo para calcular ECL = PD_calibrada _ LGD _ EAD.
"""


