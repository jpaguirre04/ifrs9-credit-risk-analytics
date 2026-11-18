# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:01:18 2026

@author: Juan Pablo Aguirre
"""


"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# PASO 0: SIMULAMOS 10.000 CLIENTES (tu DataFrame)
np.random.seed(42)
n = 10000


df = pd.DataFrame({
    'ingreso': np.random.normal(800000, 250000, n), # sueldo en CLP
    'edad': np.random.randint(20, 70, n),
    'mora_max_12m': np.random.randint(0, 90, n), # dias de mora
    'deuda_total': np.random.normal(3000000, 1500000, n)
    })


# Creamos un default realista: a más mora y menos ingreso, más prob de default
prob_real = 1/ (1+ np.exp(-(-4 + 0.05*df['mora_max_12m']- 0.000001*df['ingreso'])))
df['default'] = np.random.binomial(1, prob_real) # 0 = paga, 1= default

# print(f'Registros: {len(df)}')
print(f'Tasa real default: {df['default'].mean():.2%}')

df.columns


X = df[['ingreso', 'edad', 'mora_max_12m', 'deuda_total']]
y = df['default']


# DE AQUÍ SIGUE LO DE ANTES
modelo_raw = LogisticRegression()
modelo_raw.fit(X, y)
pd_raw = modelo_raw.predict_proba(X)[:,1]


# ahora 
calibrador = LogisticRegression()
calibrador.fit(pd_raw.reshape(-1, 1), y)
pd_calibrada = calibrador.predict_proba(pd_raw.reshape(-1,1))[:,1]


df['pd_raw'] = pd_raw
df['pd_calibrada'] = pd_calibrada

# TABLITA IFRS9

df['decil'] = pd.qcut(df['pd_calibrada'], 10)
tabla = df.groupby('decil').agg(
                  clientes = ('default', 'count'),
                  pd_promedio = ('pd_calibrada', 'mean'),
                  tasa_real = ('default', 'mean')
                  )

print(tabla)
# Con esto ya tienes tus n filas listas

# Fíjate que la última tabla es la clave: si `pd_promedio` y `tasa_real` 
# están parecidos por decil, estás calibrado y cumples IFRS9.

"""


###############################################################################


"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


# PASO 0: SIMULAMOS 10.000 CLIENTES (tu DataFrame)
np.random.seed(42)
n = 1000000


df = pd.DataFrame({
    'ingreso': np.random.normal(800000, 250000, n), # sueldo en CLP
    'edad': np.random.randint(20, 70, n),
    'mora_max_12m': np.random.randint(0, 90, n), # dias de mora
    'deuda_total': np.random.normal(3000000, 1500000, n)
    })


# Creamos un default realista: a más mora y menos ingreso, más prob de default
prob_real = 1/ (1+ np.exp(-(-4 + 0.05*df['mora_max_12m']- 0.000001*df['ingreso'])))
df['default'] = np.random.binomial(1, prob_real) # 0 = paga, 1= default

# print(f'Registros: {len(df)}')
print(f'Tasa real default: {df['default'].mean():.2%}')

# df.columns


X = df[['ingreso', 'edad', 'mora_max_12m', 'deuda_total']]
y = df['default']
# DE AQUÍ SIGUE LO DE ANTES
# ------------------------------------------------
# PASO 1: TU MODELO RAW (sin calibrar)
# ------------------------------------------------
modelo_raw = LogisticRegression()
modelo_raw.fit(X, y)
pd_raw = modelo_raw.predict_proba(X)[:,1]


# Calibración Isotonic (la buena para IFRS9)
# ------------------------------------------------
# PASO 2: CALIBRACIÓN ISOTONIC (la buena para IFRS9)
# ------------------------------------------------
# Este bloque es el que arregla la calibración
calibrador_iso = CalibratedClassifierCV(modelo_raw, method = 'isotonic', cv= 'prefit')
calibrador_iso.fit(X, y)
pd_calibrada = calibrador_iso.predict_proba(X)[:,1]

df['pd_calibrada'] = pd_calibrada


# ------------------------------------------------
# PASO 3: TABLA FINAL IFRS9
# ------------------------------------------------
df['decil_cal'] = pd.qcut(df['pd_calibrada'], 10, duplicates= 'drop')

tabla_cal = df.groupby('decil_cal', observed= True).agg(
                      clientes = ('default', 'count'),
                      pd_promedio = ('pd_calibrada', 'mean'),
                      tasa_real = ('default', 'mean')
                      )

print('\n--- TABLA CALIBRADA IFRS9 (pd_promedio debe ser ~ tasa_real)---')
print(tabla_cal)

"""


###############################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# PASO 0: SIMULAMOS 10.000 CLIENTES
# ------------------------------------------------
np.random.seed(42)
n = 1000000

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


# ------------------------------------------------
# PASO 1: DIVIDIMOS EN 3 (ESTO ES LO PROFESIONAL)
# ------------------------------------------------
# 60% entrena, 20% calibra, 20% prueba final

X_train, X_temp, y_train, y_temp = train_test_split(X, y, 
                                                    test_size= 0.4, 
                                                    random_state= 42)

X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp,
                                                test_size= 0.5, 
                                                random_state= 42)

print(f'Train: {len(X_train)}, Cal {len(X_cal)}, Test: {len(X_test)}')


# ------------------------------------------------
# PASO 2: MODELO RAW SOLO CON TRAIN
# ------------------------------------------------
modelo_raw = LogisticRegression()
modelo_raw.fit(X_train, y_train)

# ------------------------------------------------
# PASO 3: CALIBRACIÓN ISOTONIC SOLO CON CAL
# ------------------------------------------------
calibrador_iso = CalibratedClassifierCV(modelo_raw, 
                                        method = 'isotonic', 
                                        cv= 'prefit')

calibrador_iso.fit(X_cal, y_cal) # aprende la corrección SOLO en CAL


# ------------------------------------------------
# PASO 4: EVALUAMOS EN TEST (DATOS QUE NUNCA HA VISTO)
# ------------------------------------------------

pd_test_calibrada = calibrador_iso.predict_proba(X_test)[:,1]

df_test = pd.DataFrame({
    'pd_calibrada': pd_test_calibrada,
    'default': y_test.values
    })

df_test['decil'] = pd.qcut(df_test['pd_calibrada'], 10, duplicates = 'drop')

tabla_final = df_test.groupby('decil', observed= True).agg(
    clientes = ('default', 'count'),
    pd_promedio= ('pd_calibrada', 'mean'),
    tasa_real = ('default', 'mean')
    )

print("\n--- TABLA FINAL PRO (debe ser CERCANA, no idéntica) ---")
print(tabla_final)
