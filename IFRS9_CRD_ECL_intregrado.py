# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:30:23 2026

@author: Juan Pablo Aguirre
"""

# https://www.kaggle.com/datasets/laotse/credit-risk-dataset

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from IFRS9_CRD_PD_PASO_3_matriz_transición import df_PD
from IFRS9_CRD_LGD_2_calibracion import df_LGD


# 1.-  INTEGRADO - Los dos tienen el mismo tamaño muestral, same sample
# Me quedaré con df_PD como base y le pegaré sólo la LGD


df_ECL = df_PD.copy()
df_ECL['LGD_final'] = df_LGD['LGD_final'].values # .values para que no falle por índice
df_ECL['EAD'] = df_ECL['loan_amnt'] # EAD = monto del préstamo

# 2.- ECL 12 Meses -. Esta es la que usa la norma para Stage 1
# Usamos tu PD calibrada final
df_ECL['ECL_12m'] = df_ECL['pd_12m_calibrated'] * df_ECL['LGD_final'] * df_ECL['EAD']

# 3.- ECL LIFETIME - Para Stage 2

df_ECL['ECL_lifetime'] = df_ECL['pd_lifetime_calibrada'] * df_ECL['LGD_final'] * df_ECL['EAD']


# 4 .- ECL con escnarios (forward-looking)
df_ECL['ECL_12m_FL'] = df_ECL['pd_12m_fl'] * df_ECL['LGD_final'] * df_ECL['EAD']


print(f"Filas: {len(df_ECL)}")
print(f"Provisión total 12m: ${df_ECL['ECL_12m'].sum():,.0f}")
print(f"Provisión total Lifetime: ${df_ECL['ECL_lifetime'].sum():,.0f}")
print(f"Provisión total 12m con Ofrward-Looking: ${df_ECL['ECL_12m_FL'].sum():,.0f}")

# Revisión rápida por grado
print(df_ECL.groupby('loan_grade')[['pd_12m_calibrated', 'LGD_final', 'ECL_12m', 'ECL_lifetime', 'ECL_12m_FL']].mean())


B = df_ECL.groupby('loan_grade')[['pd_12m_calibrated', 'LGD_final', 'ECL_12m', 'ECL_lifetime', 'ECL_12m_FL']].mean()


# Saltos de Stages, cálculos:
    
# regla práctica    

def asignar_stage(row):
    
     if row['loan_status'] == 1:
        return 3# ya es default
     elif row['loan_grade'] in ['E', 'F', 'G']:
        return 2 # grados especulativos = ya se deterioró
     elif row['pd_12m_calibrated'] > 0.30 :
        return 2 # PD alta = SICR
     else:
         return 1 # sano


df_ECL['stage']= df_ECL.apply(asignar_stage, axis=1)
### ahora construyo el ELC que corresponde ###

import numpy as np
df_ECL['ECL_final'] = np.where(
                               df_ECL['stage']==1,
                               df_ECL['ECL_12m_FL'], # stage 1 -> 12m
                               df_ECL['ECL_lifetime']) # stage 2 y 3 -> LifeTime  






