# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:11:30 2026

@author: Juan Pablo Aguirre
"""

# https://www.kaggle.com/datasets/denychaen/lending-club-loans-rejects-data


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


from IFRS9_LCLD_PD_Paso_3_matriz_de_transición import df_PD
from IFRS9_LCLD_LGD_Paso_2_calibración import df_LGD


df_PD.columns 
df_LGD.columns


# ===== ECL FINAL (INTEGRACION) =====
# 1.-  INTEGRADO - Los dos tienen el mismo tamaño muestral, same sample
# Me quedaré con df_PD como base y le pegaré sólo la LGD


df_ECL = df_PD.copy()
df_ECL['LGD_final']= df_LGD['LGD_final'].values # .values para que no falle por índice
df_ECL['EAD'] = df_ECL['loan_amnt'] # EAD = monto del préstamo


# 2.- ECL 12 Meses -. Esta es la que usa la norma para Stage 1
# Usamos tu PD calibrada final
df_ECL['ECL_12m'] = df_ECL['pd_12m_calibrated'] * df_ECL['LGD_final'] * df_ECL['EAD']


# 3.- ECL LIFETIME - Para Stage 2
df_ECL['ECL_lifetime'] = df_ECL['pd_lifetime_cal'] * df_ECL['LGD_final'] * df_ECL['EAD']

# 4 .- ECL con escenarios (forward-looking)
df_ECL['ECL_12m_FL'] = df_ECL['pd_12m_fl'] * df_ECL['LGD_final'] * df_ECL['EAD']



print(f"Filas: {len(df_ECL)}")
print(f"Provisión total 12m: ${df_ECL['ECL_12m'].sum():,.0f}")
print(f"Provisión total Lifetime: ${df_ECL['ECL_lifetime'].sum():,.0f}")
print(f"Provisión total 12m con Forward-Looking: ${df_ECL['ECL_12m_FL'].sum():,.0f}")


# Revisión rápida por grado
print(df_ECL.groupby('grade')[['pd_12m_calibrated', 
                                    'LGD_final', 
                                    'ECL_12m', 
                                    'ECL_lifetime', 
                                    'ECL_12m_FL']].mean())

B = df_ECL.groupby('grade')[['pd_12m_calibrated', 
                                    'LGD_final', 
                                    'ECL_12m', 
                                    'ECL_lifetime', 
                                    'ECL_12m_FL']].mean()


# Saltos de Stages, cálculos:
# regla práctica  

def asignar_stage(row):
      if row['loan_status'] == 1:
          return 3 # ya es default
      elif row['grade'] in ['E', 'F', 'G']:
          return 2 # grados especulativos = ya se deterioró
      elif row['pd_12m_calibrated'] > 0.30:
            return 2 # PD alta = SICR significant increase credit risck
      else:
          return 1 # sano

df_ECL['stage'] = df_ECL.apply(asignar_stage, axis=1)

# ahora aplicando condicionar construyo el ECL que corresponde ###

# Una observación acá es que sacaré el factor de ajuste que le apliqué en 12m

factor_fl = df_ECL['pd_12m_fl'] / df_ECL['pd_12m_calibrated']

# Aplicarle ese mismo factor al Lifetime
df_ECL['pd_lifetime_fl'] = df_ECL['pd_lifetime_cal'] * factor_fl
df_ECL['ECL_lifetime_FL'] = df_ECL['pd_lifetime_fl'] * df_ECL['LGD_final'] * df_ECL['EAD']



import numpy as np
df_ECL['ECL_final'] = np.where(
                               df_ECL['stage']==1,
                               df_ECL['ECL_12m_FL'], # stag 1 -> 12m
                               df_ECL['ECL_lifetime_FL']) # stage 2 y 3 -> Lifetime
    

df_ECL['ECL_final'].describe()


provision_total = df_ECL['ECL_final'].sum()
ead_total = df_ECL['EAD'].sum()

print(f"Provision total: ${provision_total:,.0f}")
print(f"EAD total: ${ead_total:,.0f}")
print(f"ECL / EAD: {provision_total / ead_total:.2%}")


# df_ECL.columns
