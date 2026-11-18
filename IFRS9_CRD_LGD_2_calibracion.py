# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:58:24 2026

@author: Juan Pablo Aguirre
"""
# Para ver despues fannie mae y freddie mac
# https://github.com/kr900910/mortgage_data_analysis/tree/master
# https://github.com/xdderekchen/public_loan_data
# https://github.com/joshisn26/Freddie-Mac-Data-Analysis


# https://www.kaggle.com/datasets/laotse/credit-risk-dataset

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from IFRS9_CRD_LGD_1 import df, df_lgd, resultados_escenarios, glm_model


# =============================================================================
# PASO 4: CALIBRACIÓN NIIF 9 - 
# =============================================================================
import pickle

# 1. CALIBRACION A LA MEDIA (Obligatoria IFRS 9)
# Tu modelo predice, pero la media no cuadra con lo observado

mean_obs = df_lgd['lgd_observada'].mean()
mean_pred = df_lgd['LGD_predicha'].mean()

factor_calib = mean_obs / mean_pred


print(f"\n--- CALIBRACIÓN ---")
print(f"LGD Observada promedio: {mean_obs:.2%}")
print(f"LGD Predicha promedio (sin calibrar): {mean_pred:.2%}")
print(f"Factor de calibración: {factor_calib:.4f}")


# Aplicamos factor
df_lgd['LGD_calibrada'] = (df_lgd['LGD_predicha']* factor_calib).clip(0.001, 0.999)


# 2. CALIBRACION PIT -> TTC (Opcional pero te deja pro)
# Suavizamos para que no sea tan point-in-time

df_lgd['LGD_TTC'] = df_lgd['LGD_calibrada'] * 0.9 + mean_obs * 0.1


# 3. GUARDAR PARA TU PROVISION INTEGRADA
with open('glm_lgd_model.pkl', 'wb') as f:
      pickle.dump(glm_model, f)


with open('factor_lgd.txt', 'w') as f:
    f.write(str(factor_calib))

# 4. APLICAR A TODA LA CARTERA VIGENTE (df completo)
# df es tu cartera total con person_home_ownership

df['LGD_pred_raw'] = glm_model.predict(df)
df['LGD_final'] =  (df['LGD_pred_raw'] * factor_calib).clip(0.05, 0.95)


# Si quieres usar solo el promedio calibrado para toda la cartera:
print(f"\n--> Para tu tesis, usa en el integrado: LGD = {mean_obs:.2%} (promedio observado calibrado)")
print(f"--> O usa la columna df['LGD_final'] que ya está calibrada por tipo de vivienda")

print(df.groupby('person_home_ownership')['LGD_final'].mean())

df_LGD = df


df_lgd.columns

df_LGD.columns

