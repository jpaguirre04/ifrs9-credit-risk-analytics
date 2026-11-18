# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:26:49 2026

@author: Juan Pablo Aguirre
"""


# https://www.kaggle.com/datasets/denychaen/lending-club-loans-rejects-data



import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle


from IFRS9_LCLD_LGD_Paso_1 import df, df_lgd, glm_model


# =============================================================================
#  CALIBRACIÓN NIIF 9 - 
# =============================================================================

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
df_lgd['LGD_calibrada'] = (df_lgd['LGD_predicha']* factor_calib)


# 2. VERIFICACION - Que la media calibrada ya cuadre
print(f"LGD Calibrada promedio: {df_lgd['LGD_calibrada'].mean():.2%}")
print(f"Diferencia vs Observada: {(df_lgd['LGD_calibrada'].mean() - mean_obs):.4f}")


# 3. VERIFICACION POR SEGMENTO - Que no te quede chueco por dentro
calib_segmento = df_lgd.groupby('home_ownership').agg(
       lgd_obs = ('lgd_observada', 'mean'),
       lgd_pre = ('LGD_predicha', 'mean'),
       lgd_cal = ('LGD_calibrada', 'mean')
     ).reset_index
print(calib_segmento)


# 4. CALIBRACION PIT -> TTC (La que tenias en el otro caso)
df_lgd['LGD_TTC'] = df_lgd['LGD_calibrada'] * 0.9 + mean_obs * 0.1

# 5. GUARDAR PARA TU PROVISION INTEGRADA
with open('glm_lgd_model.pkl', 'wb') as f:
    pickle.dump(glm_model, f)

with open('factor_lgd.txt', 'w') as f:
    f.write(str(factor_calib))


# 6. APLICAR A TODA LA CARTERA VIGENTE (df completo) - ESTO ES LO QUE USA EL INTEGRADO
df['LGD_pred_raw'] = glm_model.predict(df)
df['LGD_final'] = (df['LGD_pred_raw'] * factor_calib).clip(0.05, 0.95)


print(f"\n--> Para tu tesis, usa en el integrado: LGD = {mean_obs:.2%} (promedio observado calibrado)")
print(f"--> O usa la columna df['LGD_final'] que ya esta calibrada por tipo de vivienda")


print(df.groupby('home_ownership')['LGD_final'].mean())

df_LGD = df

