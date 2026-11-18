# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:33:18 2026

@author: Juan Pablo Aguirre
"""

# https://www.kaggle.com/datasets/denychaen/lending-club-loans-rejects-data



import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf




# =============================================================================
# construcción de la norma ifrs9 niif9 cálculo de proviciones parámetro LGD
# =============================================================================

# 1. Carga de datos
df = pd.read_parquet('lending_club_stress_testing_new.parquet')

# 2. Selección de columnas clave
columnas_clave = [
 'funded_amnt',             # Monto desembolsado. Es tu EAD, lo que pusiste en riesgo
 'recoveries',              # Lo que recuperaste DESPUES del default, por cobranza
 'collection_recovery_fee', # Lo que te costó cobrar eso. Se resta a recoveries
 'total_rec_prncp',         # Capital que sí te pagó antes de defaultear
 'home_ownership',          # RENT / OWN / MORTGAGE. Para segmentar la LGD, tu proxy de colateral
 'loan_status',             # Para filtrar solo los Charged Off / Default, que son los que tienen LGD
 'target',                  # Target de Default (0 = Good, 1 = Bad/Default)
]

df = df[columnas_clave].copy()


df.columns

# 3. Filtrar únicamente la cartera en incumplimiento (loan_status == 1)
df_lgd = df[df['target']==1].copy().reset_index(drop= True)


# 4. CALCULO DE LA LGD OBSERVADA POR CREDITO

# 4.1 Recuperación neta: lo que recuperaste menos lo que te costó cobrarlo
df_lgd['recuperacion_neta'] = df_lgd['recoveries'] - df_lgd['collection_recovery_fee']

# 4.2 Tasa de recupero: qué % de lo prestado lograste recuperar
df_lgd['tasa_recupero'] = df_lgd['recuperacion_neta'] / df_lgd['funded_amnt']

# 4.3 LGD observada: 1 - lo que recuperaste
df_lgd['lgd_observada'] = 1 - df_lgd['tasa_recupero']

# 4.4 Limpiar: la LGD no puede ser menor a 0 ni mayor a 1
# Si recuperaste más de lo que prestaste por intereses, se deja en 0
df_lgd['lgd_observada'] = df_lgd['lgd_observada'].clip(0,1)



##### OBSERVACIÓN ######
# 5 LGD PROMEDIO Y SEGMENTACION POR COLATERAL

# 5.1 LGD promedio de toda la cartera en default
lgd_promedio_total = df_lgd['lgd_observada'].mean()
print(f"lgd_promedio_total: {lgd_promedio_total:.4f}")


# 5.2 LGD promedio por tipo de vivienda (tu segmentación)
# RENT / OWN / MORTGAGE / OTHER

lgd_por_segmento = df_lgd.groupby('home_ownership')['lgd_observada'].agg(
    n_creditos='count',
    lgd_promedio='mean',
    lgd_mediana='median',
    desvación= 'std'
  ).reset_index().sort_values('lgd_promedio', ascending=False)

print(lgd_por_segmento)

# 5.3 Revisar si tienes valores raros
print(df_lgd['lgd_observada'].describe())


# =============================================================================
# PASO 2: MODELO ESTADISTICO (FRACTIONAL LOGIT / GLM)
# =============================================================================

# 1. Especificacion de la formula (solo home_ownership)
model_formula = "lgd_observada ~ C(home_ownership)"

# 2. Ajuste del Modelo Fractional Logit (GLM con familia Binomial y enlace Logit)
glm_model = smf.glm(
              formula = model_formula,
              data = df_lgd,
              family = sm.families.Binomial(link= sm.families.links.Logit())
           ).fit()

# 3. Prediccion de la LGD estimada dentro de la muestra (Point-In-Time)
df_lgd['LGD_predicha'] = glm_model.predict(df_lgd)

print(glm_model.summary())
print(df_lgd[['lgd_observada', 'LGD_predicha', 'home_ownership']].head())






