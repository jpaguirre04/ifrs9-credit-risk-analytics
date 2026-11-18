# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:24:31 2026

@author: Juan Pablo Aguirre
"""

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. CARGA Y MAPPING - TUS COLUMNAS REALES
# -----------------------------------------------------------------------------
df_lc = pd.read_parquet('lending_club_stress_testing.parquet')

# A. PD histórica real por sub_grade usando 'target'
# target = 1 si falló, 0 si pagó
pd_por_subgrade = df_lc.groupby('sub_grade')['target'].transform('mean')
# print(pd_por_subgrade)
# pd_por_subgrade.unique()

# B. LGD real por sub_grade
lgd_real = 1 - (df_lc['recoveries'] / df_lc['funded_amnt'])

# B = df_lc[['recoveries', 'funded_amnt']]

# Promedio de LGD por sub_grade, SOLO con los que fallaron (target == 1)
lgd_por_subgrade = df_lc[df_lc['target'] == 1].groupby('sub_grade').apply(
    lambda x: (1 - x['recoveries'] / x['funded_amnt']).mean(),
    include_groups=False
)

# Si falló -> su LGD real. Si no falló -> promedio de SU sub_grade
lgd_promedio_por_grado = df_lc['sub_grade'].map(lgd_por_subgrade)

lgd_final = np.where(
    df_lc['target'] == 1,
    lgd_real,
    lgd_promedio_por_grado
)
lgd_final = pd.Series(lgd_final).fillna(0.89).clip(0.10, 0.95)

portafolio = pd.DataFrame({
    'account_id': df_lc.index + 1001,
    'ead': df_lc['funded_amnt'],
    'pd_baseline': pd_por_subgrade.clip(0.001, 0.999).round(4),
    'lgd_baseline': lgd_final.round(2),
    'stage_base': 1,
})

# -----------------------------------------------------------------------------
# 2. ESCENARIO
# -----------------------------------------------------------------------------
shock_pib = -0.05
shock_desempleo = 0.04
beta_pd_pib = -1.2
beta_pd_unemp = 0.8
beta_lgd_pib = -0.6

# -----------------------------------------------------------------------------
# 3. FUNCIONES
# -----------------------------------------------------------------------------
def compute_stressed_pd(pd_base, delta_pib, delta_unemp, b_pib, b_unemp):
    pd_base = pd_base.clip(0.001, 0.999)
    logit_base = np.log(pd_base / (1 - pd_base))
    logit_stressed = logit_base + (b_pib * delta_pib) + (b_unemp * delta_unemp)
    return 1 / (1 + np.exp(-logit_stressed))

def compute_stressed_lgd(lgd_base, delta_pib, b_pib):
    lgd_base = lgd_base.clip(0.01, 0.99)
    logit_lgd = np.log(lgd_base / (1 - lgd_base))
    logit_lgd_stressed = logit_lgd + (b_pib * delta_pib)
    return 1 / (1 + np.exp(-logit_lgd_stressed))

# -----------------------------------------------------------------------------
# 4. MOTOR
# -----------------------------------------------------------------------------
portafolio['pd_stressed'] = compute_stressed_pd(
    portafolio['pd_baseline'], shock_pib, shock_desempleo, beta_pd_pib, beta_pd_unemp)

portafolio['lgd_stressed'] = compute_stressed_lgd(
    portafolio['lgd_baseline'], shock_pib, beta_lgd_pib)

# -----------------------------------------------------------------------------
# 5. SICR / STAGING
# -----------------------------------------------------------------------------
portafolio['stage_stressed'] = np.where(
    (portafolio['pd_stressed'] > portafolio['pd_baseline'] * 1.20) |
    ((portafolio['pd_stressed'] - portafolio['pd_baseline']) > 0.01),
    2, 1
)

# portafolio['stage_stressed'].unique()

# portafolio['stage_stressed'].describe()

# -----------------------------------------------------------------------------
# 6. ECL
# -----------------------------------------------------------------------------
factor_lifetime = 3.0
portafolio['ecl_baseline'] = portafolio['pd_baseline'] * portafolio['lgd_baseline'] * portafolio['ead']
portafolio['ecl_stressed'] = np.where(
    portafolio['stage_stressed'] == 2,
    portafolio['pd_stressed'] * portafolio['lgd_stressed'] * portafolio['ead'] * factor_lifetime,
    portafolio['pd_stressed'] * portafolio['lgd_stressed'] * portafolio['ead']
)

# -----------------------------------------------------------------------------
# 7. RESUMEN
# -----------------------------------------------------------------------------
total_ead = portafolio['ead'].sum()
total_ecl_base = portafolio['ecl_baseline'].sum()
total_ecl_stressed = portafolio['ecl_stressed'].sum()
incremento_pct = ((total_ecl_stressed - total_ecl_base) / total_ecl_base) * 100
migran_stage2 = (portafolio['stage_stressed'] == 2).sum()

print("==================================================")
print("       RESUMEN EJECUTIVO - STRESS TESTING IFRS 9  ")
print("==================================================")
print(f"Total Cartera (EAD):       ${total_ead:,.2f}")
print(f"Reserva ECL Baseline:     ${total_ecl_base:,.2f}")
print(f"Reserva ECL Stressed:     ${total_ecl_stressed:,.2f}")
print(f"Impacto en Provisiones:   +{incremento_pct:.2f}%")
print(f"Migran a Stage 2:         {migran_stage2} cuentas")
print("==================================================")
print(lgd_por_subgrade)
print(portafolio[['lgd_baseline','lgd_stressed']].describe())

