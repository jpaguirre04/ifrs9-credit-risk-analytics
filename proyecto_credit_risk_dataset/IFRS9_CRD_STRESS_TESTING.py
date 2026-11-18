# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:28:38 2026

@author: Juan Pablo Aguirre
"""


# https://www.kaggle.com/datasets/laotse/credit-risk-dataset

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. CARGA Y MAPPING - TUS COLUMNAS REALES
# -----------------------------------------------------------------------------
df_lc = pd.read_csv('credit_risk_dataset.csv')
# df_lc.columns

# DEFINICIÓN DEL TARGET
# En este dataset 'loan_status' ya viene como binario:
# 0 = Fully Paid / Good
# 1 = Default / Bad
df_lc['target'] = df_lc['loan_status'].astype(int)

# A. PD histórica real por loan_grade usando 'target'
# target = 1 si falló, 0 si pagó
pd_por_grado = df_lc.groupby('loan_grade')['target'].transform('mean')

# B. LGD por grado - ASUMIDA porque no tenemos 'recoveries'
# A pierde menos, G pierde más. Supuesto documentado.

lgd_por_grado_dict = {
    'A': 0.45,
    'B': 0.55,
    'C': 0.65,
    'D': 0.72,
    'E': 0.80,
    'F': 0.85,
    'G': 0.90
    }

lgd_final = df_lc['loan_grade'].map(lgd_por_grado_dict)
lgd_final = pd.Series(lgd_final).fillna(0.65).clip(0.10, 0.95)



portafolio = pd.DataFrame({
    'account_id' : df_lc.index +1001,
    'ead' : df_lc['loan_amnt'],
    'loan_grade': df_lc['loan_grade'],
    'pd_baseline': pd_por_grado.clip(0.001, 0.99).round(4),
    'lgd_baseline': lgd_final.round(2),
    'stage_base' : 1
    })


# -----------------------------------------------------------------------------
# 2. ESCENARIO - Lo puedes hacer más fuerte que el anterior
# -----------------------------------------------------------------------------
shock_pib = -0.05
shock_desempleo = 0.04
beta_pd_pib = -1.2
beta_pd_unemp = 0.8
beta_lgd_pib = -0.6


# -----------------------------------------------------------------------------
# 3. FUNCIONES - IGUAL QUE ANTES, NO CAMBIAN
# -----------------------------------------------------------------------------

def compute_stressed_pd(pd_base, delta_pib, delta_unemp, b_pib, b_unemp):
    pd_base = pd_base.clip(0.001, 0.999)
    logit_base = np.log(pd_base / (1 - pd_base))
    logit_stressed = logit_base + (b_pib * delta_pib) + (b_unemp * delta_unemp)
    return 1 /(1 + np.exp(-logit_stressed))


def compute_stressed_lgd(lgd_base, delta_pib, b_pib):
    lgd_base = lgd_base.clip(0.001, 0.99)
    logit_lgd = np.log(lgd_base / (1 - lgd_base))
    logit_lgd_stressed = logit_lgd + (b_pib * delta_pib)
    return 1 / (1 + np.exp(- logit_lgd_stressed))



# -----------------------------------------------------------------------------
# 4. MOTOR
# -----------------------------------------------------------------------------

portafolio['pd_stressed'] = compute_stressed_pd(
                 portafolio['pd_baseline'],
                 shock_pib, 
                 shock_desempleo, 
                 beta_pd_pib, 
                 beta_pd_unemp)

portafolio['lgd_stressed'] = compute_stressed_lgd(
                portafolio['lgd_baseline'],
                shock_pib,
                beta_lgd_pib
              )

# -----------------------------------------------------------------------------
# 5. SICR / STAGING
# -----------------------------------------------------------------------------

portafolio['stage_stressed'] = np.where(
    (portafolio['pd_stressed']>portafolio['pd_baseline']*1.20)|
    ((portafolio['pd_stressed'] - portafolio['pd_baseline'])>0.01),
    2, 1
    )

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
print(portafolio.groupby('loan_grade')[['pd_baseline','lgd_baseline','pd_stressed','lgd_stressed']].mean())




