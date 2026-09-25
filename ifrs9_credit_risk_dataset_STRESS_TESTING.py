# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:23:24 2026

@author: Juan Pablo Aguirre
"""

import numpy as np
import pandas as pd

# Carga de tu dataframe (asegúrate de que tenga tus 12 columnas)
df = pd.read_csv('credit_risk_dataset.csv')

# A. Calculamos la PD histórica real promedio por segmento de riesgo ('loan_grade')
pd_por_grade = df.groupby('loan_grade')['loan_status'].transform('mean')

# B. Definimos la LGD (al no estar 'recoveries' en tus 12 columnas, se fija en 0.45)
lgd_final = pd.Series(0.45, index=df.index)

# -----------------------------------------------------------------------------
# 1. CARGA Y MAPPING DE CARTERA REAL
# -----------------------------------------------------------------------------
portafolio = pd.DataFrame({
    'account_id'  : df.index + 1001,
    'ead'         : df['loan_amnt'],                         # Monto de la deuda
    'pd_baseline' : pd_por_grade.round(4),                      # PD observada por loan_grade
    'lgd_baseline': lgd_final.round(2),                        # LGD estándar de consumo (45%)
    'stage'       : np.where(df['loan_status'] == 1, 2, 1), # 1 = Default (Stage 2), 0 = Normal (Stage 1)
})

# print(portafolio.head(10))



# -----------------------------------------------------------------------------
# 2. DEFINICIÓN DE ESCENARIO DE ESTRÉS MACROECONÓMICO Y COEFICIENTES
# -----------------------------------------------------------------------------
# Shocks Macroeconómicos (Escenario Adverso Severo)

shock_pib = -0.05 # caida del PIB aumenta la PD
shock_desempleo = 0.04 # incremento del 4% en la tasa de desempleo


# Sensibilidades Econométricas (Betas del modelo)
beta_pd_pib = -1.2 # caida del PIB aumenta la PD
beta_pd_unemp = 0.8 # aumento del desempleo aumenta la PD
beta_lgd_pib = -0.6 # caida del PIB reduce recuperaciones (aumenta LGD)



# -----------------------------------------------------------------------------
# 3. FUNCIONES PURAS DE CÁLCULO (PD y LGD)
# -----------------------------------------------------------------------------


def compute_stressed_pd(pd_base, delta_pib, delta_unemp, b_pib, b_unemp):
    """Calcula la PD estresada mediante trasnformación Logit."""
    logit_base = np.log(pd_base / (1- pd_base))
    logit_stressed = logit_base + (b_pib * delta_pib) + (b_unemp * delta_unemp)
    return 1/ (1 + np.exp(-logit_stressed))

def compute_stressed_lgd(lgd_base, delta_pib, b_pib):
    """Calcula la LGD estresada manteniendo el resultado acotado entre (0, 1)."""
    logit_lgd = np.log(lgd_base / (1 - lgd_base))
    logit_lgd_stressed = logit_lgd + (b_pib *delta_pib)
    return 1 / (1+ np.exp(-logit_lgd_stressed))

# -----------------------------------------------------------------------------
# 4. EJECUCIÓN DEL MOTOR DE ESTRÉS
# -----------------------------------------------------------------------------
# a) Estrés de PD
portafolio['pd_stressed'] = compute_stressed_pd(
    pd_base= portafolio['pd_baseline'],
    delta_pib= shock_pib, 
    delta_unemp = shock_desempleo, 
    b_pib= beta_pd_pib,
    b_unemp= beta_pd_unemp)

# b) Estres de LGD
portafolio['lgd_stressed'] = compute_stressed_lgd(
    lgd_base= portafolio['lgd_baseline'],
    delta_pib = shock_pib,
    b_pib = beta_lgd_pib)

# print(portafolio.head())


# -----------------------------------------------------------------------------
# 5. RECLASIFICACIÓN DE ETAPAS (STAGING LOGIC / SICR)
# -----------------------------------------------------------------------------
# Criterio: Si la PD estresada supera el doble de la PD baseline, migra a Stage 2

portafolio['stressed_stage']= np.where(
    portafolio['pd_stressed'] > portafolio['pd_baseline'] * 2, 2, portafolio['stage']
)


# -----------------------------------------------------------------------------
# 6. CÁLCULO DE PÉRDIDA ESPERADA (ECL BASE VS ESTRÉS)
# -----------------------------------------------------------------------------

portafolio['ecl_baseline'] = portafolio['pd_baseline'] * portafolio['lgd_baseline'] * portafolio['ead']
portafolio['ecl_stressed'] = portafolio['pd_stressed'] * portafolio['lgd_stressed'] * portafolio['ead']


# -----------------------------------------------------------------------------
# 7. MÉTRICAS EJECUTIVAS Y RESULTADOS
# -----------------------------------------------------------------------------
total_ead = portafolio['ead'].sum()
total_ecl_base = portafolio['ecl_baseline'].sum()
total_ecl_stressed = portafolio['ecl_stressed'].sum()
incremento_pct = ((total_ecl_stressed - total_ecl_base)/ total_ecl_base) * 100


print("==================================================")
print("       RESUMEN EJECUTIVO - STRESS TESTING IFRS 9  ")
print("==================================================")
print(f"Total Cartera (EAD):       ${total_ead:,.2f}")
print(f"Reserva ECL Baseline:     ${total_ecl_base:,.2f}")
print(f"Reserva ECL Stressed:     ${total_ecl_stressed:,.2f}")
print(f"Impacto en Provisiones:   +{incremento_pct:.2f}%")
print("==================================================\n")

print("--- PRIMERAS 5 FILAS DEL DATASET COMPLETO ---")
columns_to_show = [
    'account_id', 'ead', 'pd_baseline', 'pd_stressed', 
    'lgd_baseline', 'lgd_stressed', 'ecl_baseline', 'ecl_stressed'
]
print(portafolio[columns_to_show].head())





