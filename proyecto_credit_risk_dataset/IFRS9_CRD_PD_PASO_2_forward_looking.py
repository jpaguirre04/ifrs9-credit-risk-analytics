# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:31:26 2026

@author: Juan Pablo Aguirre
"""


# https://www.kaggle.com/datasets/laotse/credit-risk-dataset

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from IFRS9_CRD_PD_PASO_1_regresion_logistica import df


# =============================================================================
# PASO 3: FORWARD-LOOKING CON 3 ESCENARIOS (NIIF 9)
# =============================================================================

# 1. Definición de escenarios macroeconómicos (ej. Proyección Tasa de Desempleo)
# Tomando como desempleo base histórico / proyectado: 8.0%

escenarios = pd.DataFrame({
                    'escenario': ['Base', 'Adverso', 'Optimista'],
                    'prob': [0.5, 0.3, 0.2],
                    'desempleo': [8.0, 10.5, 6.5] # Proyecciones macroeconómicas a 12 meses 
              })

# 2. Función de ajuste por sensibilidad / elasticidad macroeconómica


def ajustar_pd(pd_ttc, desemplo_esc, desempleo_base= 8.0, elasticidad= 0.15):
    """Ajusta la PD TTC mediante un factor de escala según el escenario macroeconómico.

    - elasticidad: Cambio % en la PD por cada punto de variación en el desempleo.
    """
    factor = 1 + elasticidad * (desemplo_esc - desempleo_base)
    # Clip para asegurar que las probabilidades se mantengan en el rango [0, 1]
    return (pd_ttc * factor).clip(0, 1)
    
    

# 3. Cálculo de la PD bajo cada escenario
for esc in escenarios['escenario']:
     desempleo_val = escenarios.loc[escenarios['escenario']== esc, 'desempleo'].values[0]
     df[f'pd_12m_fl_{esc.lower()}'] = ajustar_pd(df['pd_12m_ttc'], desempleo_val)



# 4. PD Final Forward-Looking Ponderada por Probabilidad de Ocurrencia
# de los Escenarios (MÉTRICA FINAL NIIF 9)

df['pd_12m_fl'] = 0
for i, row in escenarios.iterrows():
    # print(i, row)
    esc_name = row['escenario'].lower()
    df['pd_12m_fl'] += row['prob'] * df[f'pd_12m_fl_{esc_name}']

