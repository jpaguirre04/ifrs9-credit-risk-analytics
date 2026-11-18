# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:12:51 2026

@author: Juan Pablo Aguirre
"""


# https://www.kaggle.com/datasets/denychaen/lending-club-loans-rejects-data


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from IFRS9_LCLD_PD_Paso_2_forward_looking import df


# =============================================================================
# PASO 4: PD LIFETIME CON MATRIZ DE TRANSICIÓN (INTEGRADO)
# =============================================================================

# 1. Matriz Anual de Transición de Estados (Estados: A, B, C, D, E, F, G, Default)
# La última fila representa el Estado Absorbente (Default)
M_1ano = np.array([
    # A     B     C     D     E     F     G    Default
    [0.85, 0.08, 0.03, 0.01, 0.01, 0.00, 0.00, 0.02], # A
    [0.04, 0.78, 0.10, 0.03, 0.01, 0.01, 0.00, 0.03], # B
    [0.01, 0.05, 0.72, 0.12, 0.04, 0.01, 0.00, 0.05], # C
    [0.00, 0.02, 0.06, 0.68, 0.12, 0.03, 0.01, 0.08], # D
    [0.00, 0.01, 0.02, 0.07, 0.62, 0.12, 0.03, 0.13], # E
    [0.00, 0.00, 0.01, 0.03, 0.08, 0.58, 0.10, 0.20], # F
    [0.00, 0.00, 0.00, 0.01, 0.04, 0.08, 0.55, 0.32], # G
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]  # Default (Estado Absorbente)
])


# 2. Potencias de matriz para 36 meses (3 años) y 60 meses (5 años)

M_3anos = np.linalg.matrix_power(M_1ano, 3)
M_5anos = np.linalg.matrix_power(M_1ano, 5)


# 3. Extraer solo la primera letra ('A1' -> 'A')
df['grade'] = df['sub_grade'].astype(str).str.strip().str[0].str.upper()


# 4. Mapeos de PD acumulada a 3 y 5 años según la letra
mapa_pd_3y = {
    'A': M_3anos[0, -1], 'B': M_3anos[1, -1], 'C': M_3anos[2, -1],
    'D': M_3anos[3, -1], 'E': M_3anos[4, -1], 'F': M_3anos[5, -1], 
    'G': M_3anos[6, -1]
}

mapa_pd_5y = {
    'A': M_5anos[0, -1], 'B': M_5anos[1, -1], 'C': M_5anos[2, -1],
    'D': M_5anos[3, -1], 'E': M_5anos[4, -1], 'F': M_5anos[5, -1], 
    'G': M_5anos[6, -1]
}

# 5. Asignar vectorialmente según el plazo (term_months)
pd_3y = df['grade'].map(mapa_pd_3y)
pd_5y = df['grade'].map(mapa_pd_5y)


df['pd_lifetime'] = np.where(df['term_months']<= 36, pd_3y, pd_5y )


# =============================================================================
# CALIBRACIÓN PASO 4: MATRIZ PARA QUE CUADRE CON PD 12M FL
# =============================================================================

# 1. PD que dice tu matriz a 12 meses por grado
pd_matriz_12m = {
    'A': M_1ano[0, -1], 'B': M_1ano[1, -1], 'C': M_1ano[2, -1],
    'D': M_1ano[3, -1], 'E': M_1ano[4, -1], 'F': M_1ano[5, -1],
    'G': M_1ano[6, -1],
}


# Mapea esa PD de matriz a cada cliente
df['pd_matriz_12m'] = df['grade'].map(pd_matriz_12m)

# Evita división por cero
df['pd_matriz_12m'] = df['pd_matriz_12m'].clip(lower = 0.0001, upper = 0.999)

# 2. Factor de calibración por cliente
# Cuánto más riesgoso es en la realidad vs lo que dice la matriz
df['factor_calib_matriz'] = df['pd_12m_fl'] / df['pd_matriz_12m']


# 3. Función para calibrar la fila completa
# Esta es la calibración propiamente tal
def calibrar_fila(fila_matriz_original, pd_nueva):
          pd_vieja = fila_matriz_original[-1]
          # print(pd_vieja)
          # Si la PD vieja ya es (estado Default), no se toca
          if pd_vieja >= 0.999:
              return fila_matriz_original
          
          # Factor para re-escalar las probabilidades de no-default
          factor_no_default = (1- pd_nueva) / (1- pd_vieja)
          fila_calibrada = fila_matriz_original.copy()
          # print(fila_calibrada[:-1])
          fila_calibrada[:-1] = fila_calibrada[:-1] * factor_no_default
          fila_calibrada[-1] = pd_nueva
          # print(pd_nueva)
          
          # Clip final por seguridad
          return np.clip(fila_calibrada, 0, 1)
                   
          
# Ejemplo de uso para un cliente grado B con pd_12m_fl = 0.055
ejemplo_fila_B = M_1ano[1] # [0.04, 0.78, 0.10, 0.03, 0.01, 0.01, 0.00, 0.03]
ejemplo_pd_nueva = 0.055          
fila_calibrada_B = calibrar_fila(ejemplo_fila_B, ejemplo_pd_nueva)          
          
print("Fila B original, Default=3%:", ejemplo_fila_B)
print("Fila B calibrada, Default=5.5%:", fila_calibrada_B)
print("Suma fila calibrada:", fila_calibrada_B.sum()) # Siempre debe dar 1.0


# 4. Ahora tu PD Lifetime ya queda calibrada
# Recalculas M_3anos y M_5anos pero usando filas calibradas por cliente
# En la práctica bancaria no se hace 1 matriz por cliente, se hace 1 matriz por grado
# escalando el Default a la PD promedio del grado:


# PD promedio real por grado
pd_promedio_real_por_grado = df.groupby('grade')['pd_12m_fl'].mean()
# Y con eso calibras 7 matrices (una por grado) en vez de 32 mil


# =============================================================================
# PASO 5: GENERACIÓN DE TODOS LOS ESTADOS CALIBRADOS 12M + 3Y + 5Y
# =============================================================================

# --- 1. Calibra las 7 filas 12M ---
mapa_letras_idx = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}

matrices_cal_12m = {} # guarda las 7 matrices calibradas
filas_cal_12m = {}


for grado,idx in mapa_letras_idx.items():
       # print(grado, idx)
       pd_nueva = pd_promedio_real_por_grado.get(grado, M_1ano[idx, -1])
       fila_cal = calibrar_fila(M_1ano[idx], pd_nueva)

       filas_cal_12m[grado] = fila_cal # guardas la fila
       
       M_cal = M_1ano.copy()
       M_cal[idx] = fila_cal 
       matrices_cal_12m[grado] = M_cal  # guardas la MATRIZ COMPLETA 
       # print(matrices_cal_12m)
       print(f"Fila {grado} calibrada 12M: Default {M_1ano[idx,-1]:.2%} -> {fila_cal[-1]:.2%} | {np.round(fila_cal,4)}")



# --- 2. Matrices a 3 y 5 años calibradas ---
mapa_pd_3y_cal = {}
mapa_pd_5y_cal = {}


for grado, idx in mapa_letras_idx.items():
        M_3_cal = np.linalg.matrix_power(matrices_cal_12m[grado], 3)
        M_5_cal = np.linalg.matrix_power(matrices_cal_12m[grado], 5)

        mapa_pd_3y_cal[grado] = M_3_cal[idx, -1]
        mapa_pd_5y_cal[grado] = M_5_cal[idx, -1]


print("\n--- MAPEOS CALIBRADOS ---")
print("PD 3Y CAL:", mapa_pd_3y_cal)
print("PD 5Y CAL:", mapa_pd_5y_cal)

# df.columns

df['pd_lifetime_cal'] = np.where(df['term_months']<= 36, df['grade'].map(mapa_pd_3y_cal), df['grade'].map(mapa_pd_5y_cal))


df_PD = df


