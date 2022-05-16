# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:18:30 2020

@author: Micha.Vardy
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor

## import data
data = pandas.read_excel('Temp_calc_2.xlsx',None)['material'] 

##### Constants ###########

A=6
h_w = 4.3
T_es1 = 32.8 + 273.15
T_infty = 17 + 273.15


################ GD resistor system  ############
r_2_GD = data.iloc[1,:]['Resistance [K/W]']
r_3_GD = data.iloc[2,:]['Resistance [K/W]']
################ Plasan resistor system  ############
r_2_Pl = data.iloc[5,:]['Resistance [K/W]']
r_3_Pl = data.iloc[6,:]['Resistance [K/W]']
r_4_Pl = data.iloc[7,:]['Resistance [K/W]']


############## Matrix_Operations ########################
def inv(r):
    res_inv = 1 / r
    return (res_inv)

def matrix_inv(matrix):
    mat_inv = np.linalg.inv(matrix)
    return (mat_inv)

def matrix_solution(mat_inv , mat_res):
    mat_sol = mat_inv * mat_res
    return (mat_sol)

def mat_2_list(matrix):
    mat_list = [32.8]
    for i in range(len(matrix)):
        mat_list.append(round(matrix[(i,0)]-273.15 , 2) )
    return(mat_list)

def cumulative_thickness(thickness_list):
    cumulative = [thickness_list[0]]
    for i in thickness_list[1:]:
        c = i + cumulative[-1]
        cumulative.append(c)
    return(cumulative)


############## GD_Linear_Equation_Matrix ########################
def mat_GD():
    
    r_2 = r_2_GD
    r_3 = r_3_GD
    
    mat2 = np.matrix([
            [ inv(r_2) + inv(r_3), -inv(r_3)],
            [inv(r_3) , -inv(r_3) - h_w * A ]
            ])
    
    return (mat2)

############## GD_Solution_Vector ########################
def sol_vec_GD():
    
    r_2 = r_2_GD
    r_3 = r_3_GD
    
    vec2 = np.matrix([
            [inv(r_2) * T_es1],
            [-h_w * A * T_infty]
            ])
    
    return(vec2)


############## Plasan_Matrix_Function ########################
def mat_Plasan():
    
    r_2 = r_2_Pl
    r_3 = r_3_Pl
    r_4 = r_4_Pl
    
    mat3 = np.matrix([
            [inv(r_2) + inv(r_3) ,-inv(r_3), 0 ],
            [inv(r_3) , -inv(r_3) - inv(r_4) , inv(r_4)],
            [0 , inv(r_4) , -inv(r_4) - h_w * A]
            ])
    
    return(mat3)

############## Plasan_Solution_Vector ########################
def sol_vec_Plasan():
    
    r_2 = r_2_Pl
    r_3 = r_3_Pl
    r_4 = r_4_Pl
    
    vec3 = np.matrix([
            [inv(r_2) * T_es1],
            [0],
            [-h_w * A * T_infty]
            ])
    
    return(vec3)  
    
############## GD_Solution ########################
def temperature_distribution_GD():
    
    mat2 = mat_GD()
    vec2 = sol_vec_GD()
    mat_inv = matrix_inv(mat2)
    mat_sol = matrix_solution(mat_inv , vec2)
    sol = mat_2_list(mat_sol)
    return (sol)

    
############## Plasan_Solution ########################
def temperature_distribution_Plasan():
    
    mat3 = mat_Plasan()
    vec3 = sol_vec_Plasan()
    mat_inv = matrix_inv(mat3)
    mat_sol = matrix_solution(mat_inv , vec3)
    sol = mat_2_list(mat_sol)
    return (sol)

############## Solution Vizualization ########################
def viz():
    
    # slice dataframes
    gd_df = data.iloc[1:4,:]
    pl_df = data.iloc[5:9,:]
    
    # thickness
    gd_th = list(gd_df['Thickness [m]'])
    pl_th =list(pl_df['Thickness [m]'])
    
    # cumulative thickness
    gd_cum = [i*1000 for i in cumulative_thickness(gd_th) ]
    pl_cum = [i*1000 for i in cumulative_thickness(pl_th)] 
    
    # temperature distribution 
    gd_dist = temperature_distribution_GD()
    pl_dist = temperature_distribution_Plasan()
    
    # plot
    plt.plot(gd_cum, gd_dist , label ='GD:D3', linewidth=3)
    plt.plot(pl_cum, pl_dist , label ='Plasan:D3', linewidth=3)
    plt.legend()
    plt.title('Temperature Distribution', fontsize=18) 
    plt.xlabel('distance from Base Hull [mm]', fontsize=12)
    plt.ylabel('Temperature [°C]', fontsize=12)
    plt.show()  

def write_to_excel():

    # slice dataframes
    gd_df = data.iloc[1:4,:]
    pl_df = data.iloc[5:9,:]
    
    # thickness
    gd_th = list(gd_df['Thickness [m]'])
    pl_th =list(pl_df['Thickness [m]'])
    
    # cumulative thickness
    gd_cum = [i*1000 for i in cumulative_thickness(gd_th) ]
    pl_cum = [i*1000 for i in cumulative_thickness(pl_th)] 
    
    # temperature distribution 
    gd_dist = temperature_distribution_GD()
    pl_dist = temperature_distribution_Plasan()
    
    # Label
    gd_label = ['GD_D3'] * len(gd_df)
    pl_label = ['Plasan_D3'] * len(pl_df)
    
    result_data_dict = {
            'Label': gd_label + pl_label,
            'Layer Material': list(gd_df['Layer Material']) + list(pl_df['Layer Material']),
            'Cumulative thickness [mm]': gd_cum + pl_cum,
            'Temperature Distribution [°C]': gd_dist + pl_dist
            }
    
    result_df = pandas.DataFrame(data = result_data_dict)
    result_df.to_excel('results_3.xlsx', engine='xlsxwriter')

print(temperature_distribution_Plasan())
print(temperature_distribution_GD())