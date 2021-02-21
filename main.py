# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:30:12 2021

@author: groes
"""

import pandas as pd
import metoder

# Load dataset and split into stress related questions and leadership related questions
df_brancher = pd.read_excel("arbejdsmarkedsanalyse_brancher.xlsx") 
df_koen_alder = pd.read_excel("arbejdsmarkedsanalyse_koen_alder.xlsx") 

# Investigating correlations between stress and leadership variables across all occupations
brancher_gaussian_matrix_top, brancher_gaussian_matrix_bottom, \
    brancher_nongaussian_matrix_top, brancher_nongaussian_matrix_bottom = \
        metoder.run_correlation_analysis(df_brancher, "Score", "Mean")

# Investigating correlations between stress and leadership variables across all genders
gender_gaussian_matrix_top, gender_gaussian_matrix_bottom, \
    gender_nongaussian_matrix_top, gender_nongaussian_matrix_bottom = \
        metoder.run_correlation_analysis(df_koen_alder, "Score", "Score")        
    

# There are some strong correlations between stress and leadership on the gender level,
# so now I'm going to break the correlation analysis down to the male and female levels

# Investigating correlations between stress variables and leadership variables among female groups
female_groups = [group for group in df_koen_alder["Group"].unique() if "Kvinder" in group]
female_groups_df = df_koen_alder[df_koen_alder["Group"].isin(female_groups)]

female_gaussian_matrix_top, female_gaussian_matrix_bottom, \
    female_nongaussian_matrix_top, female_nongaussian_matrix_bottom = \
        metoder.run_correlation_analysis(female_groups_df, "Score", "Score")  

# Investigating correlations between stress variables and leadership variables among male groups
male_groups = [group for group in df_koen_alder["Group"].unique() if "Mænd" in group]
male_groups_df = df_koen_alder[df_koen_alder["Group"].isin(male_groups)]

male_gaussian_matrix_top, male_gaussian_matrix_bottom, \
    male_nongaussian_matrix_top, male_nongaussian_matrix_bottom = \
        metoder.run_correlation_analysis(male_groups_df, "Score", "Score")  
