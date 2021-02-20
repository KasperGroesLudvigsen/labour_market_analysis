# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:30:12 2021

@author: groes
"""
import metoder
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table 
from IPython.display import display, HTML
from scipy import stats

# type 1 = række 1 t.o.m. 4105
df_brancher = pd.read_excel("arbejdsmarkedsanalyse_brancher.xlsx") 
original_df = df_brancher.head(4104)

# Inspecting which variables (i.e. questions) are correlated
correlation_matrix_normal, correlation_matrix_non_normal = metoder.create_correlation_matrix(original_df)
correlation_matrix_normal_upper, correlation_matrix_normal_lower = metoder.correlations_above_threshold(correlation_matrix_normal, 0.7, -0.6)

###############################
### Looking at correlation between "den samlede score for ledelseskvalitet" and others
ledelseskvalitet_corr = correlation_matrix_normal.loc["Den samlede score for ledelseskvalitet"]
# Visualizing df
ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
table(ax, ledelseskvalitet_corr)  # where df is your data frame
ledelseskvalitet_strong_corr = ledelseskvalitet_corr[ledelseskvalitet_corr < -0.6]
ledelseskvalitet_strong_corr = ledelseskvalitet_strong_corr.append(
    ledelseskvalitet_corr[ledelseskvalitet_corr > 0.7])
ledelseskvalitet_strong_corr = ledelseskvalitet_strong_corr[ledelseskvalitet_strong_corr < 1]
display(ledelseskvalitet_strong_corr)

### Hvor vigtigt er dit arbejde for dig foruden indkomsten?
job_importance_bottom, job_importance_top = metoder.get_high_and_low_mean_pr_question(
    original_df, "I hvilken grad er dit arbejde vigtigt for dig (foruden indkomsten)?", 10)

### Ledelseskvalitet
leadership_quality_bottom, leadership_quality_top = metoder.get_high_and_low_mean_pr_question(
    original_df, "Den samlede score for ledelseskvalitet", 10) 


##### Connection between leadership quality and general wellbeing
unique_occupations = list(original_df["Group"].unique())
unique_questions = list(original_df["Spm Formulering"].unique())

leadership_quality_all, _ = metoder.get_high_and_low_mean_pr_question(
    original_df, "Den samlede score for ledelseskvalitet", len(unique_occupations))

unique_questions.remove("Den samlede score for ledelseskvalitet")
df_all_except_leadership, _ = metoder.get_high_and_low_mean_pr_question(
        original_df, unique_questions[0], len(unique_occupations))

for question in unique_questions[1:]:
    question_all_replies, _ = metoder.get_high_and_low_mean_pr_question(
        original_df, question, len(unique_occupations))
    #if type(df_all_except_leadership.iloc[0,0]) == "str":
    #    print(question)
    #    break
    #df_all_except_leadership["Mean"] = df_all_except_leadership["Mean"] + question_all_replies["Mean"]
    df_all_except_leadership = df_all_except_leadership.append(question_all_replies)

group_means = []
for occupation in unique_occupations:
    df = df_all_except_leadership[df_all_except_leadership["Group"] == occupation]
    group_means.append(df["Mean"].mean())

df_mean_per_occupation = pd.DataFrame()
df_mean_per_occupation["Group"] = unique_occupations
df_mean_per_occupation["Mean"] = group_means 

# Fitting
x = leadership_quality_all["Mean"].values
y = df_mean_per_occupation["Mean"].values
print(stats.shapiro(x)) # it's normally distributed
print(stats.shapiro(y)) # it's normally distributed

pearson_rho, pvalue = stats.pearsonr(x, y)

#df_all_except_leadership["Mean"] = df_all_except_leadership["Mean"] / len(unique_questions)
    
# are the two variables (overall mean and leadership quality) normally distributed
shapiro = stats.shapiro(values)


leadership_quality_all.iloc[0,1] = leadership_quality_all["Mean"] + leadership_quality_all["Mean"]

