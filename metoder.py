# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:39:11 2021

@author: groes
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import seaborn as sns
from scipy import stats

def get_high_and_low_mean_pr_question(original_df, question, n):
    """
    Get bottom and top n mean scores for question

    Parameters
    ----------
    original_df : TYPE
        DESCRIPTION.
    question : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    bottom : pandas df
        df with two columns: "Group" (occupation) "Mean" (mean score for question)
        Values in "Mean" are bottom n mean scores for question
    top : pandas df
        df with two columns: "Group" (occupation) "Mean" (mean score for question)
        Values in "Mean" are top n mean scores for question
    """
    
    question_df = original_df[original_df["Spm Formulering"] == question]
    columns = question_df.columns
    dont_drop = ["Group", "Mean"]
    drop_columns = [i for i in columns if i not in dont_drop]
    question_df = question_df.drop(drop_columns, axis=1)
    question_df.sort_values(by=["Mean"], ascending=False, inplace=True)
    bottom = question_df.tail(n)
    top = question_df.head(n)
    
    return bottom, top

def get_df_for_correlation_analysis(original_df):
    """Makes df where each column is the mean answer for a question, and each row
    is an occupation
    
    This function can also be used to generate df needed for comparing mean
    scores across occupations
    """
    unique_questions = list(original_df["Spm Formulering"].unique())
    unique_occupations = list(original_df["Group"].unique())

    data_dictionary = {}
    previous_occupation_order = []
    number_false = 0
    
    for question in list(unique_questions):
        values = list(original_df.loc[original_df["Spm Formulering"] == question, 'Mean'])
        occupation_order = list(original_df.loc[original_df["Spm Formulering"] == question, 'Group'])
        # Chceking if the occupations are always listed in the same order
        if previous_occupation_order != occupation_order:
            number_false += 1
        if number_false > 1:
            print("Occupation order not identical for all questions")
        previous_occupation_order = occupation_order.copy()
        data_dictionary[question] = values
    
    data_dictionary["occupation"] = unique_occupations    
    df_correlation = pd.DataFrame(data_dictionary)
    df_correlation.index = occupation_order
    
    return df_correlation, data_dictionary

def variables_by_distribution(original_df):
    """Testing for gaussian/normal distribution in order to determine whether to
    use Pearson (for gaussian) or Spearman (for non guassian).
    Shapiro is used over e.g. Kolmogorov-Smirnov test for alleged higher statistical
    power cf. https://www.spss-tutorials.com/spss-shapiro-wilk-test-for-normality/ """
    unique_questions = list(original_df["Spm Formulering"].unique())
    _, data_dictionary = get_df_for_correlation_analysis(original_df)
    shapiro_df = pd.DataFrame(columns=["question", "p_value"])
    
    for question in unique_questions:
        values = list(original_df.loc[original_df["Spm Formulering"] == question, 'Mean']) #data_dictionary[question]
        shapiro = stats.shapiro(values)
        shapiro_df.loc[len(shapiro_df)] = [question, shapiro.pvalue]
    
    normal_variables = shapiro_df.loc[shapiro_df["p_value"] >= 0.05]["question"]
    non_normal_variables = shapiro_df.loc[shapiro_df["p_value"] < 0.05]["question"]
    
    return normal_variables, non_normal_variables

def create_correlation_matrix(original_df):
    """
    Generates correlation matrix based on distribution of variables (gaussian vs non-gaussian)
    Pearson correlation is used if gaussian, Spearman otherwise. Variables are 
    the questions in the dataset.

    Parameters
    ----------
    df_original : pandas df
        df made from arbejdsmarkeds_analyse.xlsx first 4001 rows
    
    Returns
    -------
    correlation_matrix_normal, correlation_matrix_non_normal : pandas df

    """
    df_correlation, _ = get_df_for_correlation_analysis(original_df)
    normal_variables, non_normal_variables = variables_by_distribution(original_df)

    # Creating Pearson correlation matrix for normally distributed variables (variables = questions)
    df_normal_variables = df_correlation.drop(non_normal_variables, axis=1)
    correlation_matrix_normal = df_normal_variables.corr()

    # Creating Spearman correlation matrix for variables with non-normal distribution (variables = questions)
    df_non_normal_variables = df_correlation.drop(normal_variables, axis=1)
    correlation_matrix_non_normal = df_non_normal_variables.corr(method="spearman")
        
    return correlation_matrix_normal, correlation_matrix_non_normal


def correlations_above_threshold(correlation_matrix, upper, lower):
    """
    Returns two pandas series containing unique variables with a correlation 
    above upper and below lower

    Parameters
    ----------
    correlation_matrix : pandas correlation matrix generated with 
    create_correlation_matrix()
    
    upper : FLOAT
        Upper threshold which correlation must exceed in order to be included
        in series
        
    lower : FLOAT
        Lower threshold (negative float) which correlation must be smaller than
        in order to be included in series.

    Returns
    -------
    correlation_matrix_top : TYPE
        DESCRIPTION.
    correlation_matrix_bottom : TYPE
        DESCRIPTION.

    """
    # Getting correlations that are in upper triangle in correlation matrix
    upper_corr_mat = correlation_matrix.where( 
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool)) 
    
    unique_corr_pairs = upper_corr_mat.unstack().dropna()
    correlation_matrix_sorted = unique_corr_pairs.sort_values()
    correlation_matrix_top = correlation_matrix_sorted[correlation_matrix_sorted > upper]
    correlation_matrix_bottom = correlation_matrix_sorted[correlation_matrix_sorted < lower]
    
    return correlation_matrix_top, correlation_matrix_bottom







#avg_per_question = df_brancher.loc[df_brancher['Spm Formulering'] == unique_questions[0], 'Mean']

def calculate_avg_per_level(df, column, level):
    """ Calculates average of mean scores for a specific level in a column """
    level_average = df.loc[df[column] == level, 'Mean']
    
    return np.sum(level_average) / len(level_average)
    
def collect_avg_per_level(df, column_name):
    """ Return df with 2 columns: Level, which contains all the unique 
    questions in type1 df, and Mean which is the mean score of all the answers 
    for the given question """
    # Get unique values/levels in df[column_name]
    unique_levels = list(df[column_name].unique()) # e.g. ["Spm Formulering"]
    averages = [calculate_avg_per_level(df, column_name, level) for level in unique_levels]
    return pd.DataFrame(list(zip(unique_levels, averages)), columns=["Level", "Mean"])
    
def get_n_largest_values(df, column_name, n):
    df = df.nlargest(n, column_name, keep="all")
    display(df)
    return df

def get_n_smallest_values(df, column_name, n):
    df = df.nsmallest(n, column_name, keep="all")
    display(df)
    return df


"""
test = df_brancher_type1.loc[df["Spm Formulering"] == unique_questions[0]]





column_name = "Spm Formulering"
n = 5
df_mean = collect_avg_per_question(df_brancher_type1, "Spm Formulering")
_ = get_n_largest_values(df_mean, "Mean", n)
_ = get_n_smallest_values(df_mean, "Mean", n)


# Gennemsnit pr fagområde på tværs af alle spørgsmål - this is the same as df_totals
# avg_per_occupation = collect_avg_per_level(df_brancher_type1, "Group")
# Gennemsnit pr spm på tværs af alle spørgsmål
avg_per_question = collect_avg_per_level(df_brancher_type1, "Spm Formulering")
# Extract total fra df
df_totals = pd.DataFrame(df_brancher_type1.loc[df_brancher_type1["Group"] == "Total", 'Mean'])
df_totals.insert(0, "Spoergsmaal", unique_questions)





math.isnan(unique_questions[66])

def remove_nans_from_list(list_object):
    list_copy = list_object.copy()
    indices_to_remove = []
    idx = -1
    for element in list_object:
        idx += 1
        try:
            if math.isnan(element) or element == "Nan" or element == "nan" or element == "NaN":
                list_copy.remove(idx)
        except:
            print("didnt work")
    return list_copy
    
    
unique_questions = remove_nans_from_list(unique_questions)
            

unique_questions = [question for question in unique_questions if "Spørgsmål" in question]
unique_questions[0]






# Plotting to check distribution / linearity
variable = pd.Series(df_correlation.iloc[:,0])
variable.hist()
variable.plot(kind = "box")
sns.pairplot(df_correlation)








    
test = 1    

columns = ["Occupation"]
columns.extend(unique_occupations)

df_correlation = pd.DataFrame(columns=columns)
df_correlation = pd.DataFrame()
df["Occupation"] = unique_occupations


test = [[1,2,3], [3,4,5]]
testdf = pd.DataFrame(test, columns=["C1", "C2"])



# Plot of mean of one question per occupation
df = df_brancher_type1.loc[df["Spm Formulering"] == unique_questions[0]]
df_smallest = get_n_smallest_values(df, "Mean", 10)
df_largest = get_n_largest_values(df, "Mean", 10) 
#x = np.arange(len(test))
#height = test["Mean"]
objects = list(df["Group"])
x_pos = np.arange(len(objects))
mean_values = df["Mean"]

plt.bar(x_pos, mean_values, align='center', alpha=0.5)
plt.xticks(x_pos, objects)
plt.ylabel("Mean")
plt.xlabel("Occupation")
plt.show
plt.title(unique_questions[0])

#del unique_questions[66] ## 66th element was nan




    """
