# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:39:11 2021

@author: groes
"""
import numpy as np
import pandas as pd
from scipy import stats

def run_correlation_analysis(df, score_column_stress, score_column_leadership,
                         lower_corr_threshold=-0.5, upper_corr_threshold=0.5):
    
    """
    Wrapper function that produces correlation coefficients for relationship
    between stress-related and leadership-related variables. 
    
    The function generate correlation matrices for variables with Gaussian
    and non-Gaussian distribution and splits them into the 4 returned series 
    objects 
    
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    score_column_stress : TYPE
        DESCRIPTION.
    score_column_leadership : TYPE
        DESCRIPTION.
    lower_corr_threshold : TYPE, optional
        DESCRIPTION. The default is -0.5.
    upper_corr_threshold : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    gaussian_matrix_top : Series
        
        Series object containing Pearson correlation coefficients above upper
        threshold
    
    gaussian_matrix_bottom : Series
        Series object containing Pearson correlation coefficients below lower
        threshold
    
    nongaussian_matrix_top : Series
        Series object containing Spearman correlation coefficients below lower
        threshold
    
    nongaussian_matrix_bottom : Series
        Series object containing Spearman correlation coefficients below lower
        threshold

    """
    
    # Creating two df's: One for questions/variables related to stress, one for
    # leadership
    df_stress = df[df["Topic Label"] == "Uoverskuelighed og stress"]
    df_leadership = df[df["Topic Label"] == "Ledelseskvalitet"]
    
    # Name of column containing the questions/variables related to stress and 
    # leadership
    column = "Question Label"
    
    # Stress variables - Using only work related stress variables
    stress_labels = ["Kun stress fra arbejdet blandt personer med stress",
              "Stress fra arbejdet blandt personer med stress", "Uoverskuelighed",
              "Arbejdsrelateret stress", "Stress"]
    
    # Leadership variables
    leadership_labels = list(df_leadership[column].unique())
    
    # Filter df based on stress and leadership variables
    df_stress_to_use = filter_df_by_label(df_stress, stress_labels)
    df_leadership_to_use = filter_df_by_label(df_leadership, leadership_labels)
    
    # Make stress and leadership df's where each column is the mean value for
    # a variable, and each row is a group (eg gender or occupation group)
    df_stress_to_use, _ = get_df_for_correlation_analysis(
        df_stress_to_use, stress_labels, column, score_column_stress
        ) 
    df_leadership_to_use, _ = get_df_for_correlation_analysis(
        df_leadership_to_use, leadership_labels, column, score_column_leadership
        )
        
    # Concatenating df's
    df_stress_leadership = pd.concat([df_stress_to_use, df_leadership_to_use],
                                     axis=1)
    
    # Getting variables (i.e. questions) that follow a gaussian and non-gaussian
    # distribution. Going to run Pearson correlation on gaussian,
    # and run Spearman on non-gaussian
    gaussian_variables, nongaussian_variables = variables_by_distribution_new(
        df_stress_leadership)
    
    # Getting correlation matrices for the gaussian and non-gaussian variables
    gaussian_matrix, nongaussian_matrix = create_correlation_matrix_new(
        df_stress_leadership, gaussian_variables, nongaussian_variables)
        
    # Slicing matrices so they only correlations between stress and leadership 
    # variables and not correlations between, for instance, two leadership 
    # variables
    gaussian_matrix = gaussian_matrix.iloc[:4,4:]
    nongaussian_matrix = nongaussian_matrix.iloc[:4,4:]
    
    # Making matrices that only contain variables with coefficients beyond 
    # thresholds
    gaussian_matrix_top, gaussian_matrix_bottom = correlations_above_threshold(
        gaussian_matrix, upper_corr_threshold, lower_corr_threshold)
    
    nongaussian_matrix_top, nongaussian_matrix_bottom = \
        correlations_above_threshold(
        nongaussian_matrix, upper_corr_threshold, lower_corr_threshold
        )
    
    return gaussian_matrix_top, gaussian_matrix_bottom, nongaussian_matrix_top,\
        nongaussian_matrix_bottom


def filter_df_by_label(df, list_of_labels):
    
    """
    Returns a dataframe that only contains rows where the "Group" column value
    is equal to one of the values of list_of_labels
    """
    
    length = len(df["Group"].unique())
    new_df = pd.DataFrame()
    for label in list_of_labels:
        sub_df = df[df["Question Label"] == label]
        # Making sure that all questions are answered by all groups
        if len(sub_df) == length:
            new_df = new_df.append(sub_df)
            
    return new_df 


def get_df_for_correlation_analysis(
        original_df, questions, question_column, score_column):
    """Makes df where each column is question/variable, and each row
    is an occupation, and the values are the mean scores for each question
    
    This function can also be used to generate df needed for comparing mean
    scores across occupations
    
    original_df : df from which rows and columns are extracted
    questions : list unique questions from the column by which values are 
    extracted question_column : name of column from which unique questions are 
    taken
    """
    length = len(original_df["Group"].unique())
    data_dictionary = {}

    for question in questions:
        values = list(original_df.loc[original_df[question_column] == question,\
                                      score_column])
        # Making sure that questions added are answered by all groups
        if len(values) == length:
            data_dictionary[question] = values
      
    df_correlation = pd.DataFrame(data_dictionary)

    return df_correlation, data_dictionary
    
    
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


def variables_by_distribution(original_df):
    """Testing for gaussian/normal distribution in order to determine whether to
    use Pearson (for gaussian) or Spearman (for non guassian).
    Shapiro is used over e.g. Kolmogorov-Smirnov test for alleged higher statistical
    power cf. https://www.spss-tutorials.com/spss-shapiro-wilk-test-for-normality/ """
    unique_questions = list(original_df["Spm Formulering"].unique())
    _, data_dictionary = get_df_for_correlation_analysis(original_df)
    shapiro_df = pd.DataFrame(columns=["question", "p_value"])
    
    for question in unique_questions:
        values = list(original_df.loc[original_df["Spm Formulering"] == question, 'Mean'])
        shapiro = stats.shapiro(values)
        shapiro_df.loc[len(shapiro_df)] = [question, shapiro.pvalue]
    
    normal_variables = shapiro_df.loc[shapiro_df["p_value"] >= 0.05]["question"]
    non_normal_variables = shapiro_df.loc[shapiro_df["p_value"] < 0.05]["question"]
    
    return normal_variables, non_normal_variables

def variables_by_distribution_new(df):
    """Testing for gaussian/normal distribution in order to determine whether to
    use Pearson (for gaussian) or Spearman (for non guassian).
    Shapiro is used over e.g. Kolmogorov-Smirnov test for alleged higher statistical
    power cf. https://www.spss-tutorials.com/spss-shapiro-wilk-test-for-normality/ 
    
    Return series objects containing variable names"""
    
    columns = df.columns
    shapiro_df = pd.DataFrame(columns=["question", "p_value"])
    
    for column in columns:
        values = list(df[column])
        shapiro = stats.shapiro(values)
        shapiro_df.loc[len(shapiro_df)] = [column, shapiro.pvalue]
    
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

    # Creating Pearson correlation matrix for normally distributed variables 
    # (variables = questions)
    df_normal_variables = df_correlation.drop(non_normal_variables, axis=1)
    correlation_matrix_normal = df_normal_variables.corr()

    # Creating Spearman correlation matrix for variables with non-normal 
    # distribution (variables = questions)
    df_non_normal_variables = df_correlation.drop(normal_variables, axis=1)
    correlation_matrix_non_normal = df_non_normal_variables.corr(method="spearman")
        
    return correlation_matrix_normal, correlation_matrix_non_normal

def create_correlation_matrix_new(df, gaussian_variables, nongaussian_variables):
    """
    Generates correlation matrix based on distribution of variables 
    (gaussian vs non-gaussian) Pearson correlation is used if gaussian, 
    Spearman otherwise. Variables are the questions in the dataset.

    Parameters
    ----------
    df_original : pandas df
        df made from arbejdsmarkeds_analyse.xlsx first 4001 rows
    
    Returns
    -------
    correlation_matrix_normal, correlation_matrix_non_normal : pandas df

    """

    # Creating Pearson correlation matrix for normally distributed variables (variables = questions)
    df_normal_variables = df.drop(nongaussian_variables, axis=1)
    correlation_matrix_normal = df_normal_variables.corr()

    # Creating Spearman correlation matrix for variables with non-normal distribution (variables = questions)
    df_non_normal_variables = df.drop(gaussian_variables, axis=1)
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
        series with coefficients above upper threshold.
    correlation_matrix_bottom : TYPE
        series with coefficients below lower threshold.

    """
    unique_corr_pairs = correlation_matrix.unstack().dropna() 
    correlation_matrix_sorted = unique_corr_pairs.sort_values()
    correlation_matrix_top = correlation_matrix_sorted[correlation_matrix_sorted > upper]
    correlation_matrix_bottom = correlation_matrix_sorted[correlation_matrix_sorted < lower]
    
    return correlation_matrix_top, correlation_matrix_bottom


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
    return df

def get_n_smallest_values(df, column_name, n):
    df = df.nsmallest(n, column_name, keep="all")
    return df


