# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 17:26:30 2021

@author: groes
"""

import metoder
import pandas as pd

def unittest_get_nlargest():
    column1 = [1,2,3,4,5,6,7,8,9]
    column2 = ["a", "a", "a", "a", "a", "a", "a", "a", "a"]
    
    df = pd.DataFrame(list(zip(column1, column2)), columns=["Values", "Names"])
    
    largest = metoder.get_n_largest_values(df, "Values", 3)
    
    assert largest.iloc[:][0][0] == 9