# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:34:27 2023

@author: rcuppari
"""

import pickle
import pandas as pd 

## NOTES: subbasins 5 seems to work well

#path = 'E:/Global_RS/Results_country_basin'
#path = 'Results_subbasins5/Zambia' 
path = 'Results_biggest_basin/Zimbabwe3'
results_name = 'biggest_basin3' #'country_subbasins6'
## take whichever countries want to explore
def id_hydro_countries(pct_gen = 30, year = 2015):
    wb_hydro = pd.read_csv("Other_data/wb_hydro%.csv")
    gen_data = pd.read_csv("Other_data/eia_hydropower_data.csv")
    
    countries = wb_hydro[wb_hydro['2015'] > pct_gen][['Country Name']] 
    return countries


## 
#for country in countries['Country Name']:
    
#with open(path + "_gen_top_results_det", "rb") as handle: 
#     results_det = pickle.load(handle)
     
with open(path + "_gen_top_results", "rb") as handle: 
     results = pickle.load(handle)

#countries = id_hydro_countries()       
#results = {}
#for c in countries.iloc[:,0]:
#    print(c)
#    try: 
#        with open("Results/" + c + "top_results", "rb") as handle: 
#            results_new = pickle.load(handle)
#        if len(results_new) > len(results):
#            results = results_new
#            print(str(c))
#    except: print(str(c) + ' has failed')
     
## need a table to show the dictionary 
df = pd.DataFrame.from_dict(results, orient = 'index')
df.reset_index(inplace = True)
df.columns = ['country', 'data']

results_df = pd.DataFrame(columns = ['country', 'inputs', 'r2', 'pval', 'test_r2'])
for i in range(0, len(df)): 
    country = df['country'].iloc[i]
    unfolded = df.iloc[i,1]
    print(country)
    try: 
        inputs = unfolded['inputs']
        r2 = unfolded['r2']
        pval = unfolded['pval'] 
        test_r2 = unfolded['test_r2']
        new = pd.DataFrame([country, inputs[0], r2[0], pval[0], test_r2[0]]).T

    except: 
        print('failed :(')
        new = pd.DataFrame([country, 'nan', 'nan', 'nan', 'nan']).T  
    new.columns = ['country', 'inputs', 'r2', 'pval', 'test_r2'] 
    results_df = pd.concat([results_df, new], axis = 0)

results_df.to_csv("results_" + results_name + "_cleaned.csv")

####################################################################
## want to find  some ways to group countries
## ideas: 
    # quantile hydroelectricity 
    # presence of glaciers (tropical?) 
    # tropical versus not
    # continent
    # number of dams (index for single dam dominated?)
    # number of watersheds with hydro? 






