# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:47:31 2024

@author: rcuppari
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import math 
import geopandas as gpd
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from cleaning_funcs2 import regs

def predict_countries(country, hybas2, country_gen, reg_inputs, coefs, variables): 
    if 'watershed' in hybas2:
        hybas = 'watersheds'
    else: hybas = hybas2
    
    try: 
        inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data.csv").iloc[:,1:]
    except: 
        try: 
            inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data_{hybas}.csv").iloc[:,1:]
        except: 
            try: 
                inputs = pd.read_csv(f"Results_all/Inputs_{hybas2}/{country}_input_data_{hybas}.csv").iloc[:,1:]
            except: 
                inputs = 'failed'
                print(f"failed to find {country}, {hybas}")
            
    if isinstance(inputs, pd.DataFrame): 
        inputs.set_index('year', inplace = True)
        inputs = inputs.merge(country_gen, on = 'year')
    
    try:         
         inputs2 = inputs[variables[1:]]
         inputs2['year'] = inputs['year']
         
         predicted = float(coefs[0])  # Initialize prediction with the intercept term
         
         for i in np.arange(0, len(variables)-1):
             i = i + 1
             predicted += float(coefs[i]) * inputs2[variables[i]]
        
    except: 
        inputs2 = inputs[variables[1:]]
        inputs2['year'] = inputs['year']

        predicted = pd.DataFrame(np.repeat(-9,20))
    
    
    predicted = pd.DataFrame(predicted)
    predicted.columns = [country]
    predicted = pd.concat([predicted, inputs2['year']], axis = 1)
    
    return predicted

def predict_countries_mult(country, hybas2, country_gen, reg_inputs, coefs, variables): 
    
    input_vals = pd.DataFrame(columns = reg_inputs)
    
    for num, var in enumerate(hybas2): 
        ## first need to retrieve the different datasets
#        print(var)
#        print(num)
        
        if 'watershed' in var:
            hybas = 'watersheds'
        else: hybas = var
        
        try: 
            inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data.csv").iloc[:,1:]
        except: 
            try: 
                inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data_{hybas}.csv").iloc[:,1:]
            except: 
                try: 
                    inputs = pd.read_csv(f"Results_all/Inputs_{var}/{country}_input_data_{hybas}.csv").iloc[:,1:]
                except: 
                    inputs = 'failed'
                    print(f"failed to find {country}, {hybas}")
        
        clean_name = reg_inputs[num+1].replace('_y','')
        clean_name = clean_name.replace('_x','')
        var_needed = inputs[clean_name]
        
        input_vals.iloc[:,num+1] = var_needed
        

    input_vals['const'] = coefs[0]
    input_vals['year'] = inputs['year']


    if isinstance(input_vals, pd.DataFrame): 
        #print('True')
        input_vals.set_index('year', inplace = True)
        input_vals = input_vals.merge(country_gen, on = 'year')
    
             
    inputs2 = input_vals[variables[1:]]
    inputs2['year'] = input_vals['year']
   
    predicted = pd.DataFrame(index = np.arange(0, len(inputs2)), 
                             columns = [country]) 
   
    predicted.iloc[:,0] = float(coefs[0])  # Initialize prediction with the intercept term
   
    for i in np.arange(1, len(variables)):
        print(coefs[i])
        print(variables[i])
        predicted.iloc[:,0] += float(coefs[i]) * inputs2[variables[i]]
  
    #predicted = pd.DataFrame(predicted)
    predicted.columns = [country]
    predicted = pd.concat([predicted, inputs2['year']], axis = 1)
    
    return predicted


def make_scatters(predictions, gen_data2, r2, name = ''): 
    ncols = 3
    nrows = math.ceil(len(predictions.columns)/ncols)
    
    fig = plt.figure()
    gs = plt.GridSpec(nrows, ncols, wspace = 0.2, hspace = .4) 
    axs = []
    for c, num in zip(predictions.columns, range(1, len(predictions.columns)+1)):
        #print(c)
        #print(num)
        axs.append(fig.add_subplot(gs[num-1]))
        im = axs[-1].scatter(predictions[c], gen_data2[c], c = r2[c], vmin = .3, vmax = 1)
        axs[-1].plot([gen_data2[c].min(), gen_data2[c].max()],
                    [gen_data2[c].min(), gen_data2[c].max()]) 
        axs[-1].set_xlabel("Predicted", fontsize = 14)
        axs[-1].set_ylabel("Observed", fontsize = 12)
        axs[-1].set_title(f"{c}", fontsize = 14)
        #plt.tight_layout()
    
    axs.append(fig.add_subplot(gs[nrows-1, 2:len(predictions.columns)]))
    fig.colorbar(im, cax = axs[-1], orientation = 'horizontal')
    axs[-1].set_xlabel("r2",fontsize = 14)
    fig.savefig(fname = f"Figures/indices{name}.jpg")

def make_numeric(df): 
    for c in range(0, len(df.columns)): 
        df.iloc[:,c] = pd.to_numeric(df.iloc[:,c], errors = 'coerce')        
    return df

def detrend(gen_data): 
    
    year_predictions = pd.DataFrame()
    det_gen = pd.DataFrame()
    det_gen['year'] = gen_data['year']
    det_gen.year = pd.to_numeric(det_gen.year).astype(int)
    
    for c, country in enumerate(gen_data.columns[1:]):
        print(country)
        data = gen_data.loc[:,['year',country]]
        ## if there is no data (or data is equal to 0), drop 
        ## nans will be dropped too 
        data.iloc[:,0] = pd.to_numeric(data.iloc[:,0], errors = 'coerce')
        data.iloc[:,1] = pd.to_numeric(data.iloc[:,1], errors = 'coerce')
        data.dropna(inplace = True, axis = 0)
        
        if len(data) > 0:
            ## get to the juicy part
            y = pd.to_numeric(data.iloc[:,1]) ## outcome should be second column
            X = sm.add_constant(pd.to_numeric(data.iloc[:, 0])) ## year should be first
            
            est_year = sm.OLS(y, X)
            est_year2 = est_year.fit()
        #    print(est_year2.summary())
        #    print(est_year2.rsquared)
        
            year_pred = est_year2.predict(X)
            year_predictions = pd.concat([year_predictions, year_pred], axis = 1)
           
        else: 
            year_pred = pd.DataFrame(np.repeat('nan', len(year_predictions)))
            year_predictions = pd.concat([year_predictions, year_pred], axis = 1)
        
        det_data = pd.DataFrame(pd.to_numeric(data.iloc[:,1]) - year_pred)
        det_data['year'] = data.iloc[:,0].astype(int)
        det_data.columns = [country,'year']
        det_gen = det_gen.merge(det_data, on = 'year', how = 'outer')
    
                
    det_gen.columns = gen_data.columns
    year_predictions.columns = gen_data.columns[1:]
    
    return det_gen, year_predictions

## for when I have the input csvs, but not the results
def analysis_and_plot(hybas, hybas2, countries, gen_data, cap_fac,
                      lag = False, r_thresh = 0.4,
                      title = '', plot_hydro = False,                       
                      pvals_all = False, snow = True, detrending = True, 
                      csv = False, num = 5): 
        
    ## read in processed geospatial data -- now aggregated and put into csv format
    ## don't need to merge with outcome data since that is done within the regs function
    country_data = {}
    regs_all = {}
    relevant_inputs = {}
    regs_top = pd.DataFrame(columns = ['Country Name', 'r2']) 
    
    for c in countries['Country Name']: 

        try: 
            new = pd.read_csv(f"Results_all/Inputs_{hybas}/{c}_input_data.csv").iloc[:,1:]
            try: 
                evi = pd.read_csv(f"Veg_indices/{c}_evi_data_country.csv")
                evi['year'] = pd.to_datetime(evi['time']).dt.year
                new = new.merge(evi[['EVI', 'year']], on = 'year')
            except: pass
        
            if snow == False:
                new = new.loc[:,~new.columns.str.contains('Snow', case=False)] 
             

            outcome = gen_data.loc[:, ['year', c]]
            outcome.columns = ['year', 'outcome']
    #        new = new.merge(outcome, on = 'year')
            country_data[c] = new
            
            if lag != False: 
                new2 = lagmat(new, maxlag = lag, use_pandas = True)
                new2 = new2.loc[:, ~new2.columns.str.contains('year')]
                new2['year'] = new['year']
                new = new2.copy()
                
            if pvals_all == False: 
                regs1, regs_best = regs(outcome, new, c, plot = False, threshold = r_thresh, 
                                        test_threshold = r_thresh, num = num)
            else: 
                regs1, regs_best = regs(outcome, new, c, plot = False, threshold = r_thresh, 
                                        test_threshold = r_thresh, pvals_all = pvals_all, num = num)

            regs_best['Country Name'] = c
           
            regs_all[c] = regs1
            regs_top = pd.concat([regs_top, regs_best])
            print(f"{c} found")
            
            if len(regs_top[c]) > 0:
                print(f"adding {c} to relevant inputs")
                inputs = regs_top[regs_top['country'] == c]['inputs'][0]
                inputs = inputs.replace("'", '')
                inputs = inputs.replace("[", '')
                inputs = inputs.replace("]", '')
                inputs = inputs.replace(" ", '')
                inputs2 = inputs.split(",")
                
                relevant_inputs[c] = new[list(inputs2[1:])]
            
        except: 
            try:
                new = pd.read_csv(f"Results_all/Inputs_{hybas}/{c}_input_data_{hybas}.csv").iloc[:,1:]
                try: 
                    evi = pd.read_csv(f"Veg_indices/{c}_evi_data_country.csv")
                    evi['year'] = pd.to_datetime(evi['time']).dt.year
                    new = new.merge(evi[['EVI', 'year']], on = 'year')
                except: pass
            
                if snow == False:
                    new = new.loc[:,~new.columns.str.contains('Snow', case=False)] 
                 
    
                outcome = gen_data.loc[:, ['year', c]]
                outcome.columns = ['year', 'outcome']
        #        new = new.merge(outcome, on = 'year')
                country_data[c] = new
                
                if lag != False: 
                    new2 = lagmat(new, maxlag = lag, use_pandas = True)
                    new2 = new2.loc[:, ~new2.columns.str.contains('year')]
                    new2['year'] = new['year']
                    new = new2.copy()
                        
                if pvals_all == False: 
                    regs1, regs_best = regs(outcome, new, c, plot = False, threshold = r_thresh, 
                                            test_threshold = r_thresh, num = num)
                else: 
                    regs1, regs_best = regs(outcome, new, c, plot = False, threshold = r_thresh, 
                                            test_threshold = r_thresh, pvals_all = pvals_all, num = num)
                    
                regs_best['Country Name'] = c
               
                regs_all[c] = regs1
                regs_top = pd.concat([regs_top, regs_best])
                print(f"{c} found")
                
                if len(regs_top) > 0:
                    print(f"adding {c} to relevant inputs")
                    inputs = regs_top[regs_top['country'] == c]['inputs'][0]
                    inputs = inputs.replace("'", '')
                    inputs = inputs.replace("[", '')
                    inputs = inputs.replace("]", '')
                    inputs = inputs.replace(" ", '')
                    inputs2 = inputs.split(",")
                    
                    relevant_inputs[c] = new[list(inputs2[1:])]
                
            except:
                try:
                    new = pd.read_csv(f"Results_all/Inputs_{hybas}/{c}_input_data_{hybas2}.csv").iloc[:,1:]
                    try: 
                        evi = pd.read_csv(f"Veg_indices/{c}_evi_data_country.csv")
                        evi['year'] = pd.to_datetime(evi['time']).dt.year
                        new = new.merge(evi[['EVI', 'year']], on = 'year')
                    except: pass
                
                    if snow == False:
                        new = new.loc[:,~new.columns.str.contains('Snow', case=False)] 
                     
        
                    outcome = gen_data.loc[:, ['year', c]]
                    outcome.columns = ['year', 'outcome']
            #        new = new.merge(outcome, on = 'year')
                    country_data[c] = new
                        
                    if lag != False: 
                        new2 = lagmat(new, maxlag = lag, use_pandas = True)
                        new2 = new2.loc[:, ~new2.columns.str.contains('year')]
                        new2['year'] = new['year']
                        new = new2.copy()
                            
                    if pvals_all == False: 
                        regs1, regs_best = regs(outcome, new, c, plot = False, threshold = r_thresh, 
                                                test_threshold = r_thresh, num = num)
                    else: 
                        regs1, regs_best = regs(outcome, new, c, plot = False, threshold = r_thresh, 
                                                test_threshold = r_thresh, pvals_all = pvals_all, num = num)
                        
                    regs_best['Country Name'] = c
                   
                    regs_all[c] = regs1
                    regs_top = pd.concat([regs_top, regs_best])
                    print(f"{c} found")
                    
                    if len(regs_top) > 0:
                        print(f"adding {c} to relevant inputs")
                        inputs = regs_top[regs_top['country'] == c]['inputs'][0]
                        inputs = inputs.replace("'", '')
                        inputs = inputs.replace("[", '')
                        inputs = inputs.replace("]", '')
                        inputs = inputs.replace(" ", '')
                        inputs2 = inputs.split(",")
                        
                        relevant_inputs[c] = new[list(inputs2[1:])]
                except:
                    print(f"{c} not found")
                    country_data[c] = 'failed'

    
    ## some real bad values... 
    for i in range(0,len(regs_top)): 
        if regs_top['r2'].iloc[i] < 0: 
            regs_top['r2'].iloc[i] = 0
            
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['pop_est','iso_a3', 'gdp_md_est', 'geometry']]
    world.columns = ['pop', 'Country Code', 'gdp', 'geometry']
    
    pct_hydro = pd.read_csv("Other_data/wb_hydro%.csv")[['Country Name', 'Country Code', '2015']]
    
    hydro_gdf = pd.merge(world, pct_hydro, on = 'Country Code')
    
                            ############ PLOT ############
                                
    ## now let's plot the performance of our regression analysis 
    merged_gdf = hydro_gdf.merge(regs_top, on = 'Country Name', how = 'outer')
    merged_gdf['r2'].fillna(0, inplace = True)
    
    
    ## failed is whatever country has r2 = 0 OR is in countries, but has no r2 
    failed = countries[['Country', 'Country Name']].merge(regs_top, how = 'outer')
    failed['r2'].fillna(0, inplace = True)
    failed = failed[failed['r2'] == 0].merge(hydro_gdf, on = 'Country Name')
    failed = gpd.GeoDataFrame(failed)
    
    if plot_hydro == True: 
            fig = merged_gdf.plot(column = 'r2', missing_kwds={'color':'lightgrey'}, 
                            figsize = (25, 20), cmap = 'YlGnBu', legend = True)
            fig.set_title('r2 for Generation Index, ' + hybas, fontsize=14)
            
            failed.plot(ax = fig, column = 'r2', figsize = (25,20), color = 'red', alpha = 0.7)
    
    regs_top['df'] = hybas2
    
    if csv == True: 
      #  regs_all.to_csv(f"regs_all_{hybas}.csv")
      #  relevant_inputs.to_csv(f"rel_inputs_{hybas}.csv")
        regs_top.to_csv(f"regs_top_{hybas}.csv")
    
#    if detrending == True:
#        return regs_all, regs_top, relevant_inputs, gen_data, year_pred
#    
#    else: 
    return regs_all, regs_top, relevant_inputs
    
## for when I have the results csvs, not the input data
def read_results(hybas, gen_data, countries, 
                 lag = False, title = '', plot_hydro = False): 
    # countries = id_hydro_countries(pct_gen = 25, year = 2015)

    # gen_data = countries.T
    # gen_data.reset_index(inplace = True)
    # ## drop the rows with the column names 
    # gen_data.columns = gen_data.iloc[0,:]
    # gen_data = gen_data.iloc[1:,:]
    # gen_data = gen_data.rename(columns = {'Country':'year'})
    
    ## if there is no data (or data is equal to 0), drop 
    ## nans will be dropped too 
    for c in gen_data.columns: 
        gen_data.loc[:,c] = pd.to_numeric(gen_data.loc[:,c], errors = 'coerce')
    
    gen_data = gen_data[(gen_data['year'] > 2000) & (gen_data['year'] < 2021)]
    gen_data['year'] = gen_data['year'].apply(np.int64)
    
    regs_all = {}
    regs_top = pd.DataFrame() 

    for c in countries['Country Name']: 
        try: 
            regs_all[c] = pd.read_csv(f"Results_all/Results_{hybas}/{c}_all_reg_results.csv")
            
            new_top = pd.read_csv(f"Results_all/Results_{hybas}/{c}_top_reg_results.csv")
            new_top['Country Name'] = c
            regs_top = pd.concat([regs_top, new_top])
            print(f"{c} found")
       
        except: 
            print(f"{c} not found")
            

    ## some real bad values... 
    for i in range(0,len(regs_top)): 
        if regs_top['r2'].iloc[i] < 0: 
            regs_top['r2'].iloc[i] = 0
            

    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['pop_est','iso_a3', 'gdp_md_est', 'geometry']]
    world.columns = ['pop', 'Country Code', 'gdp', 'geometry']
    
    pct_hydro = pd.read_csv("Other_data/wb_hydro%.csv")[['Country Name', 'Country Code', '2015']]
    
    hydro_gdf = pd.merge(world, pct_hydro, on = 'Country Code')
    
    
    ## now let's plot the performance of our regression analysis 
    merged_gdf = hydro_gdf.merge(regs_top, on = 'Country Name', how = 'outer')
    
    ## failed is whatever country has r2 = 0 OR is in countries, but has no r2 
    failed = countries[['Country', 'Country Name']].merge(regs_top, how = 'outer')
    failed['r2'].fillna(0, inplace = True)
    failed = failed[failed['r2'] == 0].merge(hydro_gdf, on = 'Country Name')
    failed = gpd.GeoDataFrame(failed)
    
                            ############ PLOT ############
    if plot_hydro == True: 

        fig = merged_gdf.plot(column = 'r2', missing_kwds={'color':'lightgrey'}, 
                        figsize = (25, 20), cmap = 'YlGnBu', legend = True)
        fig.set_title(f'r2 for Generation Index, {title}', fontsize=14)
        
        failed.plot(ax = fig, column = 'r2', figsize = (25,20), color = 'red')

        
    #     fig = merged_gdf.plot(column = 'r2', missing_kwds={'color':'lightgrey'}, 
    #                     figsize = (25, 20), cmap = 'YlGnBu', legend = True)
    #     fig.set_title(f'r2 for Generation Index, {title}', fontsize=14)
        
    #     failed.plot(ax = fig, column = 'r2', figsize = (25,20), color = 'red')

    return regs_all, regs_top

def group_countries(df, var, title, pval = 0.05, plot = True, save = True):
    grouped_df = pd.DataFrame() 
    
    for r in range(0, len(df)): 
        try: 
            if var in df.loc[r, 'inputs']: 
                new =  pd.DataFrame(df.loc[r, :]).T
                grouped_df = pd.concat([grouped_df, new])
        
        except: pass
    
        if df.loc[r, 'r2'] < 0: 
            df.loc[r, 'r2'] = 0
            
    if plot == True: 
        
        grouped_df['r2'] = pd.to_numeric(grouped_df['r2'], errors = 'coerce')
        
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['pop_est','iso_a3', 'gdp_md_est', 'geometry']]
        world.columns = ['pop', 'Country Code', 'gdp', 'geometry']
        
        pct_hydro = pd.read_csv("Other_data/wb_hydro%.csv")[['Country Name', 'Country Code', '2015']]
        
        hydro_gdf = pd.merge(world, pct_hydro, on = 'Country Code')
        
        
        ## now let's plot the performance of our regression analysis 
        merged_gdf = hydro_gdf.merge(grouped_df, right_on = 'country', left_on = 'Country Name', how = 'outer')
        
        fig = merged_gdf.plot(column = 'r2', missing_kwds={'color':'lightgrey'}, 
                        figsize = (25, 20), cmap = 'YlGnBu', \
                        vmin = 0, vmax = 1, legend = True)
        fig.set_title(f'r2 for Generation Index, {title}', fontsize=14)
        
        if save == True: 
            plt.savefig(f"Results_all/Figures/{title}.png")

    return grouped_df 

def run_groupings(df, hybas, plot = True):     
    snow_avg = group_countries(df, 'Snow', title = f'Snow, {hybas}')
    temp_avg = group_countries(df, 'LST', title = f'LST, {hybas}')
    evi_avg = group_countries(df, 'CMG', title = f'CMG, {hybas}')
    precip_avg =  group_countries(df, 'precip', title = f'Precip, {hybas}')
    
    summ_avg =  group_countries(df, 'summ', title = f'Summer Inputs, {hybas}') 
    spri_avg =  group_countries(df, 'spri', title = f'Spring Inputs, {hybas}') 
    wint_avg =  group_countries(df, 'wint', title = f'Winter Inputs, {hybas}') 
    fall_avg =  group_countries(df, 'fall', title = f'Fall Inputs, {hybas}') 

def clean_pvals(df1, sig_thresh = 0.05): 
    df = df1.copy()
    df['pval'] = df['pval'].astype(str)
    df['pval'] = df['pval'].str.replace('[', '')
    df['pval'] = df['pval'].str.replace(']', '')
    df['pval'] = df['pval'].str.split(", ")
    
    df.pval.fillna(0, inplace = True)
    new_df = pd.DataFrame()
    
    ## definitely not the most elegant way to do this... 
    ## do not shame me dear reader, pity me instead! 
    for r in range(0, len(df)): 
        try: 
            pvals = df.loc[r, 'pval'][1:]
            pvals = pd.to_numeric(pvals, errors = 'coerce')
    
            if len(pvals) > 0: 
                if all(i <= sig_thresh for i in pvals):
                    print("True") 
                    new = pd.DataFrame(df.loc[r,:]).T
                    new_df = pd.concat([new_df, new])
        except: pass
    
    return new_df   

def read_in_data(c, hybas, hybas2, snow = True):

    try: 
        new = pd.read_csv(f"Results_all/Inputs_{hybas}/{c}_input_data.csv").iloc[:,1:]
        try: 
            evi = pd.read_csv(f"Veg_indices/{c}_evi_data_country.csv")
            evi['year'] = pd.to_datetime(evi['time']).dt.year
            new = new.merge(evi[['EVI', 'year']], on = 'year')
        except: pass
    
        if snow == False:
            new = new.loc[:,~new.columns.str.contains('Snow', case=False)] 
                     
    except: 
        try:
            new = pd.read_csv(f"Results_all/Inputs_{hybas}/{c}_input_data_{hybas}.csv").iloc[:,1:]
            try: 
                evi = pd.read_csv(f"Veg_indices/{c}_evi_data_country.csv")
                evi['year'] = pd.to_datetime(evi['time']).dt.year
                new = new.merge(evi[['EVI', 'year']], on = 'year')
            except: pass
        
            if snow == False:
                new = new.loc[:,~new.columns.str.contains('Snow', case=False)] 
             
        except:
            try:
                new = pd.read_csv(f"Results_all/Inputs_{hybas}/{c}_input_data_{hybas2}.csv").iloc[:,1:]
                try: 
                    evi = pd.read_csv(f"Veg_indices/{c}_evi_data_country.csv")
                    evi['year'] = pd.to_datetime(evi['time']).dt.year
                    new = new.merge(evi[['EVI', 'year']], on = 'year')
                except: pass
            
                if snow == False:
                    new = new.loc[:,~new.columns.str.contains('Snow', case=False)] 
            except: new = 'failed'
    return new 

