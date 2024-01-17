# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:34:27 2023

@author: rcuppari
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from cleaning_funcs import id_hydro_countries
from cleaning_funcs import regs
import geopandas as gpd
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
import seaborn as sns
from itertools import combinations
import math 
from scipy.stats import spearmanr
import matplotlib as mpl
from cleaning_funcs import group_data

plt.rc('xtick', labelsize = 14)  # x tick labels fontsize
plt.rc('ytick', labelsize = 14)  # y tick labels fontsize

detrending = False #True ## don't need to do this with capacity factor :) 
loading_factor = 1.3
strike_val = 25
r_thresh = 0.4

## first things first, let's see what the correlations are between countries
countries = id_hydro_countries(pct_gen = 25, year = 2015, cap_fac = True)

gen_data = countries.T
gen_data.reset_index(inplace = True)
## drop the rows with the column names 
gen_data.columns = gen_data.iloc[0,:]
gen_data = gen_data.iloc[1:-4,:]
gen_data = gen_data.rename(columns = {'Country':'year'})

for c in gen_data.columns: 
    gen_data[c] = pd.to_numeric(gen_data[c], errors = 'coerce')

## let's detrend them to get a better idea
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

det_gen, det_predictions = detrend(gen_data)

## just decide reference point to which everything should be correlated, and plot
det_gen['sum'] = det_gen.loc[:,'Albania':].mean(axis = 1)

corrs = pd.DataFrame(det_gen.corr().loc['Albania':, 'sum'])
corrs.reset_index(inplace = True)
corrs.columns = ['Country Name', 'Correlation']

# mask = np.triu(np.ones_like(det_gen.loc[:, 'Albania':].corr()))
# sns.heatmap(det_gen.loc[:, 'Albania':].corr(), mask = mask)

# mask2 = np.triu(np.ones_like(gen_data.loc[:, 'Albania':].corr()))
# sns.heatmap(gen_data.loc[:, 'Albania':].corr(), mask = mask2)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['name','pop_est','iso_a3', 'gdp_md_est', 'geometry']]
world.columns = ['country', 'pop', 'Country Code', 'gdp', 'geometry']
## for some strange reason, France is listed as -99 instead of FRA... replace
world.iloc[43,2] = 'FRA'

pct_hydro = pd.read_csv("Other_data/wb_hydro%.csv")[['Country Name', 'Country Code', '2015']]

def make_numeric(df): 
    for c in range(0, len(df.columns)): 
        df.iloc[:,c] = pd.to_numeric(df.iloc[:,c], errors = 'coerce')        
    return df

cleaned_biggest3 = pd.read_csv("Results_all/results_biggest_basin3_cleaned.csv")
cleaned_avg = pd.read_csv("Results_all/results_cleaned.csv")
cleaned_subbasins6 = pd.read_csv("Results_all/results_country_subbasins6_cleaned.csv")
cleaned_subbasins5 = pd.read_csv("Results_all/results_country_subbasins_cleaned.csv")

# reg_sheds3_top.to_csv("reg_sheds3_top_det.csv")
# reg_sheds2_top.to_csv("reg_sheds2_top_det.csv")
# reg_sheds_top2.to_csv("reg_sheds_top2_det.csv")
# reg_top_4.to_csv("reg_top4_det.csv")
# reg_top_avg.to_csv("reg_top_avg_det.csv")

# gen_sheds3.to_csv("gen_sheds3_det.csv")
# gen_sheds2.to_csv("gen_sheds2_det.csv")
# gen_sheds42.to_csv("gen_sheds42_det.csv")
# gen4.to_csv("gen4_det.csv")
# gen4_lag.to_csv("gen4_lag_det.csv")
# gen_avg.to_csv("gen_avg_det.csv")

# year_sheds3.to_csv("year_sheds3_det.csv")
# year_sheds2.to_csv("year_sheds2_det.csv")
# year_sheds42.to_csv("year_sheds42_det.csv")
# year4.to_csv("year4_det.csv")
# year4_lag.to_csv("year4_lag_det.csv")
# year_avg.to_csv("year_avg_det.csv")
names1 = ['sheds3_top', 'sheds2_top', 'sheds_top2', 'top4', 'top_avg']
names2 = ['_sheds3', '_sheds2', '_sheds42', '4', '_avg']

regs_top = pd.DataFrame()
for name in names1: 
    if detrending == True: 
        new = pd.read_csv(f"Results_all/reg_{name}_det.csv").loc[:,'inputs':]
    else: 
        new = pd.read_csv(f"Results_all/reg_{name}.csv")
        
    new['df'] = name.replace('top','').replace('_', '')
    regs_top = pd.concat([regs_top, new])

regs_top.reset_index(inplace= True)

if detrending == True: 
    year_dict = {}
    gen_dict = {}    
    for name in names2: 
        year_dict[name] = pd.read_csv(f"Results_all/year{name}_det.csv")
        gen_dict[name] = pd.read_csv(f"Results_all/year{name}_det.csv")

                 ################# groupings #################
## identify which countries have certain variables that are significant  
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
    
#run_groupings(cleaned_subbasins6, 'subbasins 6')
#run_groupings(cleaned_subbasins5, 'subbasins 5')

                ################ significance ###############
## not every value that was saved in the regression is significant 
## frankly, most of them are not :( 
## so, pull out only the regressions that have significant p-values for all variables    
## dilemma: cannot blanket do this, because the pval for the constant can be whatever
## but, that's the first one, so can just skip it
def clean_pvals(df1, sig_thresh = 0.05): 
    df = df1.copy()
    df['pval'] = df['pval'].astype(str)
    df['pval'] = df['pval'].str.replace('[', '')
    df['pval'] = df['pval'].str.replace(']', '')
    df['pval'] = df['pval'].str.split(", ")
    
    df.pval.fillna(0, inplace = True)
    df.dropna(subset = ['inputs', 'pval'], inplace = True)
    new_df = pd.DataFrame()
    
    ## definitely not the most elegant way to do this... 
    ## do not shame me dear reader, pity me instead! 
    for r in range(0, len(df)): 
            pvals = df['pval'].iloc[r][1:]
            print(pvals)
            pvals = pd.to_numeric(pvals, errors = 'coerce')

            if len(pvals) > 0: 
                if all(i <= sig_thresh for i in pvals):
                    print("True") 
                    new = pd.DataFrame(df.iloc[r,:]).T
                    new_df = pd.concat([new_df, new])    

    return new_df   

df_avg = clean_pvals(cleaned_avg)
df_big = clean_pvals(cleaned_biggest3)
df5 = clean_pvals(cleaned_subbasins5)
df6 = clean_pvals(cleaned_subbasins6)
reg_top= clean_pvals(regs_top)

reg_top = reg_top.rename(columns = {'Country Name':'country'})

df_avg['df'] = 'df_avg'
df_big['df'] = 'df_big'
df5['df'] = 'df5'
df6['df'] = 'df6'

all_sig = pd.concat([reg_top, df_avg, df_big, df5, df6])

#all_sig = all_sig.iloc[:, :-1]
all_sig.r2 = pd.to_numeric(all_sig.r2, errors = 'coerce')
all_sig.test_r2 = pd.to_numeric(all_sig.test_r2, errors = 'coerce')

## need to drop duplicates based on r2
all_sig = all_sig.sort_values('r2').drop_duplicates(subset = 'country', keep = 'last')
all_sig = all_sig.rename(columns = {'country':'Country Name'})
#all_sig.to_csv("Results_all/all_sig_det.csv")

all_sig = all_sig[all_sig['r2'] >= 0.3]
all_sig.reset_index(inplace = True)

hydro_gdf = pd.merge(world, pct_hydro, on = 'Country Code')
merged_gdf = hydro_gdf.merge(all_sig, on = 'Country Name', how = 'outer')
merged_gdf = merged_gdf.merge(corrs, on = 'Country Name', how = 'outer')

cmap = mpl.cm.YlGnBu(np.linspace(0,1,100))
cmap = mpl.colors.ListedColormap(cmap[30:,:-1])

plt.rc('xtick', labelsize = 12)  # x tick labels fontsize
plt.rc('ytick', labelsize = 12)  # y tick labels fontsize
plt.rc('axes', labelsize = 14)  # x tick labels fontsize

fig, ax = plt.subplots()
fig2 = merged_gdf.plot(column = 'r2', ax = ax, missing_kwds={'color':'silver'}, 
                cmap = cmap, legend = False)
#fig.set_title('Significant Regression Results', fontsize=14)
cbar = plt.colorbar(fig2.get_children()[0], ax=ax, orientation = 'horizontal', 
                    aspect = 30)
cbar.set_label('r2', fontsize=14)
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
plt.tight_layout(h_pad=1)

############################ making a portfolio ##############################
## DOUBLE CHECK CONVERSIONS!
## JUST USE LCOE DATA?
## either https://www.iea.org/reports/projected-costs-of-generating-electricity-2020
## or IRENA 2021 report https://sciwheel.com/work/item/12749865/resources/14115980/pdf
gen_vals = id_hydro_countries(pct_gen = .01, year = 2015)
gen_vals = gen_vals.T
gen_vals.reset_index(inplace = True)
## drop the rows with the column names 
gen_vals.columns = gen_vals.iloc[0,:]
gen_vals = gen_vals.iloc[1:-4,:]
gen_vals = gen_vals.rename(columns = {'Country':'year'})

for country in gen_vals.columns[1:]:
    print(country)
    
    try: 
        price = (.044)*pow(10,9) ## gen is in billion KWh
        gen_vals[country] = pd.to_numeric(gen_vals[country], errors = 'coerce')
        gen_vals[country] = gen_vals[country] * price#.iloc[0]
    
    except: 
        gen_vals[country] = np.nan
        print(f"could not match {country}")

avg_val = pd.DataFrame(gen_vals.mean()).iloc[1:,:]
avg_val.columns = ['avg_val']
avg_val.reset_index(inplace = True)
avg_val.columns = ['Country Name', 'avg_val']

merged_vals = hydro_gdf.merge(avg_val, on = 'Country Name', how = 'inner')

for r in range(0, len(merged_vals)): 
    if merged_vals['avg_val'].iloc[r] == 0: 
        merged_vals['avg_val'].iloc[r] = 'na'

merged_vals['avg_val'] = pd.to_numeric(merged_vals['avg_val'], errors = 'coerce')
        
## maybe also sort so that only LICs/LMICs/MICs are included in the pool? 
## good news: regressions really only worked for a few, smaller ones
merged_gdf2 = merged_gdf.dropna(axis = 0, subset = ['inputs', 'Correlation'])
merged_vals2 = merged_vals[merged_vals['Country Name'].isin( merged_gdf2['Country Name'])]

#merged_vals2.to_file('merged_vals2.shp')  

# fig, (ax1, ax2)  = plt.subplots(2,1, sharex = True, sharey = True) 
# hydro_gdf.plot(ax = ax1, color = 'darkgrey')
# merged_gdf2.plot(ax = ax1, column = 'Correlation', missing_kwds={'color':'lightgrey'}, 
#                 figsize = (35, 25), cmap = 'Reds', legend = True)
# ax1.set_title('Generation Correlations (relative to global sum)', fontsize=16)

# hydro_gdf.plot(ax = ax2, color = 'darkgrey')
# merged_vals2.plot(ax = ax2, column = 'avg_val', missing_kwds={'color':'lightgrey'}, 
#                 figsize = (35, 25), cmap = 'Greens', legend = True)
# ax2.set_title("Generation Value ($ '000)", fontsize=16)


###############################################################################
###############################################################################
###############################################################################
## so, need to pull in the regressions and the results for the relevant
## countries. Need to find a way to automate this process if I am pulling
## from various different docs/averages/etc. 
## want to make a dictionary with predictions for each country 
all_sig = all_sig.dropna(subset = ['Country Name'], axis = 0).iloc[:,2:]
all_sig.reset_index(inplace = True)


## CHANGED THIS!
def predict_countries(country, hybas, reg_inputs, detrend = detrending): 
    #if 'watershed' in hybas2:
    #    hybas = 'watersheds'
    if 'shed' in hybas: 
        hybas2 = 'watersheds'

    # if hybas == 'avg': 
    #     #hybas2 = 'country_avg2'
    #     hybas2 = 'country_avg'
        

    try: 
        inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data.csv").iloc[:,1:]
    except: 
        try: 
            inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data_{hybas}.csv").iloc[:,1:]
        except: 
            try: 
                inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data_country.csv").iloc[:,1:]
            except: 
                try: 
                    inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data_{hybas2}.csv").iloc[:,1:]
                except: 
                    inputs = 'failed'
                    print(f"failed to find {country}, {hybas}")        
                    
                    
    evi = pd.read_csv(f"Veg_Indices/{country}_evi_data_country.csv")
    evi = group_data(evi)
    
    inputs = pd.merge(inputs, evi, on = 'year')

    try:   
#        input_list = reg_inputs['inputs'].iloc[0]
        try: 
            reg_inputs2 = inputs[[i for i in reg_inputs[1:]]] ## skip constant
        except: 
            #reps = {'CMG0.': 'EVI'}
            #reg_inputs2 = [reps.get(x,x) for x in reg_inputs]
            reg_inputs = [reg_inputs.replace('CMG0.', 'EVI') for reg_inputs in reg_inputs]
            reg_inputs.str.replace("CMG0.", "EVI", regex=True)
            reg_inputs2 = inputs[[i for i in reg_inputs[1:]]] ## skip constant

        reg_inputs2 = sm.add_constant(reg_inputs2)
        reg_inputs2['year'] = inputs['year']

        if detrend == True:            
            y = det_gen[['year', country]]
        else: 
            y = gen_data[['year', country]]
            
        y = y[y['year'].isin(reg_inputs2.year)]
        y.reset_index(inplace = True)
        
        reg_inputs2 = reg_inputs2[reg_inputs2['year']<= y.year.max()]
        X = reg_inputs2[['const', 'summ_precip_2040548500']]#reg_inputs2.iloc[:, :-1]
        
        est = sm.OLS(y[country], X)
        est2 = est.fit()
        
        predicted = pd.DataFrame(est2.predict(X))#.iloc[:, :-1]))
        predicted.columns = [country]
        
        if detrend == False: 
            predicted[predicted < 0] = 0
    #    predictions = pd.concat([predictions, predicted], axis = 1)  
        #predictions.index = inputs.year
        
        #gen_data2 = make_numeric(gen_data)
        #gen_data2 = gen_data2[(gen_data2['year'] <= inputs.year.max()) & (gen_data2['year'] >= inputs.year.min())]
        #gen_data2.set_index('year', inplace = True)
    #    error_pct = ((predictions - gen_data2)/gen_data2).dropna(axis =1)
    #    error = (predictions - gen_data2).dropna(axis =1)
        error_pct = ((predicted.iloc[:,0] - y[country])/y[country]).mean()*100
        error = (predicted.iloc[:,0] - inputs.iloc[:,-1])
    
      #  print(f"{country} error pct: {error_pct}")

    except: 
        predicted = pd.DataFrame(np.repeat(-9,20))
        predicted.columns = [country]

    return predicted

predictions = pd.DataFrame()
for r, country in enumerate(all_sig['Country Name']):    
    print(country)
    country_gen = gen_data[['year', country]]
    country_gen['year'] = pd.to_numeric(country_gen['year'], errors = 'coerce')
    country_gen[country] = pd.to_numeric(country_gen[country], errors = 'coerce')
    
    hybas = all_sig.loc[r,'df']   
    
    reg_inputs = all_sig['inputs'].iloc[r]
    df = all_sig['df'].iloc[r]

    if type(reg_inputs) is list: 
        ''
    else:
        reg_inputs = reg_inputs.replace(" ", "")
        reg_inputs = reg_inputs.replace("[", "")
        reg_inputs = reg_inputs.replace("]", "")
        reg_inputs = reg_inputs.replace("'", "")
    
        reg_inputs = reg_inputs.split(",")
    
    fail = False
    if detrending == True:
        
        if hybas == 'top4': 
            key = '4'
        else: key = hybas
            
        # else: 
        #     try: 
        #         hybas = f'{hybas.partition("_")[0]}'
        #         predict_det = predict_countries(country, hybas, reg_inputs)
        #         year_pred = year_dict['_' + str(hybas)][country]
        #     except: 
        #         try: 
        #             hybas = f'{hybas.partition("_")[2]}'
        predict_det = predict_countries(country, hybas, reg_inputs)
        
        ## actually doesn't matter which key you use because they are all the same
        ## (detrending based on year, which is not relevant to watershed)
        if hybas == '4': 
            year_pred = year_dict['' + str(hybas)][country]
        else: 
            year_pred = year_dict['_' + str(hybas)][country]
        #         except: 
        #             print(f'do not have inputs for {hybas}, {country}')
        #             fail = True
        # # if fail != True: 
        year_pred.loc[-1] = 0  # adding a row
        year_pred.index = year_pred.index + 1  # shifting index
        year_pred = year_pred.sort_index()  # sorting by index
        if (predict_det.mean() < 0)[0]: 
            predict = predict_det
        else: 
            predict = year_pred + predict_det.iloc[:,0]
        # plt.scatter(predict, country_gen[country_gen['year'] >= 2000][country])
        
    else:
        ## probably need to do the same fixing here
        predict = predict_countries(country, hybas, reg_inputs, detrend = False)
        
    if fail != True:
        predictions = pd.concat([predictions, predict],
                                axis = 1)

#predictions.to_csv("predictions_det.csv")
gen_data['year'] = pd.to_numeric(gen_data['year'], errors = 'coerce')
gen_data2 = gen_data[gen_data['year'] >= 2000]

predictions.dropna(axis = 1, inplace = True)
r2 = pd.DataFrame(index = gen_data2.index, columns = predictions.columns)
rho = pd.DataFrame(index = gen_data2.index, columns = predictions.columns)

for c in predictions.columns: 
    r2.loc[:,c] = np.corrcoef(predictions[c], gen_data2[c])[1,0]**2
    rho.loc[:,c] = spearmanr(predictions[c], gen_data2[c])[0]
    

def make_scatters(predictions, name = ''): 
    ncols = 3
    nrows = math.ceil(len(predictions.columns)/ncols)
    
    fig = plt.figure()
    gs = plt.GridSpec(nrows, ncols, wspace = 0.2, hspace = .4) 
    axs = []
    for c, num in zip(predictions.columns, range(1, len(predictions.columns)+1)):
        print(c)
        print(num)
        axs.append(fig.add_subplot(gs[num-1]))
        im = axs[-1].scatter(predictions[c], gen_data2[c], c = r2[c], vmin = .3, vmax = 1)
        axs[-1].plot([gen_data2[c].min(), gen_data2[c].max()],
                    [gen_data2[c].min(), gen_data2[c].max()], 
                    color = 'black', linestyle = '--')
        axs[-1].set_xlabel("Predicted", fontsize = 14)
        axs[-1].set_ylabel("Observed", fontsize = 12)
        axs[-1].set_title(f"{c}", fontsize = 14)
        #plt.tight_layout()
    
    axs.append(fig.add_subplot(gs[nrows-1, 2:len(predictions.columns)]))
    fig.colorbar(im, cax = axs[-1], orientation = 'horizontal')
    axs[-1].set_xlabel("r2",fontsize = 14)
    fig.savefig(fname = f"Figures/indices{name}.jpg")
#     ncols = 3
#     nrows = math.ceil(len(predictions.columns)/ncols)
    
#     fig = plt.figure()
#     gs = plt.GridSpec(nrows, ncols, wspace = 0.2, hspace = .4) 
#     axs = []
#     for c, num in zip(predictions.columns, range(1, len(predictions.columns)+1)):
#         print(c)
#         print(num)
#         axs.append(fig.add_subplot(gs[num-1]))
#         im = axs[-1].scatter(predictions[c], gen_data2[c], c = r2[c], vmin = .3, vmax = 1)
#         axs[-1].plot([gen_data2[c].min(), gen_data2[c].max()],
#                     [gen_data2[c].min(), gen_data2[c].max()], 
#                     color = 'black', linestyle = '--')
#         axs[-1].set_xlabel("Predicted", fontsize = 14)
#         axs[-1].set_ylabel("Observed", fontsize = 12)
#         axs[-1].set_title(f"{c}", fontsize = 14)
#         #plt.tight_layout()
    
#     axs.append(fig.add_subplot(gs[nrows-1, 2:len(predictions.columns)]))
#     fig.colorbar(im, cax = axs[-1], orientation = 'horizontal')
#     axs[-1].set_xlabel("r2",fontsize = 14)
#     fig.savefig(fname = f"Figures/indices{name}.jpg")

# ## hard to fit more than 8 on one page (to work on later)
make_scatters(predictions.iloc[:,:8], name = 'pred1')
make_scatters(predictions.iloc[:,8:16], name = 'pred2')
make_scatters(predictions.iloc[:,16:], name = 'pred3')


# ## take the top 8 
# top_eight = all_sig.sort_values(by = 'r2', ascending = False)['Country Name'].iloc[:8]
# make_scatters(predictions[top_eight.values], name = 'tops')

## should set the strike as some percentile of flow? Maybe 20%? 
## drop the -99
predictions = predictions[[c for c
        in list(predictions)
        if len(predictions[c].unique()) > 1]]

strikes = pd.DataFrame(index = predictions.columns, columns = ['strike'])
payouts = pd.DataFrame(index = np.arange(0, len(predictions)), 
                       columns = predictions.columns)

#predictions['year'] = np.arange(2000, 2000 + len(predictions))
## as it stands, the -99 always have a payout, ignore for now 
for c in predictions.columns: 
    ## want to base our strike on the full historical record?? 
    ## TO THINK ABOUT: base strike on full record? 
    ## Should if use detrended one
    ## so use the entire gen_data set 
    #gen_c = gen_data[c].dropna()
    strikes.loc[c, 'strike'] = np.percentile(gen_data2[c], strike_val)
    print(f"{c} strike: {strikes.loc[c, 'strike']}")
    price = avg_val[avg_val['Country Name'] == c]['avg_val']
    for year in np.arange(0, len(predictions)):
        #print(year)
        if predictions[c].iloc[year] < strikes.loc[c, 'strike']:
            ## payout predicted losses payout
            payouts.loc[year,c] = abs(predictions[c].iloc[year] * price.values[0])
        else: 
            payouts.loc[year, c] = 0
    print(f"{c} times below strike: \
          {len(predictions[predictions[c] < strikes.loc[c, 'strike']])}")
    
    payouts[c] = pd.to_numeric(payouts[c])

## plot coefficient of variation for predictions (not including $0 payouts)
preds2 = predictions[predictions.sum().sort_values(ascending = True).index]#.loc[:,:'Suriname']#:]
for i, c in enumerate(preds2):
    preds2[c] = preds2.loc[:,:c].sum(axis = 1)#/payouts2.loc[:,:c].mean().mean()
    if i > 0:
        new = str(preds2.columns[i-1]) + ' + ' + str(preds2.columns[i])    
        preds2 = preds2.rename(columns = {preds2.columns[i]:new})    


########################### calculate premiums ################################

## to start, need to identify expected value losses 
## placeholder loading, say 25% ish, for individual countries
from wang_transform import wang_slim
from wang_transform import wang

loading = pd.DataFrame(index = predictions.columns, 
                       columns = ['mean_payout', 'loading', 'wang_prem'])
for c in predictions.columns: 
    payouts3 = pd.DataFrame(payouts[c])
    payouts3.columns = ['payout']
    loading.loc[c, 'mean_payout'] = payouts[c].mean()
    loading.loc[c, 'loading'] = payouts[c].mean() * loading_factor   
    loading.loc[c, 'wang_prem'] = wang_slim(payouts3, contract = 'put', 
                              premOnly = True, from_user = True)
    
loading['wang_pct'] = loading['wang_prem']/loading['mean_payout']
#loading.fillna(0,inplace = True)

## but, in a portfolio, can divvy up differently -- avg %
## and weigh by portion of generation(?) 
sum_prem = loading['loading'].mean()
sum_prem_wang = loading['wang_prem'].mean()

loading_pool = payouts.sum(axis = 1).mean() * loading_factor
pooled_payouts = pd.DataFrame(payouts.sum(axis = 1))
pooled_payouts.columns = ['payout']
loading_pool_wang = wang_slim(pooled_payouts, 
                              contract = 'call', premOnly = True, 
                              from_user = False)

## what if use the Braun 
## rolx = lane financial rate on line index -- 0.95 from 1997-2012??? 
## reins = diversification level of reinsurer (assume global + great) 
## grade = investment grade or not (assume no, esp because of LICs/LMICs)
## peak = is this a peak territory, i.e., the US? Assume no 
## bbspr = bond spread with Treasury curve 

def braun_pricing(payouts, bbspr = 3.5, rolx = 0.95, reins = 1, grade = 0, 
                  peak = 0):
    exp_loss = payouts.mean() 
    braun_prem = 221.04*exp_loss + 175.08*peak + 103.58*reins + \
        161.85*rolx - 159.76*grade + 26.57*bbspr
    return braun_prem
    
loading['braun_prem'] = 0
avg_pool = pooled_payouts.mean()
for c in loading.index:
    pct = (loading.loc[c, 'mean_payout']/avg_pool)*100
    loading.loc[c, 'braun_prem'] = \
        braun_pricing(pct)
loading['braun_pct'] = loading['braun_prem']/loading['mean_payout']

########################## changing payout distro #############################
######################### nice lil ridgeline plot #############################
### courtesy of https://python-graph-gallery.com/ridgeline-graph-seaborn/ :) ##
## might be most interesting to show payouts as percentage of mean 
## since that's the whole point -- much less variation in a pool 

payouts2 = payouts.copy()    
payouts2 = payouts2[payouts2.sum().sort_values(ascending = True).index]#.loc[:,:'Suriname']#:]
 
# can also make with the cumulative effect...
for i, c in enumerate(payouts2):
    ## summed
#    payouts2[c] = payouts2.loc[:,:c].sum(axis = 1)
    
    ## standardized    
    payouts2[c] = (payouts2.loc[:,:c].sum(axis = 1) - \
                   payouts2.loc[:,:c].sum(axis = 1).mean()) / payouts2.loc[:,:c].sum(axis = 1).std()

    if i > 0:
        print(c)
        new = str(payouts2.columns[i-1]) + ' + ' + str(payouts2.columns[i])    
        payouts2 = payouts2.rename(columns = {payouts2.columns[i]:new})    

payouts2 = payouts2.rename(columns = {payouts2.columns[-1]:"Pool"})    
payouts_melt = payouts2.melt()
#payouts_melt['value'] = payouts_melt['value'].astype(float)
payouts_melt.fillna(0, inplace = True)
#payouts_melt.sort_values('variable', inplace = True)

######################### calculate Shapley value #############################
## basically want to identify the marginal contribution of each policy 
## to the overall payouts/losses to then weigh the premiums assigned
## to each member of the pool 
## find all combinations of countries and, really, associated payouts

from shapley import Shapley_calc
from shapley import powerset

################ reserves needed for each and within a pool? ##################
################################# PLOTS #######################################
expected_payouts_pool = pd.DataFrame(index = np.arange(0, len(payouts2.columns)), 
                                      columns = ['sum', 'std', 'VaR', 
                                                 'sum_individ_var'])

## change order of payouts so it goes smallest to greatest 
payouts_sorted = payouts.sum().sort_values()
payouts = payouts.loc[:, payouts_sorted.index]
var_test = 99

new = [payouts.columns[0]]
for i, c in enumerate(payouts):
#    print(f"col values: {payouts.loc[:,c].head()}")
    print(f"col: {i}")
    
    expected_payouts_pool.loc[i, 'sum'] = payouts.loc[:,:c].sum(axis = 1).mean()
    expected_payouts_pool.loc[i, 'std'] = payouts.loc[:,:c].sum(axis = 1).std()
    expected_payouts_pool.loc[i, 'VaR'] = np.percentile(payouts.loc[:,:c].sum(axis = 1), 95)
    
    ## also find sum of individ vars     
    var= 0
    for col in payouts.loc[:,:c]: 
        #print(f"{col}")

        individ_var = np.percentile(payouts.loc[:,col], var_test)
        var = var + individ_var
#        print(f"Summed VaR: {var}, Individ: {individ_var}")
        
    expected_payouts_pool.loc[i, 'sum_individ_var'] = var
    
    if i > 0:
        new2 = [str(expected_payouts_pool.index[i-1]) + ' + ' + str(payouts.columns[i])]
        new.append(new2)
        
expected_payouts_pool['name'] = new
for c in expected_payouts_pool.columns[:-1]:
    expected_payouts_pool.loc[:,c] = pd.to_numeric(expected_payouts_pool.loc[:,c])


###############################################################################
## figure showing changes in distro payouts (normalized)
###############################################################################
per_cap_payouts1 = payouts['Ghana']
per_cap_payouts5 = payouts[['Uruguay', 'Cameroon', 'Romania', 'Namibia', 'Peru']].mean(axis = 1)/5
per_cap_payouts10 = payouts[['Uruguay', 'Cameroon', 'Romania', 'Namibia', 'Peru',
                             'Ghana', 'Croatia', 'Austria', 'Slovenia', 'Angola']].mean(axis = 1)/10

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex = True)
sns.kdeplot(per_cap_payouts1, ax = ax1, fill = True, color = '#21918c', alpha = 1)
ax1.set_xlabel("Per Capita Payout - 1 Country ($B)", fontsize = 14)
ax1.set_ylabel("Density", fontsize = 14)
ax1.set_xticks([-0.5*pow(10,9), 0, 0.5*pow(10,9), 1*pow(10,9), 1.5*pow(10,9), 2*pow(10,9)],
               [-0.5, 0, 0.5, 1, 1.5, 2])
ax1.set_yticks([])

sns.kdeplot(per_cap_payouts5, ax = ax2, fill = True, color = '#3b528b', alpha = 1)
ax2.set_xlabel("Per Capita Payout - 5 Countries ($B)", fontsize = 14)
ax2.set_ylabel("Density", fontsize = 14)
ax2.set_xticks([-0.5*pow(10,9), 0, 0.5*pow(10,9), 1*pow(10,9), 1.5*pow(10,9), 2*pow(10,9)],
               [-0.5, 0, 0.5, 1, 1.5, 2])
ax2.set_yticks([])

sns.kdeplot(per_cap_payouts10, ax = ax3, fill = True, color = '#440154', alpha = 1)
ax3.set_xlabel("Per Capita Payout - 10 Countries ($B)", fontsize = 14)
ax3.set_ylabel("Density", fontsize = 14)
ax3.set_xticks([-0.5*pow(10,9), 0, 0.5*pow(10,9), 1*pow(10,9), 1.5*pow(10,9), 2*pow(10,9)],
               [-0.5, 0, 0.5, 1, 1.5, 2])
ax3.set_yticks([])

###############################################################################
## now let's go to premiums instead of reserves ##
###############################################################################

expected_losses = pd.DataFrame(index = payouts.columns, columns = ['expected_payout'])
for country in expected_losses.index: 
    exp_loss = payouts[country].mean()
    expected_losses.loc[country, 'expected_payout'] = exp_loss

expected_losses['expected_payout'] = pd.to_numeric(expected_losses['expected_payout'], errors = 'coerce')

combs = []
for subset in powerset(payouts.columns): 
    combs.append(list(subset))
    
## now find the expected payouts for each combination
group_loss = []
for i in combs:   
    group_loss.append(payouts[i].sum(axis = 1).mean())

expected_losses['Shapley'] = 0
for country in payouts.columns: 
    ## shapley value = 
        ## 1/n * sum (n-1! / S!)^-1 * marg contribution S
        ## CHECK WHY NOT THE SAME LENGTH
    marg_contrib, summed = Shapley_calc(country, combs, payouts = payouts, 
                                        pricing = 'wang', code = 'wang_slim')
    expected_losses.loc[country, 'Shapley'] = summed
    
expected_pool_loss = payouts.sum(axis = 1).mean()

## pooled payouts + premiums make sense :) except for Suriname, but let's ignore that for now
pooled_payouts = pd.DataFrame(payouts.sum(axis = 1))
pooled_payouts.columns = ['payout']
#pooled_payouts.columns = ['asset']
#pooled_payouts2 = pooled_payouts.copy()

pool_stats = pd.DataFrame(index = ['stat'])
pool_stats['prem'], net = wang_slim(pooled_payouts, contract = 'put', lam = 0.25,
              from_user = True, premOnly = False)

#pooled_payouts.mean()/pow(10,11)
pool_stats['load_prem'] = (pooled_payouts.payout.mean() * loading_factor)
pool_stats['shap_prem'] = expected_losses.Shapley.sum()
pool_stats['net'] = pooled_payouts.payout.mean() - pool_stats.load_prem
pool_stats['net_shap'] = pooled_payouts.payout.mean() - pool_stats.shap_prem
pool_stats['exp_payout'] =  pooled_payouts.payout.mean()

## now calculate what they would be individually 
for country in payouts.columns:
    payout_individ = pd.DataFrame(payouts[country])
    payout_individ.columns = ['payout']
    expected_losses.loc[country, 'individ_prem'] = \
        (wang_slim(payout_individ, contract = 'put', lam = 0.25,
                  from_user = True, premOnly = True))
        
print(f"Diff between individ prem sum and pool premium: \
      ${(-expected_losses['individ_prem'].sum() + pool_stats.prem.mean())/pow(10,9):,}B")

print(f"Diff between individ prem sum and pool premium: \
      {(expected_losses['individ_prem'].sum() - pool_stats.prem.mean())/expected_losses['individ_prem'].sum()*100:,}%")
      
expected_losses['loading'] = 100*(expected_losses['individ_prem']/payouts.sum())


###############################################################################
#################### max payout as percentage of premium ######################
loading['pct_max'] = 0
loading['contrib_port'] = 0
port_prem = loading.loading.sum()
net_payouts = payouts.copy()
for c in payouts.columns:
    loading.loc[c, 'pct_max'] = payouts[c].mean() / port_prem
    loading.loc[c, 'contrib_port'] = loading.loc[c, 'pct_max'] * port_prem
#    net_payouts[c] = payouts[c] - loading.loc[c, 'pct_max'] * port_prem
    
## how many reserves would the insurer have to hold onto? 
var_port = np.percentile(payouts.sum(axis = 1),95)

lab_size = 16
tick_size = 14


###############################################################################
#################### FIG 1 HYDROPOWER REV AS PCT GDP ##########################
gdp = pd.read_csv("Other_data/wb_gdp.csv", skiprows = 3)[['Country Code', '2015']]
gdp.columns = ['Country Code', '2015_gdp']
merged_vals = merged_vals.merge(gdp, on = 'Country Code', how = 'outer')
merged_vals['hydro_pct_gdp'] = (merged_vals['avg_val']/merged_vals['2015_gdp'])*100
merged_vals = merged_vals.merge(hydro_gdf[['Country Code', 'geometry']], how = 'outer')


cmap = mpl.cm.YlGnBu(np.linspace(0,1,100))
cmap = mpl.colors.ListedColormap(cmap[20:,:-1])


fig, (ax1, ax2) = plt.subplots(ncols=1, nrows = 2, sharex=True, sharey=True)
hydro_gdf.plot(column='2015', ax = ax1,  missing_kwds={'color': 'lightgrey'},
            figsize = (25, 20), cmap = cmap, legend = True)
ax1.set_ylabel("Longitude", fontsize = 14)
ax1.set_yticks([90, 60, 30, 0, -30, -60],fontsize = 14)

merged_vals.plot(column = 'hydro_pct_gdp', ax = ax2, missing_kwds={'color':'lightgrey'}, 
                cmap = cmap, legend = True)
ax2.set_xlabel("Latitude", fontsize = 14)
ax2.set_ylabel("Longitude", fontsize = 14)

###############################################################################
###################### FIG 2 BPA INDEX PERFORMANCE ############################

###############################################################################
##################### FIG 3 HYDROPOWER CORRELATIONS ###########################

## panel 1: correlations between regions (as defined by WBG)
## South Asia, Sub-Saharan Africa, Europe & Central Asia, 
## East Asia & Pacific, Latin America & Caribbean, 
## Middle East & North Africa, North America
regions2 = pd.read_csv("Other_Data/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5607117.csv")
country_conversion = pd.read_csv("Other_data/country_conversions.csv")
regions2 = regions2.merge(country_conversion[['iso_a3', 'EIA Name']], \
                        left_on = 'Country Code', right_on = 'iso_a3')
regions = regions2[regions2['EIA Name'].isin(gen_data.columns)]

sa = det_gen[regions[regions['Region'] == 'South Asia']['EIA Name']]
sa['sum'] = sa.sum(axis = 1)
ssa = det_gen[regions[regions['Region'] == 'Sub-Saharan Africa']['EIA Name']]
ssa['sum'] = ssa.sum(axis = 1)
eca = det_gen[regions[regions['Region'] == 'Europe & Central Asia']['EIA Name']]
eca['sum'] = eca.sum(axis = 1)
eap = det_gen[regions[regions['Region'] == 'East Asia & Pacific']['EIA Name']]
eap['sum'] = eap.sum(axis = 1)
nam = det_gen[regions[regions['Region'] == 'North America']['EIA Name']]
nam['sum'] = nam.sum(axis = 1)
latam = det_gen[regions[regions['Region'] == 'Latin America & Caribbean']['EIA Name']]
latam['sum'] = latam.sum(axis = 1)
mena = det_gen[regions[regions['Region'] == 'Middle East & North Africa']['EIA Name']]


merged_gdf = hydro_gdf.merge(corrs, on = 'Country Name', how = 'outer')

## non in MENA so omit. For others, sum generation. I think that's best, 
## since the idea is cumulative losses/relationships, but maybe average?
reg_sums = pd.DataFrame({'Year': gen_data.year, 
                         'SA': sa['sum'], 
                         'SSA': ssa['sum'], 
                         'ECA': eca['sum'], 
                         'EAP': eap['sum'], 
                         'NAM': nam['sum'], 
                         'LATAM': latam['sum']})
reg_corrs = reg_sums.corr()    
## this would be useful to show the benefit of a global versus regional 
## pool 

## other panels: 
latam_corrs = pd.DataFrame(latam.corr()['sum'].iloc[:-1])
latam_corrs.reset_index(inplace = True)
latam_corrs.columns = ['Country Name', 'corrs']
latam_corrs = gpd.GeoDataFrame(latam_corrs.merge(merged_gdf[['Country Name', 'geometry']], \
                                on = 'Country Name'))
hydro_gdf = pd.merge(world, pct_hydro, on = 'Country Code')

nam_corrs = pd.DataFrame(nam.corr()['sum'].iloc[:-1])
nam_corrs.reset_index(inplace = True)
nam_corrs.columns = ['Country Name', 'corrs']
nam_corrs = gpd.GeoDataFrame(nam_corrs.merge(merged_gdf[['Country Name', 'geometry']], \
                                on = 'Country Name'))

eap_corrs = pd.DataFrame(eap.corr()['sum'].iloc[:-1])
eap_corrs.reset_index(inplace = True)
eap_corrs.columns = ['Country Name', 'corrs']
eap_corrs = gpd.GeoDataFrame(eap_corrs.merge(merged_gdf[['Country Name', 'geometry']], \
                                on = 'Country Name'))

eca_corrs = pd.DataFrame(eca.corr()['sum'].iloc[:-1])
eca_corrs.reset_index(inplace = True)
eca_corrs.columns = ['Country Name', 'corrs']
eca_corrs = gpd.GeoDataFrame(eca_corrs.merge(merged_gdf[['Country Name', 'geometry']], \
                                on = 'Country Name'))
    
ssa_corrs = pd.DataFrame(ssa.corr()['sum'].iloc[:-1])
ssa_corrs.reset_index(inplace = True)
ssa_corrs.columns = ['Country Name', 'corrs']
ssa_corrs = gpd.GeoDataFrame(ssa_corrs.merge(merged_gdf[['Country Name', 'geometry']], \
                                on = 'Country Name'))

sa_corrs = pd.DataFrame(sa.corr()['sum'].iloc[:-1])
sa_corrs.reset_index(inplace = True)
sa_corrs.columns = ['Country Name', 'corrs']
sa_corrs = gpd.GeoDataFrame(sa_corrs.merge(merged_gdf[['Country Name', 'geometry']], \
                                on = 'Country Name'))
   

sa_countries = regions2[regions2['Region'] == 'South Asia']
sa_countries = sa_countries.merge(world, on = 'Country Code')
sa_corrs = sa_corrs.merge(sa_countries, how = 'outer')

## going to include MENA with SSA since no countries considered in MENA anyway
ssa_countries = regions2[(regions2['Region'] == 'Sub-Saharan Africa') | 
                         (regions2['Region'] == 'Middle East & North Africa')]
ssa_countries = ssa_countries.merge(world, on = 'Country Code')
ssa_corrs = ssa_corrs.merge(ssa_countries, how = 'outer')

eca_countries = regions2[regions2['Region'] == 'Europe & Central Asia']
eca_countries = eca_countries.merge(world, on = 'Country Code')
eca_corrs = eca_corrs.merge(eca_countries, how = 'outer')

eap_countries = regions2[regions2['Region'] == 'East Asia & Pacific']
eap_countries = eap_countries.merge(world, on = 'Country Code')
eap_corrs = eap_corrs.merge(eap_countries, how = 'outer')

nam_countries = regions2[regions2['Region'] == 'North America']
nam_countries = nam_countries.merge(world, on = 'Country Code')
nam_corrs = nam_corrs.merge(nam_countries, how = 'outer')

latam_countries = regions2[regions2['Region'] == 'Latin America & Caribbean']
latam_countries = latam_countries.merge(world, on = 'Country Code')
latam_corrs = latam_corrs.merge(latam_countries, how = 'outer')

# gs = plt.GridSpec(2, 5, wspace=0.2, hspace=0) 
# ax0 = plt.subplot(gs[0, :2])
# ax1 = plt.subplot(gs[1, :2])
# ax2 = plt.subplot(gs[0, 2])
# ax3 = plt.subplot(gs[1, 2])
# ax4 = plt.subplot(gs[0, 3])
# ax5 = plt.subplot(gs[1, 3])
# #fig1, (ax0, ax1) = plt.subplots(ncols=1, nrows = 2,sharex = True)
# merged_gdf.plot(ax = ax0, column = 'Correlation', missing_kwds={'color':'grey'}, 
#                 cmap = 'PuOr', legend = False, vmin = -1, vmax = 1)
# eca_corrs.plot(ax = ax1, column = 'corrs', missing_kwds={'color':'grey'}, 
#                 cmap = 'PuOr', legend = False, vmin = -1, vmax = 1)
# eap_corrs.plot(ax = ax2, column = 'corrs', missing_kwds={'color':'grey'}, 
#                 cmap = 'PuOr', legend = False, vmin = -1, vmax = 1)
# nam_corrs.plot(ax = ax3, column = 'corrs', missing_kwds={'color':'grey'}, 
#                 cmap = 'PuOr', legend = False, vmin = -1, vmax = 1)
# latam_corrs.plot(ax = ax4, column = 'corrs', missing_kwds={'color':'grey'}, 
#                 cmap = 'PuOr', legend = False, vmin = -1, vmax = 1)
# ssa_corrs.plot(ax = ax5, column = 'corrs', missing_kwds={'color':'grey'}, 
#                 cmap = 'PuOr', legend = True, vmin = -1, vmax = 1)
# ax1.set_xlim(-100,179)
# ax2.set_xlim(50,179)

# ax1.set_ylim(30,90)
# ax5.legend(bbox_to_anchor=(1.5, 1), bbox_transform=ax1.transAxes)

# merged_gdf.plot(column = 'Correlation', missing_kwds={'color':'grey'}, 
#                 cmap = 'PuOr', legend = True, vmin = -1, vmax = 1)

# gs2 = plt.GridSpec(3,1)
# ax3 = plt.subplot(gs2[0, 0])
# ax4 = plt.subplot(gs2[1, 0])
# ax5 = plt.subplot(gs2[2, 0])
# fig, (ax3, ax4, ax5) = plt.subplots(ncols=3, nrows = 1, figsize = (45, 20))
# nam_corrs.plot(ax = ax3, column = 'corrs', missing_kwds={'color':'lightgrey'}, 
#                 cmap = 'PuOr', legend = False, vmin = -1, vmax = 1)
# latam_corrs.plot(ax = ax4, column = 'corrs', missing_kwds={'color':'lightgrey'}, 
#                 cmap = 'PuOr', legend = False, vmin = -1, vmax = 1)
# ssa_corrs.plot(ax = ax5, column = 'corrs', missing_kwds={'color':'lightgrey'}, 
#                 cmap = 'PuOr', legend = False, vmin = -1, vmax = 1)

###############################################################################
######################## FIG 4 INDEX PERFORMANCE ##############################

###############################################################################
############## FIG 5 COST OF INDIVIDUAL VS POOLED INSTRUMENTS #################
##################### alternative for individual countries ####################

## PANEL A: 99% VaR comparison (i.e., reserves to be held)
## input: df with the generation value, countries as columns
## var is, well, the var 
## bond_rates can be a single value or a dataframe with different rates per country
## and it is the decimal value, not percent
def calc_reserves(gen_vals2, var = 95, bond_rates = [0.0492], time_horizon = 30,
                  plot = False): 
    reserves = pd.DataFrame(index = gen_vals2.columns, columns = ['res_val', 'ann_pay'])
    for c in gen_vals2.columns: 
        country = gen_vals2[c]
        country.dropna(axis = 0, inplace = True)
        
        try: 
            reserves.loc[c, 'res_val'] = np.percentile(country, 100-var)
        
            if len(bond_rates) == 1: 
                rate = bond_rates[0]
            else: 
                rate = bond_rates[c] ## NEED TO MAKE THIS WORK :) 
                
            num = rate * pow((1 + rate), time_horizon)
            deno = pow((1 + rate), time_horizon) - 1
            reserves.loc[c, 'ann_pay'] = reserves.loc[c, 'res_val'] * (num/deno)
        except: 
            reserves.loc[c, :] = np.nan
    
    if plot == True: 
        sns.scatterplot(reserves['res_val'], reserves['ann_pay'], hue = reserves.index, 
                        s = 50)
        plt.xlabel("Reserve Fund Value ($)")
        plt.ylabel("Amortized Annual Payment ($)")
        plt.legend(frameon = False)
    
    return reserves

countries = predictions.columns
necessary_res = calc_reserves(gen_vals[countries], plot = False)

# fig, (ax1, ax2) = plt.subplots(1, 2)
# sns.scatterplot(necessary_res['res_val'], necessary_res['ann_pay'], 
#                 ax = ax1, hue = necessary_res.index, s = 50)
# ax1.set_xlabel("Reserve Fund Value ($B)", fontsize = lab_size)
# ax1.set_ylabel("Amortized Annual Payment ($ '000s)", fontsize = lab_size)
# #ax1.get_legend().remove()
# ax1.set_xticks([0, 500*pow(10,6), 1000*pow(10,6), 1500*pow(10,6), 
#                 2000*pow(10,6)], #2.5*pow(10,6), 3*pow(10,6)], 
#                 ['0', '500', '1000', '1500', '2000'], 
#                 fontsize = tick_size)
# ax1.set_yticks([0, 50*pow(10,6), 100*pow(10,6), 150*pow(10,6), 
#                 200*pow(10,6)], #2.5*pow(10,6), 3*pow(10,6)], 
#                 ['0', '50', '100', '150', '200'], 
#                 fontsize = tick_size)
# ax1.legend(frameon = False, fontsize = tick_size-2)

# x = np.linspace(0, 100*pow(10,9))
# y = np.linspace(0, 100*pow(10,9))
# sns.scatterplot(loading['loading'], loading['contrib_port'], 
#                 ax = ax2, s = 100, hue = loading.index)
# plt.plot([loading.min(), loading.max()], 
#          [loading.min(), loading.max()],# ax = ax2, 
#           linestyle = '--', color = 'black', label = '1:1 line')
# ax2.set_yscale("log")
# ax2.set_xscale("log")
# ax2.set_xlabel("Portfolio Premiums ($M)", fontsize = lab_size)
# ax2.set_ylabel("Individual Premiums ($M)", fontsize = lab_size)
# ax2.set_xticks([10*pow(10,3), 10*pow(10,4), 10*pow(10,5), \
#             10*pow(10,6), 100*pow(10,6), 1000*pow(10,6), 
#             10000*pow(10,6), 100*pow(10,9)], 
#             ['.010', '.100', '1', '10', '100', '1,000', '10,000', '100,000'], \
#             fontsize = tick_size)
# ax2.set_yticks([10*pow(10,3), 10*pow(10,4), 10*pow(10,5), \
#             10*pow(10,6), 100*pow(10,6), 1000*pow(10,6), 
#             10000*pow(10,6), 100*pow(10,9)], 
#             ['.010', '.100', '1', '10', '100', '1,000', '10,000', '100,000'], \
#             fontsize = tick_size)
# plt.fill_between(x, y, 100*pow(10,9), color = '#4EB265', alpha = .25, 
#     label ='Portfolio More Cost-Effective')
# ax2.set_xlim(0, 100*pow(10,9))
# ax2.set_ylim(0, 100*pow(10,9))
# ax2.legend(fontsize = tick_size-2, frameon = False, loc = 'lower right')
  
## plot with bars 
ind = np.arange(len(expected_payouts_pool))[:14]
width = np.min(np.diff(ind))/3
length = len(expected_payouts_pool[:(21-14)])

fig, ax = plt.subplots()
rects2 = ax.bar(ind, expected_payouts_pool.loc[length:,'sum_individ_var'], width = width,
        label = f'Sum of Individ. VaRs, {var_test}', color = 'orange',align='edge')
rects3 = ax.bar(ind+width, expected_payouts_pool.loc[length:,'VaR'], width = width, 
        label = f'Pool VaR {var_test}', color = 'purple',align='edge')
ax.set_xlabel("Number of Pool Participants", fontsize = 18) 
ax.set_ylabel("$M", fontsize = 18)
ax.set_xticks(ind + width/2, ('10', '11', '12', '13', '14', '15', '16',
                              '17', '18', '19', '20', '21', '22', '23'), 
           fontsize = 16)
ax.set_yticks([0, 0.4*pow(10,12), 0.8*pow(10,12),
               1.2*pow(10,12)],
           ['0', '400', '800', '1000'], 
           fontsize = 16)
ax.legend(frameon = False, fontsize = 16, loc = 'upper left')
ax.set_xlim(0, 14)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                int(round(height/(1*pow(10,9)),0)),#'%d' % int(height),
                ha='center', va='bottom', fontsize = 14)

autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.draw()

fig.savefig("bar_plots_det.png", dpi = 500)
###############################################################################
####################### FIG S2 CORRELATION HEATMAP ############################
lst = list(all_sig['Country Name'].values)
gen_data_pool = gen_data[np.intersect1d(gen_data.columns, lst)]
det_gen_pool = det_gen[np.intersect1d(det_gen.columns, lst)]

plt.rc('xtick', labelsize = 16)  # x tick labels fontsize
plt.rc('ytick', labelsize = 16)  # y tick labels fontsize

#fig, (ax1, ax2) = plt.subplots(ncols = 1, nrows = 2)
#plt.figure()
mask2 = np.triu(np.ones_like(gen_data_pool.corr()))
fig, ax = plt.subplots(figsize = (20,25))
sns.heatmap(gen_data_pool.loc[:, 'Albania':].corr(), mask = mask2, ax = ax, annot_kws={"fontsize":10}, 
            cmap = 'PuOr', vmin=1, vmax=-1)
#plt.savefig("Figures/raw_gen_corrs.png")
#plt.close()

#plt.figure()
mask = np.triu(np.ones_like(det_gen_pool.loc[:, 'Albania':].corr()))
fig, ax = plt.subplots(figsize = (20,25))
sns.heatmap(det_gen_pool.loc[:, 'Albania':].corr(), mask = mask, ax = ax,annot_kws={"fontsize":10}, 
            cmap = 'PuOr', vmin=1, vmax=-1)
#plt.savefig("Figures/det_gen_corrs.png", bbox_inches = 'tight', dpi = 500)
#plt.close()

# fig = plt.figure()
# ax = plt.axes()
# im = merged_vals.plot(ax = ax, column = 'hydro_pct_gdp', missing_kwds={'color':'lightgrey'}, 
#                   cmap = 'viridis_r', legend = True)
# ax.set_aspect('auto')
# plt.tight_layout(h_pad=1)

#fig, (ax1, ax2) = plt.subplots(ncols=1, nrows = 2, sharex=True, sharey=True, figsize = (25, 20))
# hydro_gdf.plot(column='2015', missing_kwds={'color': 'lightgrey'},
#             figsize = (25, 20), cmap = 'YlGnBu', legend = True)
# plt.title('Percent Generation From Hydropower',fontsize=14)

# merged_vals.plot(column = 'hydro_pct_gdp', missing_kwds={'color':'lightgrey'}, 
#                 figsize = (25, 20), cmap = 'YlGnBu', legend = True)
# plt.title(f'Hydropower Sales as Percent of GDP', fontsize=14)




