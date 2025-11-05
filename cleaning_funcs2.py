# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:53:48 2022

@author: rcuppari
"""
## good resources: https://joehamman.com/2013/10/12/plotting-netCDF-data-with-Python/
## https://neetinayak.medium.com/combine-many-netcdf-files-into-a-single-file-with-python-469ba476fc14
## plotting: https://www.python-graph-gallery.com/map/
import geopandas as gpd
from shapely.geometry import mapping
import glob 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import itertools
from itertools import chain, combinations
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
import gc 
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from sklearn.linear_model import LinearRegression
from permetrics import RegressionMetric
   
def id_hydro_countries(pct_gen = 90, year = 2015, cap_fac = False, capacity = False):
    wb_hydro = pd.read_csv("Other_data/wb_hydro%.csv")
    country_conversion = pd.read_csv("Other_data/country_conversions.csv")
    
    countries = wb_hydro[wb_hydro['2015'] >= pct_gen][['Country Name', 'Country Code']] 
    countries = countries.merge(country_conversion, on = 'Country Name')    
    countries = countries.rename(columns={'EIA Name':'Country'})

    gen_data = pd.read_csv("Other_data/eia_hydropower_data2.csv")
    ## remove spaces from gen data country names to make consistent
    gen_data.iloc[:,0] = gen_data.iloc[:,0].str.lstrip()
    gen_data = gen_data.rename(columns = {'hydroelectricity net generation (billion kWh)': 'Country'})
    
    ## keep generation for countries with high hydro 
    gen_data2 = gen_data[gen_data.iloc[:,0].isin(countries['Country'])]

    gen_data3 = gen_data2.merge(countries, on = 'Country')

    if cap_fac == False:        
        return gen_data3
    
    else: 
        cap_data = pd.read_csv("Other_data/IRENA_global_hydro_stats_new.csv")
        cap_data = cap_data.rename(columns = {'Unnamed: 0':'Country/area'})
        ## get right country names
        cap_data = cap_data.merge(countries, left_on = 'Country/area', right_on = 'Country')
        
        ## convert cap_data to kW to match gen data
        ## note: billion kWh = 1,000,000,000 = 1M MWh = 1000 GWh
        if capacity == True: 
            return cap_data
        
        else:
            cap_factor_df = pd.DataFrame(index = gen_data3.iso_a3, 
                                         columns = cap_data.columns[1:-5])
            
            for c in cap_data.columns[1:-5]:
                cap_data.loc[:,c] = pd.to_numeric(cap_data.loc[:,c], errors = 'coerce')
                ## put capacity into B kWh from MW, to match the generation data 
                cap_data.loc[:,c] = (cap_data.loc[:, c]/1000)/1000*8760 
            
                gen_data3.loc[:,c] = pd.to_numeric(gen_data3.loc[:,c], errors = 'coerce')
            
                for r in gen_data3.iso_a3:
                    country_gen = gen_data3[gen_data3['iso_a3'] == r][c]
                    country_cap = cap_data[cap_data['iso_a3'] == r][c]
                    
                    try: 
                        cap_factor_df.loc[r,c] = (country_gen.values/country_cap.values)[0]
                    except: 
                        cap_factor_df.loc[r,c] = 'NaN'
                
                ## I know this is awfully clumsy, but I am not sure why it is not working otherwise
                cap_factor_df[c] = pd.to_numeric(cap_factor_df[c], errors = 'coerce')
    
            cap_factor_df.reset_index(inplace = True)
            
            cap_factor_df = cap_factor_df.merge(countries, on = 'iso_a3')
            
            cols = gen_data3.columns
            drop = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999']
            cols = [col for col in gen_data3.columns if col not in drop]
            
            cap_df = cap_factor_df[cols] 
            
            return cap_df

## c is country
def index_vetting(c, x, y, random_state):
    
    #define cross-validation method to use
    cv = LeaveOneOut()
    #build multiple linear regression model
    loo_model = LinearRegression()
    
    X_train, X_test, y_train, y_test = tts(x, y, test_size = .2,\
                           random_state = random_state)
    model = sm.OLS(y_train, X_train).fit()

    ## check the fit on test data 
    predicted = model.predict(X_test)
    pred_r2 = (predicted.corr(y_test)**2)

    ## the values we will want to output to track the index/index performance
    inputs = list(x.columns)
    coefs = list(model.params)
    pval = list(model.pvalues)
    r_all = model.predict(x).corr(y)
    r2_all = r_all**2

    #use LOOCV to evaluate model
    scores = cross_val_score(loo_model, x, y, scoring='neg_mean_absolute_error',
                              cv=cv, n_jobs=-1)

    ## also make sure to evaluate normality and hetereoscedasticity 
    white_test = het_white(model.resid, model.model.exog)
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    shap, shap_p = shapiro(y)
    dagpear_stat, dagpear_p = normaltest(y)

    nnse = RegressionMetric(y.values, model.predict(x).values).get_metrics_by_list_names(["NNSE"])['NNSE']

    and_darling = anderson(y)
    if and_darling[0] < and_darling[1][-1]: 
        and_darl_sig_1pct = 'normal at p 0.01'
    else: and_darl_sig_1pct = 'not-normal at p 0.01'

    mae = abs(scores).mean()

    # ## print results and save to df
    # print(f"{c} MAE: {mae}")
    # print(f"{c} white test p-value: {white_test[1]}")
    # print(f"{c} bp p-value: {bp_test[1]}")
    # print(f"{c} d'agostino and pearson's test p-value: {dagpear_p}")
    # print(f"{c} shapiro p-value: {shap_p}")
    # print(f"{c} anderson-darling result: {and_darl_sig_1pct}")
    
    ## saving this snippet of code in case we ever want to break down the 
    ## Anderson-Darling values more
    # ad_result = anderson(y, dist='norm')
    
    # print(f"Anderson-Darling Statistic (A²): {ad_result.statistic:.4f}")
    
    # print("\nSignificance Level (%) | Critical Value | Test Decision")
    # print("-" * 50)
    # for significance, critical_value in zip(ad_result.significance_level, ad_result.critical_values):
    #     decision = "Reject H₀" if ad_result.statistic > critical_value else "Fail to Reject H₀"
    #     print(f"{significance:>21}% | {critical_value:>14.4f} | {decision}")


    validation_df = pd.DataFrame([mae, nnse, 
                             white_test[1], bp_test[1], 
                             shap_p, dagpear_p, and_darl_sig_1pct])

    new = {'Country Name': c, \
            'r2': r_all, 'inputs':[inputs], 'coefs':[coefs], \
            'df':[['4']], 'pval':[pval], 
            'r2_train': model.rsquared_adj, 'test_r2': pred_r2} 
    new = pd.DataFrame(new)

    
    return new, validation_df

def get_dfs(country, country_geom, ds, hybas = 'country'):
    ## need to import 
    ## 1) MODIS LST
    ## 2) IMERG precip
    ## 3) MODIS snow cover 
    ## 4) MODIS EVI 

    ## want to save the dfs for all of the individual hydro units, as averages
    ## if watersheds == True 
        
    ## MODIS LST 
    modis_df = clean_modis_lst(country_geom, file_header = 'MOD11C3*.hdf', hybas = hybas)
    #    modis_df = 'place holder'   
    
    ## IMERG 
    imerg_df = agg_data(country_geom, ds, hybas, 'precipitation', time = '')
    #    imerg_df = 'place holder'   
    
    ## MODIS SNOW
    snow_df = clean_modis_snow(country_geom, file_header = 'MOD10CM*.hdf', hybas = hybas)
    
    ## MODIS EVI/NDVI
    evi_df, ndvi_df = clean_modis_evi(country_geom, file_header = 'MOD13C2*.hdf', hybas = hybas)
    
    return modis_df, imerg_df, snow_df, evi_df, ndvi_df


def remove_lin_trend(input_data):  ## input data should have the year 
    ## with the generation data, need to transpose so that the years are a 
    ## column instead of a row     
    df = input_data.T
    df.reset_index(inplace = True)
    ## drop the rows with the column names 
    df = df.iloc[1:,:]
    df.columns = ['year', 'outcome']

    ## if there is no data (or data is equal to 0), drop 
    ## nans will be dropped too 
    df.iloc[:,1] = pd.to_numeric(df.iloc[:,1], errors = 'coerce')
    df = df[df.iloc[:,1] > 0]
    
    ## get to the juicy part
    y = pd.to_numeric(df.iloc[:,1]) ## outcome should be second column
    X = sm.add_constant(pd.to_numeric(df.iloc[:, 0])) ## year should be first
    
    est_year = sm.OLS(y, X)
    est_year2 = est_year.fit()

    if est_year2.pvalues[1] < .05: 
        year_pred = est_year2.predict(X)

        det_data = pd.to_numeric(df.iloc[:,1]) - year_pred
        print('detrending necessary')
        detrending = True
    else: 
        det_data = df.iloc[:,1]
        year_pred = 0
        detrending = False
    det_data = pd.DataFrame(det_data)
    det_data['year'] = pd.to_numeric(df.iloc[:,0])
    det_data.columns = ['outcome', 'year']
    
    df['year'] = pd.to_numeric(df['year'])
    return det_data, df, detrending, year_pred

## clip precip to aoi, get precip, and make into df 
def agg_data(geom, ds, hybas, var, time): ## geom is the bounds, variable is a string with var of interest

    if hybas == 'watersheds':    
        
        for g in np.arange(len(geom)): 
            print(g)    
            geom2 = gpd.GeoSeries(geom.iloc[g,-1])
            name = geom.loc[g, 'HYBAS_ID']
            
            new_sub_var = ds.rio.clip(geom2.geometry.apply(mapping), geom.crs, drop = False) 
            
            new_sub_var = new_sub_var.to_dataframe()
            new_sub_var = new_sub_var.dropna() 
            new_sub_var.reset_index(inplace = True) 
            
            if (var == 'Snow_Cover_Monthly_CMG') | (var == 'dmsp') | (var == 'CMG 0.05 Deg Monthly EVI') | (var == 'CMG 0.05 Deg Monthly NDVI'):  
                new_sub_var['time'] = time
                #print(new_var)
                ## make into gpd df
                new_var3 = gpd.GeoDataFrame(
                    new_sub_var[['time', var]], 
                    geometry = gpd.points_from_xy(new_sub_var.y, new_sub_var.x))
    
            else: 
                ## make into gpd df
                new_var3 = gpd.GeoDataFrame(
                    new_sub_var[['time', var]], 
                    geometry = gpd.points_from_xy(new_sub_var.lon, new_sub_var.lat))
            ## data is already in monthly format, but want to get the mean for the aoi
            
            ## COMBINE -- keep index as time, but place geometry with hybas ID in name 
            ## had kept the geometry to unite all points within a certain geometry and then group by time
            new_sub_var2 = pd.DataFrame(new_var3.dissolve(by = 'time')[var])            
            new_sub_var2.columns = [f"{var[:6]}_{name}"]
            
            if g == 0: 
                sub_vars = new_sub_var2
            else: 
                sub_vars = sub_vars.merge(new_sub_var2, left_index = True, right_index = True)
        return sub_vars 
    
    else: 
#        name = str(geom.iloc[0,0])
        new_var = ds.rio.clip(geom.geometry.apply(mapping), geom.crs, drop = False)
            
        new_var = new_var.to_dataframe()
        new_var = new_var.dropna() 
        new_var.reset_index(inplace = True) 
    
        #    new_var['month'] = new_var.time.dt.month
        #    new_var['year'] = new_var.time.dt.year
        
        if (var == 'Snow_Cover_Monthly_CMG') | (var == 'dmsp') | (var == 'CMG 0.05 Deg Monthly EVI') | (var == 'CMG 0.05 Deg Monthly NDVI'):  
            new_var['time'] = time

            ## make into gpd df
            new_var3 = gpd.GeoDataFrame(
                new_var[['time', var]], 
                geometry = gpd.points_from_xy(new_var.y, new_var.x))
    
        else: 
            ## make into gpd df
            new_var3 = gpd.GeoDataFrame(
                new_var[['time', var]], 
                geometry = gpd.points_from_xy(new_var.lon, new_var.lat))
        ## data is already in monthly format, but want to get the mean for the aoi
        
        agg_gdf = pd.DataFrame(new_var3.dissolve(by = 'time')[var])
#        agg_gdf.columns = [f"{var[:6]}_{name}"]
        agg_gdf.columns = [f"{var[:6]}"]
        
        return agg_gdf

def get_geom(iso_code): 
    world_filepath = gpd.datasets.get_path('naturalearth_lowres')
    world = gpd.read_file(world_filepath)
     
    country_bounds = world.loc[world['iso_a3'] == iso_code] # get country boundary
    geom = country_bounds.geometry   
    return geom

def clean_modis_lst(geom, file_header, hybas): ## need country boundary geometry
    ## need to have all MODIS files downloaded in directory
    urls = pd.read_csv("MOD_LST2/urls.txt", sep = '/', header = None).iloc[:,7:]
    urls.columns = ['time', 'name'] 
    
#    paths =  glob.glob('MOD_LST2/' + file_header) 

    agg_gdf = gpd.GeoDataFrame() ## initialize df to store final combined modis data

    import rioxarray as rxr

    for name in urls['name']: ### will read and clean them all one at a time and then convert 
        ## can use the list of urls to identify year-month 
        ## thankfully reading in with '/' does half the work! 
        print(name)
        time = urls[urls['name'] == name].iloc[0,0]
        print(time)
        
        try: 
            ds = rxr.open_rasterio("MOD_LST2/" + name, masked = True)
            
    #        geom = geom.set_crs('EPSG:4326')
            new_var = ds.rio.clip(geom.geometry.apply(mapping), crs = geom.crs, drop = False)#'EPSG:4326', drop = False)
            gc.collect() 
            
            new_var1 = new_var.to_dataframe()
            new_var1 = new_var1.dropna() 
            new_var1.reset_index(inplace = True) 
            new_var1['time'] = pd.to_datetime(time)
    
            ## make into gpd df
            new_var1 = gpd.GeoDataFrame(
                new_var1[['time', 'LST_Day_CMG']], 
                geometry = gpd.points_from_xy(new_var1.x, new_var1.y))
            
            new_var1 = new_var1.dissolve(by = 'time')
            agg_gdf = pd.concat([agg_gdf, new_var1])
            
            gc.collect()
            
        except: print('failed to read ' + str(name))
        ## data is already in monthly format, but want to get the mean for each 
        ## subbasin of interest 
        
    return agg_gdf ## this will be a geodataframe with geometry and will hold the whole modis timeseries       

def clean_modis_snow(geom, file_header, hybas = 'country', country = '', subbasin = '', continent = ''):
    ## need to have all MODIS files downloaded in directory
    file_header = 'MOD_snow/' + file_header
    paths =  glob.glob(file_header) 
    count = 1
    year1 = paths[1][18:22] ## extract first year 
    
    
    new_df = pd.DataFrame() ## initialize df to store final combined modis data
    import rioxarray as rxr

    for name in paths[25:45]: ### will read and clean them all one at a time and then convert 
        year = name[18:22]
        if year == year1: 
            count += 1
        else: 
            count = 1
            
        print(name)
        print(year)
        time = pd.to_datetime(year + '-' + str(count))
        print(time)
        print()
 
       
        ds = rxr.open_rasterio(name, masked = True)
        
        new_var3 = agg_data(geom = geom, ds = ds, var = 'Snow_Cover_Monthly_CMG', 
                           hybas = hybas, time = time)    
                    
        
        new_df = pd.concat([new_df, new_var3])
        
        year1 = year
           
    return new_df ## this will be a geodataframe with geometry and will hold the whole modis timeseries       

def clean_modis_evi(geom, file_header, hybas = 'country'): ## need country boundary geometry
    ## need to have all MODIS files downloaded in directory

     file_header = 'MOD_EVI/' + file_header
     paths =  glob.glob(file_header) 
     paths = sorted(paths) ## sort so that it is oldest to newest
     count = 1
     year1 = paths[1][17:21] ## extract first year 
     
     
     evi_df = pd.DataFrame() ## initialize df to store final combined modis data
     ndvi_df = pd.DataFrame() ## initialize df to store final combined modis data
     
     import rioxarray as rxr
    
     for n, name in enumerate(paths): ### will read and clean them all one at a time and then convert 
         year = name[17:21]
         if year == year1: 
             count += 1 
         else: 
             count = 1
             
         print(name)
         print(year)
         time = pd.to_datetime(year + '-' + str(count))
         print(time)
         print()    
      
         try: 
             ds = rxr.open_rasterio(name, masked = True)
 
            
             new_evi3 = agg_data(geom = geom, ds = ds, var = 'CMG 0.05 Deg Monthly EVI', 
                               hybas = hybas, time = time)    
             new_evi3.columns = ['EVI']
             evi_df = pd.concat([evi_df, new_evi3])
             gc.collect()
            
            # print('evi saved')
             
             new_ndvi3 = agg_data(geom = geom, ds = ds, var = 'CMG 0.05 Deg Monthly NDVI', 
                               hybas = hybas, time = time)    
             new_ndvi3.columns = ['NDVI']
             
             ndvi_df = pd.concat([ndvi_df, new_ndvi3])
             gc.collect()
            # print('ndvi saved')
        
         except: print('failed to read ' + str(name))
         
         year1 = year
         ## data is already in monthly format, but want to get the mean for each 
         ## subbasin of interest 
         return evi_df, ndvi_df ## this will be a geodataframe with geometry and will hold the whole m

     
 
####################### run regressions ##############################
def plot_reg(training, y_train, predicted, y_test, r2, country, count): 
    val_x = training.mean()
    val_y = y_train.mean()
    plt.figure()
    plt.scatter(training, y_train, label = 'Training Set')
    plt.scatter(predicted, y_test, label = 'Test Set')
    plt.xlabel("Predicted " + country, fontsize = 16)
    plt.ylabel("Observed " + country, fontsize = 16)
    plt.legend()
    r2 = round(r2,2)
    plt.annotate(r2, (val_x, val_y))
    plt.savefig('Results/' + str(country) + str(count))
    plt.close()


## gdf should have a time column ('time') that is at least monthly
def group_data(gdf): 
    gdf['time'] = pd.to_datetime(gdf['time'])
    gdf['month'] = gdf.time.dt.month    
    gdf['year'] = gdf.time.dt.year    
        
    ## aggregate annually 
    gdf_ann = gdf.groupby('year').mean()

    ## and then seasonally  
    spri_mons = [3,4,5]
    summ_mons = [6,7,8]
    wint_mons = [12,1,2]
    fall_mons = [9,10,11]

    isin_spring = gdf['month'].isin(spri_mons)
    spring = gdf.loc[isin_spring]
        
    isin_fall = gdf['month'].isin(fall_mons)
    fall = gdf.loc[isin_fall]
    
    isin_winter = gdf['month'].isin(wint_mons)
    winter = gdf.loc[isin_winter]
    
    isin_summer = gdf['month'].isin(summ_mons)
    summer = gdf.loc[isin_summer]
    
    summer = summer.groupby('year').mean()
    spring = spring.groupby('year').mean()
    winter = winter.groupby('year').mean()
    fall = fall.groupby('year').mean()
    
    ## group by year 
    cols = spring.columns
    cols_summ = 'summ_' + cols
    cols_fall = 'fall_' + cols
    cols_wint = 'wint_' + cols
    cols_spri = 'spri_' + cols

    summer.columns = cols_summ        
    fall.columns = cols_fall        
    winter.columns = cols_wint        
    spring.columns = cols_spri 
    
    max_rows = max(len(summer), len(fall), len(spring), len(winter))
    
    if len(summer) == max_rows: 
        
        if len(fall) == max_rows: 
            gdf_seas = summer.merge(fall, left_index = True, right_index = True)
            
            if len(winter) == max_rows:
                gdf_seas = gdf_seas.merge(winter, left_index = True, right_index = True)        
                
                if len(spring) == max_rows: 
                    gdf_seas = gdf_seas.merge(spring, left_index = True, right_index = True)
                     
        else: 
            if len(winter) == max_rows:
                gdf_seas = summer.merge(winter, left_index = True, right_index = True)
                
                if len(spring) == max_rows: 
                    gdf_seas = gdf_seas.merge(spring, left_index = True, right_index = True)

            elif len(spring) == max_rows: 
                gdf_seas = summer.merge(spring, left_index = True, right_index = True)
                
        gdf_seas.reset_index(inplace = True)   
        grouped = gdf_ann.merge(gdf_seas, on = 'year')
        return grouped
    
    elif len(fall) == max_rows: ## no summer
                
        if len(winter) == max_rows:
            gdf_seas = fall.merge(winter, left_index = True, right_index = True)        
            
            if len(spring) == max_rows: 
                gdf_seas = gdf_seas.merge(spring, left_index = True, right_index = True)
                
        elif len(spring) == max_rows: 
            gdf_seas = fall.merge(spring, left_index = True, right_index = True)

        else: 
            gdf_seas = fall
                
        gdf_seas.reset_index(inplace = True)   
        grouped = gdf_ann.merge(gdf_seas, on = 'year')
        return grouped     
            
    elif len(winter) == max_rows: ## no summer or fall 
        if len(spring) == max_rows: 
            gdf_seas = winter.merge(spring, left_index = True, right_index = True)
        else: 
            gdf_seas = winter
        gdf_seas.reset_index(inplace = True)   
        grouped = gdf_ann.merge(gdf_seas, on = 'year')
        return grouped  
    
    elif len(spring) == max_rows: 
        gdf_seas = spring
        gdf_seas.reset_index(inplace = True)   
        grouped = gdf_ann.merge(gdf_seas, on = 'year')
        return grouped  
    
    else: 
        return gdf_ann

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def r2_calc(predicted, actual): 
    RSS = ((predicted - actual)**2).sum()
    TSS = ((actual - actual.mean())**2).sum()
    r2 = 1 - (RSS/TSS)
    return r2

## input gdf should be a dataframe with timestamps, at a seasonal 
## or annual frequency (not gdf, I lied)
## pvals_all asks to check all pvalues 
def regs(outcome, input_gdf, country, detrending = False, 
                    year_pred = '', plot = True, threshold = 0.6, test_threshold = .3, 
                    pvals_all = False, snow = True, num = 5):  
    count = 0
    old_r2 = -10    
    regression_output = pd.DataFrame()    
    reg_top = pd.DataFrame()    
    ## merge outcome and data of interest
    all_data = outcome.merge(input_gdf, on = 'year')
    
    ## and re-sort for x and y while ensuring we are not wasting time
    ## with regressions that are not useful
    x = all_data.drop(['outcome'], axis = 1, errors = 'ignore')
    x = x.loc[:,~x.columns.str.contains('month', case=False)]         
    x = x.loc[:,~x.columns.str.contains('year', case=False)]         
    x = x.loc[:,~x.columns.str.contains('Unnamed', case=False)]         
    
    if snow == False: 
        x = x.loc[:,~x.columns.str.contains('Snow', case=False)] 
        
    x = sm.add_constant(x, has_constant = 'add')
    y = all_data['outcome']
    
    ## only want to save anything with r2 > 0.5 for train and test 
    ## and of course with solid p-values 
    ## also, not more than 5 inputs
    if len(x) > 2: 
        X_train, X_test, y_train, y_test = tts(x, y, test_size = .3, random_state = 1)
    
        for subset in powerset(x.columns): 
            if 'const' in itertools.chain(subset):
    #            print('we good')
                if (len(subset) > 0) & (len(subset) < num):
                    #print(list(subset))
                    model = sm.OLS(y_train, X_train[list(subset)]).fit()
            
                    training = model.predict(X_train[list(subset)])
                    
                    ## check the fit on test data 
                    predicted = model.predict(X_test[list(subset)])
                    pred_r2 = (predicted.corr(y_test)**2)
                    ## store regression inputs, p-values, and adj rsquared 
            
                    pval = list(model.pvalues)
                    r2_mod = float(model.rsquared_adj)
                    print(r2_mod)
                    
                    if ((r2_mod > threshold) and (pred_r2 > test_threshold)):
                        
                        if pvals_all == False: 
                            
                            series = list(subset)
                            predicted_all = model.predict(x[list(subset)])
                            r2_all = r2_calc(predicted_all, y)
                        
                            new = {'inputs':[series], 'r2': r2_all, 'pval':[pval], 'test_r2': pred_r2} 
                            new = pd.DataFrame(new)
                            regression_output = pd.concat([regression_output, new], axis = 0)
    
                            ## also make sure to plot! 
                            if plot == True: 
                                count += 1
                                try: 
                                    plot_reg(training, y_train, predicted, y_test, r2_all, country, count)
                                except: print("Plotting failed")                        
                            print('country r2 all: ' + str(r2_all))
                            print('country r2 old: ' + str(old_r2))
                            
                            if r2_all > old_r2: 
                                reg_top = new.copy()
                                old_r2 = r2_all
                                
                        else:
                            
                            pval2 = pval[1:]
                            if all(i <= pvals_all for i in pval2):
                            
                                series = list(subset)
                                predicted_all = model.predict(x[list(subset)])
                                r2_all = r2_calc(predicted_all, y)
                            
                                new = {'inputs':[series], 'r2': r2_all, 'pval':[pval], 'test_r2': pred_r2} 
                                new = pd.DataFrame(new)
                                regression_output = pd.concat([regression_output, new], axis = 0)
        
                                ## also make sure to plot! 
                                if plot == True: 
                                    count += 1
                                    try: 
                                        plot_reg(training, y_train, predicted, y_test, r2_all, country, count)
                                    except: print("Plotting failed")                        
                                print('country r2 all: ' + str(r2_all))
                                print('country r2 old: ' + str(old_r2))
                                
                                if r2_all > old_r2: 
                                    reg_top = new.copy()
                                    old_r2 = r2_all
                                
                            
    else: print(country + ' insufficient data')   
    
    return regression_output, reg_top


######################### save outputs! ##############################
def write_dict(dict, name): 
    import pickle
    with open("Results/" + name, "wb") as handle: 
        pickle.dump(dict, handle)
    return "success"













