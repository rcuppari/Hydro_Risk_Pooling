# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:53:48 2022

@author: rcuppari
"""
## good resources: https://joehamman.com/2013/10/12/plotting-netCDF-data-with-Python/
## https://neetinayak.medium.com/combine-many-netcdf-files-into-a-single-file-with-python-469ba476fc14
## plotting: https://www.python-graph-gallery.com/map/
import xarray
import os 
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

## clip precip to aoi, get precip, and make into df 
def agg_data(geom, ds, hybas, var, time): ## geom is the bounds, variable is a string with var of interest
    
    new_var = ds.rio.clip(geom.geometry.apply(mapping), geom.crs, drop = False)
        
    new_var = new_var.to_dataframe()
    new_var = new_var.dropna() 
    new_var.reset_index(inplace = True) 

#    new_var['month'] = new_var.time.dt.month
#    new_var['year'] = new_var.time.dt.year
    
    if var == 'Snow_Cover_Monthly_CMG': 
        new_var['time'] = time
        #print(new_var)
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
    
    agg_gdf = new_var3.dissolve(by = 'time')

    return agg_gdf



## if you insert subbasin, will use hybas ID 
def subset_imerg(country, geom, ds, continent = '', variable = 'precipitation', subbasin = '', hybas = 'country') : 

    if hybas == 'country': 
       output = agg_data(geom, ds, hybas, 'precipitation', time = '')
    else: 
        output = ''
    #     if subbasin != '': 
    #         Hybas = gpd.read_file("Other_data/hybas_" + continent + "_lev01-12_v1c/hybas_" + continent + "_lev0" + str(hybas) + "_v1c.shp")
    #         sub = Hybas[Hybas['HYBAS_ID'] == subbasin]
    #         output = agg_data(sub, ds, hybas, 'precipitation', time = '')

    #     else: 
    #         Hybas = gpd.read_file("Other_data/hybas_" + continent + "_lev01-12_v1c/hybas_" + continent + "_lev0" + str(hybas) + "_v1c.shp")
    #         subs = Hybas.sjoin(country, how = 'inner')
    #         subs_unique = subs['HYBAS_ID'].unique() 
    
    #         output = {}
    #         for s in subs_unique:
    #             print('sub #' + str(s))
    #             geom_hybas = subs[subs['HYBAS_ID'] == s]
    #             output[str(s)] = agg_data(geom_hybas, ds, hybas, 'precipitation', time = '')
        
    return output

def get_geom(iso_code): 
    world_filepath = gpd.datasets.get_path('naturalearth_lowres')
    world = gpd.read_file(world_filepath)
     
    country_bounds = world.loc[world['iso_a3'] == iso_code] # get country boundary
    geom = country_bounds.geometry   
    return geom

def clean_modis_lst(geom, file_header): ## need country boundary geometry
    ## need to have all MODIS files downloaded in directory
    urls = pd.read_csv("MOD_LST2/urls.txt", sep = '/', header = None).iloc[:,7:]
    urls.columns = ['time', 'name'] 
    
#    paths =  glob.glob('MOD_LST2/' + file_header) 

    agg_gdf = gpd.GeoDataFrame() ## initialize df to store final combined modis data
    new_df = gpd.GeoDataFrame() ## initialize df to store iteration of modis data
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

    for name in paths: ### will read and clean them all one at a time and then convert 
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
        
        if hybas == 'country': 
           new_var3 = agg_data(geom = geom, ds = ds, var = 'Snow_Cover_Monthly_CMG', 
                               hybas = hybas, time = time)
        # else: 
        #     if subbasin != '': 
        #         Hybas = gpd.read_file("Other_data/hybas_" + continent + "_lev01-12_v1c/hybas_" + continent + "_lev0" + str(hybas) + "_v1c.shp")
        #         sub = Hybas[Hybas['HYBAS_ID'] == subbasin]
        #         output = agg_data(sub, ds, hybas, time = time)

        #     else: 
        #         Hybas = gpd.read_file("Other_data/hybas_" + continent + "_lev01-12_v1c/hybas_" + continent + "_lev0" + str(hybas) + "_v1c.shp")
        #         subs = Hybas.sjoin(country, how = 'inner')
        #         subs_unique = subs['HYBAS_ID'].unique() 
        
        #         output = {}
        #         for s in subs_unique:
        #             print('sub #' + str(s))
        #             geom_hybas = subs[subs['HYBAS_ID'] == s]
        #             output[str(s)] = agg_data(geom_hybas, ds, hybas, time = time)
                    
        
#        year1 = year ## keep track of last year 
        new_df = pd.concat([new_df, new_var3])
        
        year1 = year
    
        ## data is already in monthly format, but want to get the mean for each 
        ## subbasin of interest 
    return new_df## this will be a geodataframe with geometry and will hold the whole modis timeseries       


def subset_modis(country_name, continent, subbasin = '', hybas = 3, variable = 'Snow_Cover_Monthly_CMG'):
    output = {} ## dictionary that will hold values for all of the subbasins

    #import rioxarray as rxr
    if variable == 'Snow_Cover_Monthly_CMG': ## MODIS SNOW
        file_header = 'MOD10CM*.hdf'
        new_var3, country = clean_modis_snow(country_name, file_header)      
        
    else: ## MODIS LST  
        file_header = 'MOD11B3*.hdf' 
        new_var3, country = clean_modis_lst(country_name, file_header)      
        
    # Hybas = gpd.read_file("../hybas_" + continent + "_lev01-12_v1c/hybas_" + continent + "_lev0" + str(hybas) + "_v1c.shp")
    
    # if subbasin != '': 
    #     sub = Hybas[Hybas['HYBAS_ID'] == subbasin][['HYBAS_ID', 'geometry']]        
    #     sub_avg = sub.sjoin(new_var3, how = 'inner')
    #     ## now should be able to groupby whilst preserving the subbasin geom
    #     sub_avg2 = sub_avg.dissolve(by = ['HYBAS_ID', 'time'], aggfunc = {variable: 'max'})
    

    # else: 
    #     subs = Hybas.sjoin(country, how = 'inner')
    #     subs_unique = subs['HYBAS_ID'].unique() 
    #     ## the subbasins that intersect with the country
           
    #     for s in subs_unique:
    #         print('sub #' + str(s))
    #         geom = subs[subs['HYBAS_ID'] == s][['HYBAS_ID', 'geometry']]
            
    #         sub_avg = geom.sjoin(new_var3, how = 'inner')
    #         ## now should be able to groupby whilst preserving the subbasin geom
    #         #sub_avg2 = sub_avg.dissolve(by = 'time', aggfunc = {variable: 'max'})
            
    #         ## if we use groupby instead of dissolve, we lose the geometry, but that is okay 
    #         ## because we know the subbasin label 
    #         sub_avg['month'] = sub_avg['time'].dt.month
    #         sub_avg['year'] = sub_avg['time'].dt.year
            
    #         sub_avg2 = sub_avg.groupby(['month', 'year']).max()


    #         print('sub #' + str(s) + ' max: ' + str(sub_avg2.max()))
            
    #         output[s] = sub_avg2
        
    country_avg = country.sjoin(new_var3, how = 'inner') 
    country_avg['month'] = country_avg['time'].dt.month
    country_avg['year'] = country_avg['time'].dt.year
    
    country_avg2 = country_avg.groupby(['month', 'year']).max()
    
    output['country_avg'] = country_avg2 
    return output


def agg_country(new_var3, country):
    country_avg = country.sjoin(new_var3, how = 'inner') 
    country_avg['month'] = country_avg['time'].dt.month
    country_avg['year'] = country_avg['time'].dt.year
    
    country_avg2 = country_avg.groupby(['month', 'year']).max()
    return country_avg2


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
    spri_mons = [3,4,5]
    summ_mons = [6,7,8]
    wint_mons = [12,1,2]
    fall_mons = [9,10,11]
    
    gdf['time'] = pd.to_datetime(gdf['time'])
    gdf['month'] = gdf.time.dt.month    
    gdf['year'] = gdf.time.dt.year    
        
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

    gdf_seas = summer.merge(fall, left_index = True, right_index = True)
    gdf_seas = gdf_seas.merge(spring, left_index = True, right_index = True)
    gdf_seas = gdf_seas.merge(winter, left_index = True, right_index = True)        
    gdf_seas.reset_index(inplace = True)

    ## also aggregate annually 
    gdf_ann = gdf.groupby('year').mean()
    
    grouped = gdf_ann.merge(gdf_seas, on = 'year')

    return grouped

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def r2_calc(predicted, actual): 
    RSS = ((predicted - actual)**2).sum()
    TSS = ((actual - actual.mean())**2).sum()
    r2 = 1 - (RSS/TSS)
    return r2

## input gdf should be a geodataframe with timestamps, at a seasonal 
## or annual frequency 
def regs(outcome, input_gdf, country, detrending = False, 
                    year_pred = '', plot = True, threshold = 0.6, test_threshold = .3):  
    count = 0
    old_r2 = -10 
    reg_top = ''
    regression_output = pd.DataFrame()    
    ## merge outcome and data of interest
    all_data = outcome.merge(input_gdf, on = 'year')
    
    ## and re-sort for x and y while ensuring we are not wasting time
    ## with regressions that are not useful
    x = all_data.drop(['outcome', 'year', 'month', \
                       'wint_month', 'summ_month', 'spri_month', 'fall_month'], axis = 1)
    x = sm.add_constant(x, has_constant = 'add')
    y = all_data['outcome']
    
    ## only want to save anything with r2 > 0.5 for train and test 
    ## and of course with solid p-values 
    ## also, not more than 5 inputs
    if len(x) > 2: 
        X_train, X_test, y_train, y_test = tts(x, y, test_size=.3, random_state=1)
    
        for subset in powerset(x.columns): 
            if 'const' in itertools.chain(subset):
    #            print('we good')
                if (len(subset) > 0) & (len(subset) < 5):
                    print(list(subset))
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
                        series = list(subset)
                        predicted_all = model.predict(x[list(subset)])
                        
                        ## also make sure to plot! 
                        if plot == True: 
                            count += 1
                            r2_all = r2_calc(predicted_all, y)
                            try: 
                                plot_reg(training, y_train, predicted, y_test, r2_all, country, count)
                            except: print("Plotting failed")                        
                        new = {'inputs':[series], 'r2': r2_all, 'pval':[pval], 'test_r2': pred_r2} 
                        new = pd.DataFrame(new)
                        regression_output = pd.concat([regression_output, new], axis = 0)
                        print('r2 all: ' + str(r2_all))
                        print('r2 old: ' + str(old_r2))
                        
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













