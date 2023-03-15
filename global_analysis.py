# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:13:15 2023

@author: rcuppari
"""

##############################################################################
##################  scripts for my grand global takeover  ####################
##############################################################################

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import mapping
from cleaning_funcs import write_dict
import numpy

print('starting!')

## start by identifying countries of interest
## want to save the generation data just for the countries
## with hydro generating more than a specified % 
## use latest year of available data (2015)
   
def id_hydro_countries(pct_gen = 90, year = 2015):
    wb_hydro = pd.read_csv("Other_data/wb_hydro%.csv")
    gen_data = pd.read_csv("Other_data/eia_hydropower_data.csv")
    country_conversion = pd.read_csv("Other_data/country_conversions.csv")
    
    countries = wb_hydro[wb_hydro['2015'] > pct_gen][['Country Name', 'Country Code']] 
    countries = countries.merge(country_conversion, on = 'Country Name')    
    countries = countries.rename(columns={'EIA Name':'Country'})
    
    ## remove spaces from gen data country names to make consistent
    gen_data.iloc[:,0] = gen_data.iloc[:,0].str.lstrip()
    gen_data = gen_data.rename(columns = {'hydroelectricity net generation (billion kWh)': 'Country'})

    ## keep generation for countries with high hydro 
    gen_data2 = gen_data[gen_data.iloc[:,0].isin(countries['Country'])]

    gen_data3 = gen_data2.merge(countries, on = 'Country')
#    gen_data3 = gen_data3.iloc[:, 1:-3]

    ## previously ID'ed which country names are inconsistent to create 
    ## a country translation csv 
#    bad_index = gen_data['Country'].isin(wb_hydro['Country Name'])
#    bad_names = gen_data.loc[~bad_index]
    
    return gen_data3

## once we have the generation data, we need to iterate over each country 
## and design regressions to predict said generation over the available time period
## using AVHHR, MODIS, and IMERG inputs 
## we also want to make sure that we are accounting for any linear trends, so start
## by detrending 

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
#    print(est_year2.summary())
#    print(est_year2.rsquared)

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


## clean any input data and clip to country of interest 
## using pre-existing scripts (thanks old self)
from cleaning_funcs import subset_imerg
from cleaning_funcs import clean_modis_lst
from cleaning_funcs import clean_modis_snow
from cleaning_funcs import get_geom
from cleaning_funcs import group_data
from cleaning_funcs import regs

def get_dfs(country, country_geom, ds, hybas = 'country'):
    ## need to import 
    ## 1) MODIS LST data (temp)
    ## 2) IMERG data (precip)
    ## 3) MODIS snow cover data 
    
    ## MODIS LST 
    modis_df = clean_modis_lst(country_geom, file_header = 'MOD11C3*.hdf')
#    modis_df = 'place holder'   
    ## IMERG 
    imerg_df = subset_imerg(country, country_geom, ds, hybas)
    
    ## MODIS SNOW
    snow_df = clean_modis_snow(country_geom, file_header = 'MOD10CM*.hdf')
    
    # else: ## for when we start splitting the data into basin-level 
    #     print('do_something')
    
    return modis_df, imerg_df, snow_df


## final for loop will use all of these in order
## start by getting countries of interest

gen_data = id_hydro_countries(pct_gen = 30, year = 2015)

##     import xarray 
import xarray 
import rioxarray 

ds= xarray.open_mfdataset(paths = 'IMERG/imerge_file*.nc4', combine  = 'by_coords')
ds.rio.set_spatial_dims(x_dim = 'lon', y_dim = 'lat', inplace = True)
ds.rio.write_crs("epsg:4326", inplace = True)

## then loop over all of them and run the regression algorithm
results = {}
results_top = {}

results_det = {}
results_top_det = {}
for country in gen_data['Country Name']: 
    ## start by detrending if necessary 
    print(country)
    df = gen_data[gen_data['Country Name'] == country]
    
    ## REMEMBER: if detrending == True, will need to retrend in final step
    det_data, clean_df, detrending, year_pred = remove_lin_trend(df)
          
    ## baby steps: start by using country level data, just as a proof of concept
    ## use the iso_a3 code to retrieve country geom, just to be standard 
    iso_code = df['iso_a3'].iloc[0]
    country_geom = get_geom(iso_code)

    ## clean & clip data 
    lst, imerg, snow = get_dfs(country, country_geom, ds, hybas = 'country')

    ## coordinates are for some reason flipped... 
    import shapely
    snow['geometry'] = gpd.GeoSeries(snow['geometry']).map(lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon))

    ## need to combine into one gdf
    ## because right now just doing by country, don't need to worry about
    ## space. BUT for subbasins need to figure this out...
    input_gdf = imerg.merge(snow, on = 'time')
    input_gdf = input_gdf.merge(lst, left_index = True, right_on = 'time')
    
    input_gdf.reset_index(inplace = True)
    
    ## make sure to save for each country the top performing algorithm 
    gdf_grouped = group_data(input_gdf)
    
    if detrending == True: 
        reg_det, reg_top_det = regs(det_data, gdf_grouped, country, year_pred = year_pred, detrending = True, 
                                     threshold = 0.5)
        results_det[country] = reg_det
        results_top_det[country] = reg_top_det

    reg_all, reg_top = regs(clean_df, gdf_grouped, country, plot = True, threshold = 0.5)

    results[country] = reg_all
    results_top[country] = reg_top

    write_dict(results_top, str(country) + "_gen_top_results")
    write_dict(results_top_det, str(country) + "_gen_top_results_det")

write_dict(results, "gen_results") 
write_dict(results_det, "gen_results_det") 

