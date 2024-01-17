# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:29:57 2023

@author: rcuppari
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os 

## TWO ideas here
## 1) get the big basin that covers a country (transboundary) -- 
##    i.e., watershed that intersects with country
## 2) find smaller scale subbasins that cover the area with the big dams
##    this will also mean later intersecting country with subbasin 
##    to use in regression 
## this is a little more complicated than I first imagined :( sad

## okay, GRanD might be our best bet -- indicates whether hydro
## is a main use, and while this is an imperfect measure of 
## a big, important dam, it is a good start for cross-referencing
## with hydrologic units, since otherwise there may be too many 
## dams to consider

def max_area_attr_join(hybas, country_geom, relevant_hybas):  
    ## thanks StackExchange! https://gis.stackexchange.com/questions/273495/geopandas-how-to-spatially-join-based-on-maximally-overlapping-feature
    #spatial overlay
    country_geom = gpd.GeoDataFrame(geometry = country_geom.geometry)
    country_geom.set_crs('EPSG:4326', inplace = True)
    relevant_hybas.set_crs('EPSG:4326', inplace = True)
    
#    intersection = gpd.sjoin(relevant_hybas, country_geom, how = 'right',
#                             predicate = 'intersects')
    
    s = gpd.overlay(country_geom, relevant_hybas, how = 'intersection') 
    s['area_ov'] = s.geometry.area
    
    gb = s.groupby('HYBAS_ID')[['area_ov']].max()
    gb.reset_index(inplace = True)

    ## retrieve the one with maximum area 
    ## figure out why gb.min() works and not gb.max() #CONFUSED :(
    max_hybas = int(gb.min()['HYBAS_ID'])
    hybas_geom = relevant_hybas[relevant_hybas['HYBAS_ID'] == max_hybas]    
    
#    plot = country_geom.plot()
#    hybas_geom.plot(ax = plot, color = 'pink', alpha = .8)
    return hybas_geom

def max_dam_overlap(hybas, country_geom, relevant_hybas): 
    
        ## step 1: subset dams to in country only 
        grand = gpd.read_file("GRanD/GRanD_reservoirs_v1_3.shp")
        grand_hydro = grand[grand['USE_ELEC'] == 'Main']
        
        country_geom = gpd.GeoDataFrame(geometry = country_geom.geometry)
        
        points_in_hybas = gpd.tools.sjoin(grand_hydro, relevant_hybas, \
                                          predicate = 'within')
            
        points_in_hybas = gpd.overlay(country_geom, points_in_hybas, \
                                      how = 'intersection') 

        ## step 2: find all subbasins that intersect with dams and choose 
        ## the one with the highest amount of dams intersecting
            
        points_in_hybas['count'] = 1            
        country_hybas = points_in_hybas.groupby('HYBAS_ID').sum('count')
        country_hybas.reset_index(inplace = True)
        country_hybas = points_in_hybas.sort_values('count', ascending = False)
        
        max_hybas = country_hybas.iloc[0,:]

        max_dam_loc = relevant_hybas[relevant_hybas['HYBAS_ID'] == max_hybas['HYBAS_ID']]
        
        return max_dam_loc

def all_overlaps(country_geom, relevant_hybas): 
    ## step 1: subset dams to in country only 
    grand = gpd.read_file("GRanD/GRanD_reservoirs_v1_3.shp")
    grand_hydro = grand[grand['USE_ELEC'] == 'Main']

    ## looking to find all of the hybas units within a country and format them such that 
    ## each one can be an independent variable     
    country_geom = gpd.GeoDataFrame(geometry = country_geom.geometry)
    
    all_overlaps = gpd.overlay(country_geom, relevant_hybas, \
                                  how = 'intersection') 

    points_in_hybas = gpd.tools.sjoin(grand_hydro, relevant_hybas, \
                                      predicate = 'within')
        
    points_in_hybas = gpd.overlay(country_geom, points_in_hybas, \
                                  how = 'intersection') 

    ## step 2: find all subbasins that intersect with dams and choose 
    ## the one with the highest amount of dams intersecting
        
    points_in_hybas['count'] = 1            
    country_hybas = points_in_hybas.groupby('HYBAS_ID').sum('count')
    country_hybas.reset_index(inplace = True)
    country_hybas = points_in_hybas.sort_values('count', ascending = False)
    
    hybas_with_dams = country_hybas[country_hybas['count'] != 0]
    hybas_with_dams2 = hybas_with_dams['HYBAS_ID'].unique()
    
    all_overlaps = relevant_hybas[relevant_hybas['HYBAS_ID'].isin(hybas_with_dams2)]
    
    
    return all_overlaps 

def find_basins_in_country(country_geom, hybas, hybas2 = 4): 
    ## read in hybas units -- think about what size is appropriate -- 
    ## will likely have to experiment
    ## IMPORTANT: need to return only the basins/subbasins relevant
    ## for the specific country, NOT all of them 

    if (hybas == 'country') | (hybas == 'watersheds'): 
        ## step 1: call hybas
        hybas_af = gpd.read_file("Other_data/hybas_af_lev01-12_v1c/hybas_af_lev0" + str(hybas2) + "_v1c.shp")
        hybas_sa = gpd.read_file("Other_data/hybas_sa_lev01-12_v1c/hybas_sa_lev0" + str(hybas2) + "_v1c.shp")
        hybas_na = gpd.read_file("Other_data/hybas_na_lev01-12_v1c/hybas_na_lev0" + str(hybas2) + "_v1c.shp")
        hybas_eu = gpd.read_file("Other_data/hybas_eu_lev01-12_v1c/hybas_eu_lev0" + str(hybas2) + "_v1c.shp")
        hybas_as = gpd.read_file("Other_data/hybas_as_lev01-12_v1c/hybas_as_lev0" + str(hybas2) + "_v1c.shp")
        hybas_ar = gpd.read_file("Other_data/hybas_ar_lev01-12_v1c/hybas_ar_lev0" + str(hybas2) + "_v1c.shp")
        hybas_au = gpd.read_file("Other_data/hybas_au_lev01-12_v1c/hybas_au_lev0" + str(hybas2) + "_v1c.shp")
        hybas_gr = gpd.read_file("Other_data/hybas_gr_lev01-12_v1c/hybas_gr_lev0" + str(hybas2) + "_v1c.shp")
        hybas_si = gpd.read_file("Other_data/hybas_si_lev01-12_v1c/hybas_si_lev0" + str(hybas2) + "_v1c.shp")
        
    else:
        ## step 1: call hybas
        hybas_af = gpd.read_file("Other_data/hybas_af_lev01-12_v1c/hybas_af_lev0" + str(hybas) + "_v1c.shp")
        hybas_sa = gpd.read_file("Other_data/hybas_sa_lev01-12_v1c/hybas_sa_lev0" + str(hybas) + "_v1c.shp")
        hybas_na = gpd.read_file("Other_data/hybas_na_lev01-12_v1c/hybas_na_lev0" + str(hybas) + "_v1c.shp")
        hybas_eu = gpd.read_file("Other_data/hybas_eu_lev01-12_v1c/hybas_eu_lev0" + str(hybas) + "_v1c.shp")
        hybas_as = gpd.read_file("Other_data/hybas_as_lev01-12_v1c/hybas_as_lev0" + str(hybas) + "_v1c.shp")
        hybas_ar = gpd.read_file("Other_data/hybas_ar_lev01-12_v1c/hybas_ar_lev0" + str(hybas) + "_v1c.shp")
        hybas_au = gpd.read_file("Other_data/hybas_au_lev01-12_v1c/hybas_au_lev0" + str(hybas) + "_v1c.shp")
        hybas_gr = gpd.read_file("Other_data/hybas_gr_lev01-12_v1c/hybas_gr_lev0" + str(hybas) + "_v1c.shp")
        hybas_si = gpd.read_file("Other_data/hybas_si_lev01-12_v1c/hybas_si_lev0" + str(hybas) + "_v1c.shp")
        
    dfs = [hybas_af, hybas_sa, hybas_na, hybas_eu, hybas_as, hybas_ar, hybas_au, hybas_gr, hybas_si]
    
    ## which hybas continental dataset to use? 
    for i in dfs: 
        union = i.unary_union
        if country_geom.intersects(union).iloc[0] == True: 
            #print('contained!')
            relevant_hybas = i
    
    ## now which basins *within* the relevant hybas file intersect with our country? 
        
    #show_me = intersection.plot()
    #country_geom.plot(ax = show_me, color = 'pink', alpha = .7)

    if hybas == 'country':
        print('need to choose what has the greatest overlap')
        best_basin = max_area_attr_join(hybas, country_geom, relevant_hybas)        
        
    if hybas == 'watersheds': 
        ## want to use all possible hydrologic units that exist within a country, 
        ## and have dams within them, as independent variables
        best_basin = all_overlaps(country_geom, relevant_hybas)
        
    else: 
        best_basin = max_dam_overlap(hybas, country_geom, relevant_hybas)
        print('choosing from all subbasins!')        
            
    return best_basin
              






