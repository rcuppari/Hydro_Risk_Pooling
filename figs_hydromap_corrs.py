# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:47:32 2023

@author: rcuppari
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
from cleaning_funcs import id_hydro_countries
import geopandas as gpd

###############################################################################

LCOE = 0.044 ## per kWh
use_tariffs = True

################## figure 1: correlations between countries ###################
gen_data2 = id_hydro_countries(0, capacity = False)
gen_data3 = gen_data2.T
gen_data3.columns = gen_data3.iloc[0,:]
gen_data3 = gen_data3.iloc[1:-4,:]

for c in gen_data3.columns: 
    gen_data3.loc[:,c] = pd.to_numeric(gen_data3.loc[:,c], errors = 'coerce')

mask = np.triu(np.ones_like(gen_data3.corr(), dtype=np.bool))
cut_off = 0.5  # only show cells with abs(correlation) at this value or less
corr = gen_data3.corr()
mask |= corr > cut_off
corr = corr[~mask]

wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
corr = corr.iloc[wanted_cols, wanted_rows]

annot = [[f"{val:.2f}"
          for val in row] for row in corr.to_numpy()]

## can also include electricity prices
tariffs = pd.read_csv("Other_data/elec_prices_global_petrol.csv").iloc[:-1,:]
tariffs['Tariff ($/kWh)'] = pd.to_numeric(tariffs['Tariff ($/kWh)'], errors = 'coerce')
tariffs['Tariff ($/MWh)'] = tariffs['Tariff ($/kWh)']  * 1000 ## convert to MWh

## need to match tariff to each column (this is an EXTREMELY rough measure 
## since there are trends over time/changes in tariffs)
## mostly, we are looking to figure out the absolute cost of a loss in hydro 
## considering how much hydro and the market (ish) value
val_hydro = pd.DataFrame(index = gen_data3.index)
for c in gen_data3.columns: ## c = country name

    if use_tariffs == True: 
        ## note: manually adjusted the name for Kyrgyzstan, DRC, Burma, North Korea, Tajikistan
        cost = tariffs[tariffs['Country Name'] == c]['Tariff ($/kWh)']
        
        try:
            data = gen_data3.loc[:,c] * pow(10,9) * cost.iloc[0] #* cost.iloc[0] * 1000 * 8760
            ## the 1000 is to get kWh into MWh and the 8760 is to get MW into MWh
            val_hydro = pd.concat([val_hydro, data], axis = 1)
        except: 
            print(f"{c} does not have tariff data")
    
    else: 
        ## gen_data3 is in BILLION kWh
        data = gen_data3.loc[:,c] * LCOE * 1*pow(10,9)
        val_hydro = pd.concat([val_hydro, data], axis = 1)
        
def get_geom(iso_code): 
    world_filepath = gpd.datasets.get_path('naturalearth_lowres')
    world = gpd.read_file(world_filepath)
     
    country_bounds = world.loc[world['iso_a3'] == iso_code] # get country boundary
    geom = country_bounds.geometry   
    return geom

## get iso code 
country_conversions = pd.read_csv("Other_data/country_conversions.csv")[['EIA Name', 'iso_a3']]

vals = pd.DataFrame(val_hydro.max(axis = 0))
vals = vals.merge(country_conversions, left_index = True, right_on = 'EIA Name')

gdf2 = gpd.GeoDataFrame()
for i in np.arange(0,len(vals)):
    iso = vals.iloc[i]['iso_a3']
    gdf1 = gpd.GeoDataFrame(columns = ['Country', 'total $/yr'], geometry = get_geom(iso))
    df = pd.DataFrame(vals.iloc[i,:]).T 
    
    gdf1['Country'] = df['EIA Name'].iloc[0]
    gdf1['total $/yr'] = df.iloc[0,0]
    gdf1['Country Code'] = df['iso_a3'].iloc[0]

    gdf2 = pd.concat([gdf1, gdf2])

pct_hydro = pd.read_csv("Other_data/wb_hydro%.csv")[['Country Name', 'Country Code', '2015']]
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['pop_est','iso_a3', 'gdp_md_est', 'geometry']]
world.columns = ['pop', 'Country Code', 'gdp', 'geometry']


gdp = pd.read_csv("Other_data/wb_gdp.csv", skiprows = 3).iloc[:,:-1]#[['Country Code', '2015']]

year_str = []
for year in np.arange(2000,2023): 
    year_str.append(str(year))
columns = ['Country Name', 'Country Code'] + year_str    

gdp = gdp[columns]
gdp['avg'] = 0
for i in np.arange(0, len(gdp)):
    gdp['avg'].iloc[i] = \
        pd.to_numeric(gdp.iloc[i, 4:], errors = 'coerce').mean()
gdp = gdp[['Country Code', 'avg']]

hydro_gdf = world.merge(gdf2[['Country Code', 'total $/yr']], how = 'outer')
hydro_gdf = hydro_gdf.merge(pct_hydro, on = 'Country Code')
hydro_gdf = hydro_gdf.merge(gdp, on = 'Country Code')

hydro_gdf['pct_gdp'] = (hydro_gdf['total $/yr']/hydro_gdf['avg'])*100

#cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['paleturquoise','darkblue'], N=256)
cmap = mpl.cm.YlGnBu(np.linspace(0,1,100))
cmap = mpl.colors.ListedColormap(cmap[20:,:-1])

fig, (ax1, ax2) = plt.subplots(ncols=1, nrows = 2, sharex=True, sharey=True)
hydro_gdf.plot(column='2015', ax = ax1,  
               missing_kwds={'color': 'grey', 'hatch': 'xxxxxx', 
                             'edgecolor':'lightgrey'},
            figsize = (25, 20), cmap = cmap, legend = True, 
            vmin = 0, vmax = 100, edgecolor = 'whitesmoke', linewidth = 1)
ax1.set_yticks([],[])
ax1.set_xticks([],[])

hydro_gdf.plot(ax = ax2, column = 'pct_gdp', cmap = cmap,
               missing_kwds={'color': 'grey', 'hatch': 'xxxxxx', 
                             'edgecolor':'lightgrey'}, legend = True, 
               vmin = 0, vmax = 15, edgecolor = 'whitesmoke', linewidth = 1)
ax2.set_title('Generation as Pct. GDP',fontsize=14)
ax2.set_xticks([],[])
ax2.set_yticks([],[])

###############################################################################
######################### HYDROPOWER CORRELATIONS #############################

plt.rc('xtick', labelsize = 14)  # x tick labels fontsize
plt.rc('ytick', labelsize = 14)  # y tick labels fontsize

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


gen_data['sum'] = gen_data.loc[:,'Albania':].mean(axis = 1)

corrs = pd.DataFrame(gen_data.corr().iloc[1:,-1])
corrs.reset_index(inplace = True)
corrs.columns = ['Country Name', 'Correlation']

gen_data2 = gen_data.dropna(axis = 1)
gen_data2 = gen_data2.iloc[:, 1:-1]

mask = np.triu(np.ones_like(gen_data2.corr()))
labels = gen_data2.columns.tolist()
plt.figure()
ax1 = sns.heatmap(gen_data2.corr(), mask = mask, 
            cmap = 'PuOr')
ax1.set_yticks(np.arange(len(labels)))
ax1.set_yticklabels(labels,fontsize=10)
ax1.set_xticks(np.arange(len(labels)))
ax1.set_xticklabels(labels,fontsize=10)


mask2 = np.triu(np.ones_like(gen_data.loc[:, 'Albania':].corr()))
corrs_use = gen_data.loc[:, 'Albania':].corr()
corrs_use.dropna(how='any', axis=0).dropna(how='any', axis=1)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['name','pop_est','iso_a3', 'gdp_md_est', 'geometry']]
world.columns = ['country', 'pop', 'Country Code', 'gdp', 'geometry']
## for some strange reason, France is listed as -99 instead of FRA... replace
world.iloc[43,2] = 'FRA'

pct_hydro = pd.read_csv("Other_data/wb_hydro%.csv")[['Country Name', 'Country Code', '2015']]










