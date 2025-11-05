# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 09:17:36 2025

@author: rcuppari
"""


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import mapping
from cleaning_funcs import write_dict
import numpy
from geo_ref_dams import find_basins_in_country
from cleaning_funcs2 import id_hydro_countries
import numpy as np
import matplotlib.colors as clr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from sklearn.model_selection import train_test_split as tts


from cleaning_funcs2 import group_data
from cleaning_funcs2 import index_vetting
from analysis_plotting_funcs_updated import predict_countries
from analysis_plotting_funcs_updated import predict_countries_mult
from shapley_loading import Shapley_calc
from wang_transform import wang_slim


print('starting!')
use_country_avg = False 
hybas = 'country'
pct_gen = 25
name = 'updated_1028'
## start by identifying countries of interest
## want to save the generation data just for the countries
## with hydro generating more than a specified % 
## use latest year of available data (2015)


cap_fac = True ## use capacity factor or raw capacity?
strike_val = 20 ## strike threshold 
payout_cap = 0.5 ## percent to set cap at
LCOE = 0.044 # LCOE is 0.044/kWh
cvar = True ## whether or not to cap payments
#name = 'cap_fac995_final' ## suffix for naming
use_tariffs = False ## use LCOE or not?


all_sig = pd.DataFrame(columns = ['Country Name', 'r2', 'inputs', 'coefs', 'df', 
                                  'pval', 'r2_train', 'test_r2'])  

## also create a validation df so that we can save metrics in an excel 
## for easy viewing :) 
validation_df = pd.DataFrame(index = ['MAE', 'NNSE',
                                      'white_test', 'breuschpagan_test', 
                                      'shap_wilks', 'dagostino_pearson', 
                                      'and_darling'])

## once we have the generation data, we need to iterate over each country 
## and design regressions to predict said generation over the available time period
## using AVHHR, MODIS, and IMERG inputs 
## we also want to make sure that we are accounting for any linear trends, so start
## by detrending 

gen_data = id_hydro_countries(pct_gen = pct_gen, year = 2015)


dict_country_grouped2 = {}
dict_4_grouped2 = {}

## to make it easily findable will also write it out
for country in gen_data['Country']: 
    #print(country)
    try: 
        dict_country_grouped2[country] = pd.read_csv((f"Results_all/Inputs_country_avg/{country}_input_data_new.csv"))
    except: 
        dict_country_grouped2[country] = pd.DataFrame(columns = [country])
    try:
        dict_4_grouped2[country] = pd.read_csv((f"Results_all/Inputs_4/{country}_input_data_new.csv"))    
    except: 
        dict_country_grouped2[country] = pd.DataFrame(columns = [country])
    
    
##########################################################################
################# tariffs 
tariffs = pd.read_csv("Other_data/elec_prices_global_petrol.csv").iloc[:-1,:]
tariffs['Tariff ($/kWh)'] = pd.to_numeric(tariffs['Tariff ($/kWh)'], errors = 'coerce')
tariffs['$/MWh'] = tariffs['Tariff ($/kWh)']  * 1000 ## convert to MWh


## first things first, let's see what the correlations are between countries
countries = id_hydro_countries(pct_gen = 25, year = 2015,
                                      cap_fac = True, capacity = False)

################# hydropower generation capacity
country_capacity = id_hydro_countries(pct_gen = 25, year = 2015, 
                                      cap_fac = True, capacity = True)
country_capacity = country_capacity.T
country_capacity.reset_index(inplace = True)
country_capacity.columns = country_capacity.iloc[0,:]
country_capacity = country_capacity.iloc[3:-5,:]
country_capacity = country_capacity.rename(columns = {'Country':'year'})
country_capacity = country_capacity.rename(columns = {'Country/area':'year'})

for c in country_capacity.columns[1:]:
    country_capacity[c] = pd.to_numeric(country_capacity[c], errors = 'coerce')
    country_capacity[c] = country_capacity.loc[23,c]

################# hydropower capacity factor data (or generation if cap_fac = False)
## NOTE: MANUALLY CHANGED BOLIVIA (PLURINATIONAL STATE OF) TO BOLIVIA
## ALSO United Republic of Tanzania (the) TO TANZANIA
gen_data = countries.T
gen_data.reset_index(inplace = True)
## drop the rows with the column names 
gen_data.columns = gen_data.iloc[0,:]
gen_data = gen_data.iloc[3:-4,:]
gen_data = gen_data.rename(columns = {'Country/area':'year'})
gen_data = gen_data.rename(columns = {'Country':'year'})

for c in gen_data.columns: 
    gen_data[c] = pd.to_numeric(gen_data[c], errors = 'coerce')

################################### regular ###################################
## as in, generation 
gen_data2 = id_hydro_countries(pct_gen = 25, year = 2015, 
                                      cap_fac = cap_fac)#, capacity = True)
gen_data2 = gen_data2.T
gen_data2.columns = gen_data2.iloc[0,:]
gen_data2 = gen_data2.iloc[3:-4,:]
gen_data2 = gen_data2.rename(columns = {'Country':'year'})
gen_data2 = gen_data2.rename(columns = {'Country/area':'year'})

for c in gen_data2.columns: 
    gen_data2[c] = pd.to_numeric(gen_data2[c], errors = 'coerce') 
gen_data2.index = pd.to_numeric(gen_data2.index, errors = 'coerce') 

###############################################################################
gen_data['sum'] = gen_data.loc[:,'Albania':].sum(axis = 1)
    
corrs = pd.DataFrame(gen_data.corr().loc['Albania':,'sum'])
corrs.reset_index(inplace = True)

corrs.columns = ['Country Name', 'Correlation']

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['name','pop_est','iso_a3', 'gdp_md_est', 'geometry']]
world.columns = ['country', 'pop', 'Country Code', 'gdp', 'geometry']
## for some strange reason, France is listed as -99 instead of FRA... replace
world.iloc[43,2] = 'FRA'

pct_hydro = pd.read_csv("Other_data/wb_hydro%.csv")[['Country Name', 'Country Code', '2015']]

## trigger payouts based on capacity factor, NOT generation
## THEN translate the "payout" in cap fac equivalent to generation and then multiply by $$
cap_fac2 = gen_data
cap_fac_pay = cap_fac2.copy()
payouts = pd.DataFrame(index = cap_fac_pay.index, columns = cap_fac_pay.columns[1:])
strikes = pd.DataFrame(index = cap_fac2.columns[1:-1], columns = ['strike'])
cvars = pd.DataFrame(index = cap_fac2.columns[1:-1], columns = ['cf_cvar'])



######### calculate strikes and theoretical payouts with given strikes ########
for i, country in enumerate(cap_fac2.columns[1:-1]): 
#    print(country)
    data = cap_fac2[country]
    strikes.loc[country, 'strike'] = np.percentile(cap_fac2[country], strike_val)
    mean = cap_fac2[country].mean()
    cvars.loc[country, 'cf_cvar'] = data[data < strikes.loc[country, 'strike'] ].mean()
    
 
    for r in np.arange(0, len(cap_fac2)): 
        
        if cap_fac2[country].iloc[r] < strikes.loc[country, 'strike'] : 
            cap_fac_pay[country].iloc[r] = mean - cap_fac2[country].iloc[r]
            
            ## can also cap that with the CVAR/VAR/whatever choose... 
            if cvar == True: 
                if cap_fac_pay[country].iloc[r] > cvars.loc[country, 'cf_cvar']: 
                    cap_fac_pay[country].iloc[r] = cvars.loc[country, 'cf_cvar']
            
            country_capacity[country].iloc[r] = pd.to_numeric(country_capacity[country].iloc[r])
            
            if use_tariffs == True:
                
                try:
                    cost = tariffs[tariffs['Country Name'] == country]['$/MWh']
                    payouts.iloc[r, i] = cap_fac_pay[country].iloc[r] * \
                        country_capacity[country].iloc[r] * 8760 * cost.values[0]
                except:                         
                    cost = tariffs['$/MWh'].mean()
                    payouts.iloc[r, i] = cap_fac_pay[country].iloc[r] * \
                        country_capacity[country].iloc[r] * 8760 * cost
            else: 
                payouts.iloc[r, i] = cap_fac_pay[country].iloc[r] * \
                    country_capacity[country].iloc[r] * 8760 * 1000 * LCOE # capacity is in MW, so need MWh --> kWh 
        else: 
            cap_fac_pay.iloc[r, i] = 0 
            payouts.iloc[r, i] = 0 
        
cap_fac_pay_corrs = payouts.corr()

## okay, let's look at reserves needed to meet 95th% of payout...
## distros
var_val = 90
num_part = len(payouts.columns)
dens_height = 1.2*pow(10,-9)
ann_mean = payouts.sum(axis = 1).mean() 

individ_sums = pd.DataFrame(index = [0], columns = payouts.columns)
for c in payouts.columns: 
    individ_sums[c] = np.percentile(payouts[c], var_val)

pool_var = np.percentile(payouts.sum(axis = 1), var_val)

savings = 100*(individ_sums.iloc[0,:].sum() - pool_var)/individ_sums.iloc[0,:].sum()
print(f"Reserve savings: {round(savings)}%")
#strikes.to_csv("Results_all/strikes2_" + name + ".csv")

####################### INDEX VIABILITY/RELIABILITY ###########################

## turns out some of these fail when imported directly from the "automatic"
## algorithm, probably based on the order of operations I used. 
## HOWEVER, also turns out the regressions using those 
## (or some of those) inputs do still work so let's verify them individually

## manually importing the relevant data (for some it's just subbasin, 
## for others it's also with the country average), using a few different
## train/test split combos, and then outputting the required descriptive data
## use a helper function to output the criteria to keep the code cleaner
## but manually train/test split 

# Pakistan
c = 'Pakistan'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
comb = comb.merge(dict_4_grouped2[c], on = 'year')
comb.dropna(axis = 1, inplace= True)
print(comb.corr()[c].sort_values())

y = comb[c]
x = comb[['summ_precip_y', 'LST_Day_CMG_x']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 6)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

###############################################################################
## Angola
c = 'Angola'
comb = gen_data[['year', c]].merge(dict_4_grouped2[c])
comb.corr()[c]

y = comb[c]
x = comb[['fall_LST_Day_CMG']]
x = sm.add_constant(x)

X_train, X_test, y_train, y_test = tts(x, y, test_size = .2, \
                        random_state = 3) 
model = sm.OLS(y_train, X_train).fit()
training = model.predict(X_train)
print(model.summary())

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 3)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

###############################################################################
# # Norway 
c = 'Norway'
comb = gen_data[['year', c]].merge(dict_4_grouped2[c])
comb = comb.merge(dict_country_grouped2[c], on = 'year')
comb.dropna(axis = 1, inplace = True)

y = comb[c]
x = comb[['LST_Day_CMG_y', 'LST_Day_CMG_x']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 2)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])


model.conf_int()

###############################################################################

## Panama
c = 'Panama'
comb = gen_data[['year', c]].merge(dict_4_grouped2[c])
comb.dropna(axis = 1, inplace = True)
comb.corr()[c].sort_values()

y = comb[c]
x = comb[['LST_Day_CMG']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 4)

validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])


model.conf_int()

# ###############################################################################
## Paraguay
c = 'Paraguay'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
comb = comb.merge(dict_4_grouped2[c], on = 'year')
comb.dropna(axis = 1, inplace = True)
comb.corr()[c].sort_values()

y = comb[c]
x = comb[['fall_precip_x', 'summ_LST_Day_CMG_y']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 1)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])


model.conf_int()

###############################################################################
## Albania
c = 'Albania'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
#comb = comb.merge(dict_country_grouped2[c], on = 'year')

y = comb[c]
x = comb[['summ_LST_Day_CMG', 'fall_precip']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 20)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])


model.conf_int()

###############################################################################
## Chile 
c = 'Chile'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
comb = comb.merge(dict_4_grouped2[c], on = 'year')
comb.corr()[c]

y = comb[c]
x = comb[['precip_y', 'Snow_C_y', 'Snow_C_x']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 3)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

###############################################################################
## Costa Rica
c = 'Costa Rica'
comb = gen_data[['year', c]].merge(dict_4_grouped2[c])
comb.corr()[c]

y = comb[c]
x = comb[['summ_LST_Day_CMG', 'summ_CMG 0.']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 3)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])


model.conf_int()

###############################################################################
## Croatia
c = 'Croatia'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
comb.corr()[c]

y = comb[c]
x = comb[['precip', 'fall_CMG 0.']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 3)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

###############################################################################
## Ecuador
c = 'Ecuador'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
comb = comb.merge(dict_4_grouped2[c], on = 'year')
comb.corr()[c]

y = comb[c]
x = comb[['fall_LST_Day_CMG_x', 'fall_LST_Day_CMG_y']]
x = sm.add_constant(x)

## use helper function to keep code cleaner 
new, valid = index_vetting(c, x, y, random_state = 3)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

###############################################################################
## Georgia 
c = 'Georgia'
comb = gen_data[['year', c]].merge(dict_4_grouped2[c])

y = comb[c]
x = comb[['LST_Day_CMG', 'CMG 0.']]
x = sm.add_constant(x)

new, valid = index_vetting(c, x, y, random_state = 3)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

###############################################################################
## Latvia
c = 'Latvia'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
comb.corr()[c]

y = comb[c]
x = comb[['LST_Day_CMG']]
x = sm.add_constant(x)

new, valid = index_vetting(c, x, y, random_state = 3)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])


model.conf_int()

# ###############################################################################
## Suriname
c = 'Suriname'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
comb.corr()[c].sort_values()

y = comb[c]
x = comb[['fall_LST_Day_CMG']]
x = sm.add_constant(x)

new, valid = index_vetting(c, x, y, random_state = 14)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

###############################################################################
## Tajikistan'
c = 'Tajikistan'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])

y = comb[c]
x = comb[['summ_LST_Day_CMG']]
x = sm.add_constant(x)

new, valid = index_vetting(c, x, y, random_state = 5)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

########################################################################
## Uruguay 
c = 'Uruguay'
comb = gen_data[['year', c]].merge(dict_country_grouped2[c])
#comb = comb.merge(dict_country_grouped2[c], on = 'year')
comb.dropna(inplace = True, axis = 1)
comb.corr()[c].sort_values()

y = comb[c]
x = comb[['LST_Day_CMG', 'fall_CMG 0.']]
x = sm.add_constant(x)

new, valid = index_vetting(c, x, y, random_state = 19)
validation_df[f"{c}"] = valid.values


all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

########################################################################
## Zambia
c = 'Zambia'
comb = gen_data[['year', c]].merge(dict_4_grouped2[c])

y = comb[c]
x = comb[['precip', 'spri_LST_Day_CMG']]
x = sm.add_constant(x)

new, valid = index_vetting(c, x, y, random_state = 6)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])

model.conf_int()

# ################################################################################
# ## Zimbabwe
c = 'Zimbabwe'
comb = gen_data[['year', c]].merge(dict_4_grouped2[c])

evi_zim = pd.read_csv("Messy Data/evi_Zimbabwe_country.csv")
evi_zim = group_data(evi_zim)
comb = comb.merge(evi_zim, on = 'year')
comb.dropna(axis = 1, inplace = True)
comb.corr()[c].sort_values()

y = comb[c]
x = comb[['spri_LST_Day_CMG', 'summ_LST_Day_CMG']]
x = sm.add_constant(x)

new, valid = index_vetting(c, x, y, random_state = 1)
validation_df[f"{c}"] = valid.values

all_sig = all_sig[all_sig['Country Name'] != c]
all_sig = pd.concat([all_sig, pd.DataFrame(new)])


model.conf_int()

########################################################################
## make validation df easier to read
validation_df_t = validation_df.T

# ###################### export results if so desired
# reg_table = all_sig[['Country Name', 'inputs', 'coefs', 'r2', 'r2_train', 'test_r2','df']]
# reg_table = reg_table[reg_table['Country Name'].isin(all_sig.index)]
# reg_table.to_csv(f"Results_all/reg_table_{name}.csv")
# validation_df.to_csv(f"Results_all/validation_metrics_{name}.csv")

# ########################################################################
# ## pulling the inputs for each country from the index 

# all_sig.reset_index(inplace = True)
# all_sig = all_sig.iloc[:,1:]

# all_sig = all_sig[~all_sig['Country Name'].isin(['Pakistan', 'Gabon'])]
## weirdly world is missing Norway's country code...... no idea why. 
## manually update
world['Country Code'].iloc[21] = 'NOR'

hydro_gdf = pd.merge(world, pct_hydro, on = 'Country Code')
merged_gdf = hydro_gdf.merge(all_sig, on = 'Country Name', how = 'outer')

cmap = clr.LinearSegmentedColormap.from_list('custom', ['lightskyblue','darkblue'], N=256)

fig, ax = plt.subplots(figsize=(25, 20))  
merged_gdf.plot(ax = ax, column = 'r2', 
                missing_kwds={'color':'lightgrey', 'edgecolor':'whitesmoke'}, 
                cmap = cmap, linewidth = 1)
ax.set_xticks([-150, -100, -50, 0, 50, 100, 150],
              [-150, -100, -50, 0, 50, 100, 150], fontsize = 14)
ax.set_yticks([-60, -40, -20, 0, 20, 40, 60, 80],
              [-60, -40, -20, 0, 20, 40, 60, 80], fontsize = 14)
#ax.set_title('Significant Regression Results', fontsize=16)
# Create an axis for the colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=.5)

# Add a colorbar to the plot with aspect parameter to adjust width
cbar = plt.colorbar(ax.collections[0], cax=cax, orientation='horizontal', aspect=10)
cbar.ax.tick_params(labelsize = 16)
cbar.set_label("$r^2$ value", fontsize = 16)

plt.savefig("Figures/sig_reg_" + name + ".png")

predictions = pd.DataFrame()
predictions['year'] = np.arange(2000, 2024)

for r, country in enumerate(all_sig['Country Name']):   
    print(f"{country}, {r}")
    country_gen = gen_data[['year', country]]
    country_gen['year'] = pd.to_numeric(country_gen['year'], errors = 'coerce')
    country_gen[country] = pd.to_numeric(country_gen[country], errors = 'coerce')
    
    hybas = all_sig['df'].iloc[r]
    
    reg_inputs = all_sig['inputs'].iloc[r]
    
## NOT SURE WHY SOME COEFS ARE READING IN AS STRINGS???
    if type(all_sig['coefs'].iloc[r]) is str:
        all_sig['coefs'].iloc[r] = all_sig['coefs'].iloc[r].replace("[", "")
        all_sig['coefs'].iloc[r] = all_sig['coefs'].iloc[r].replace("]", "")
        all_sig['coefs'].iloc[r] = all_sig['coefs'].iloc[r].split(',')

    if type(all_sig['inputs'].iloc[r]) is str:
        all_sig['inputs'].iloc[r] = all_sig['inputs'].iloc[r].replace("[", "")
        all_sig['inputs'].iloc[r] = all_sig['inputs'].iloc[r].replace("]", "")
        all_sig['inputs'].iloc[r] = all_sig['inputs'].iloc[r].replace("'", "")
        all_sig['inputs'].iloc[r] = all_sig['inputs'].iloc[r].split(',')    
    
    if type(hybas) is str:   
        hybas = hybas.replace("[", "")
        hybas = hybas.replace("]", "")
        hybas = hybas.replace("'", "")
        
        hybas = hybas.split(',')

    if type(reg_inputs) is list: 
        ''
    else:
        print('not a list')
        reg_inputs = reg_inputs.replace(" ", "")
        reg_inputs = reg_inputs.replace("[", "")
        reg_inputs = reg_inputs.replace("]", "")
        reg_inputs = reg_inputs.replace("'", "")
    
        reg_inputs = reg_inputs.split(",")
        
    try:
        reg_parts = all_sig['coefs'].iloc[r]
        lines = reg_parts.split('\n')
        variables = []
        coefs = []
        
        for line in lines[:-1]:
            print(line)
            # Split the line by whitespace
            parts = re.split(r'\s{2,}', line.strip(), maxsplit=1)
            variables.append(parts[0])
            coefs.append(parts[1])
            
    except: 
        coefs = all_sig['coefs'].iloc[r]
        variables = all_sig['inputs'].iloc[r]
        

    if isinstance(hybas, list): 
        if len(hybas) > 1:
            new_predictions = predict_countries_mult(country, hybas, country_gen, reg_inputs,
                                                       coefs, variables)        
        else: 
            new_predictions = predict_countries(country, hybas, country_gen, reg_inputs,
                                                       coefs, variables)

    else: 
        new_predictions = predict_countries(country, hybas, country_gen, reg_inputs,
                                                   coefs, variables)

    predictions = predictions.merge(new_predictions, on = 'year')


## compare
actual = gen_data[predictions.columns]
actual.set_index('year', inplace = True)
predictions.set_index('year', inplace = True)

# ###############################################################################
## payouts from predictions
## ADDING COST OF COMPENSATORY NG (for comparability with the grant)
ng_opex = 0#25.89 ## $/MWh from the EIA

country_capacity.set_index('year', inplace = True)
cap_fac2.set_index('year', inplace = True)
country_capacity.index = pd.to_numeric(country_capacity.index, errors = 'coerce')

pred_pay = pd.DataFrame(index = predictions.index, 
                        columns = predictions.columns)

## also what happens once we have predicted? How does that end up impacting 
## final performance? 
gen_val = pd.DataFrame(index = predictions.index, 
                        columns = predictions.columns)
predictions2 = predictions.copy() ## a prediction for gen, not cap fac

for i, country in enumerate(pred_pay.columns): 
    print(country)
    data = pd.DataFrame(cap_fac2[country])
    
    country_capacity[country] = pd.to_numeric(country_capacity[country])
    cap_data = pd.DataFrame(country_capacity[country])
    cap_data.set_index(pd.to_numeric(country_capacity.index), inplace = True)

    merge = data.merge(cap_data, left_index = True, right_index = True)    
    merge.columns = ['cap_fac', 'capacity']

    strike_cf = np.percentile(data, strike_val)
    mean = data.mean()
    payout_cap2 = np.percentile(data, payout_cap)
    below_strike = data[data < payout_cap2]
    cvar2 = (mean - np.percentile(data, payout_cap))[0]
    country_pred = predictions[country]
    
    if use_tariffs == True:
        try: 
            cost = tariffs[tariffs['Country Name'] == country]['$/MWh']
            gen_val[country] = merge['cap_fac'] * merge['capacity'] \
            * 8760 * cost.values[0]
        except: 
            cost = tariffs['$/MWh'].mean()
            gen_val[country] = merge['cap_fac'] * merge['capacity'] \
            * 8760 * cost

    else: 
        gen_val[country] = merge['cap_fac'] * merge['capacity'] \
        * 8760 * 1000 * LCOE

    for r, year in enumerate(pred_pay.index):         
        
        if country_pred.loc[year] < strike_cf: 
            #print(year)
            pred_pay1 = (mean - country_pred.loc[year])[0]
            
            ## can also cap that  
            if cvar == True: 
                print('maxed')
                if pred_pay1 > cvar2: 
                    pred_pay1 = cvar2
            
            ## capacity is in MW, so need MWh --> kWh 
            ## include payment for penalty for NG here
            pred_pay.loc[year, country] = pred_pay1 * \
                merge.loc[year, 'capacity'] * 8760 * \
                LCOE * 1000 + \
                (strike_cf - country_pred.loc[year])* \
                merge.loc[year, 'capacity'] * 8760 * ng_opex
               
        else: 
            pred_pay.loc[year, country] = 0 
        
        predictions2.loc[year, country] = predictions2.loc[year, country] * \
            merge.loc[year, 'capacity'] 
    pred_pay[country] = pd.to_numeric(pred_pay[country], errors = 'coerce')
    
all_sig.to_csv("Results_all/all_sig_" + name + ".csv")
pred_pay.to_csv("Results_all/predicted_payouts_" + name + ".csv")

# ########################## calculate Shapley value ############################
## exclude countries with no payouts
pred_pay = pred_pay.loc[:,pred_pay.any()]
cap_fac3 = cap_fac2[pred_pay.columns]#all_sig['Country Name']]
cap_fac3.dropna(axis = 0, inplace = True)

## now find group and individual premiums
df_payouts = pd.DataFrame(pred_pay.sum(axis = 1))
df_payouts.columns = ['payout']
df_payouts = pd.DataFrame(pred_pay.sum(axis = 1))
df_payouts.columns = ['payout']
pool = wang_slim(df_payouts, contract = 'put', premOnly = True)

gen_diff = pd.DataFrame(index = cap_fac3.columns, 
                        columns = ['avg_pay', 'Shapley', 'individ'])

## find every combination of participants
from math import factorial as fac
import itertools

combs = []
weights = []
length = len(pred_pay.columns)
for L in range(1,length+1):
    print(L)
    for subset in itertools.combinations(pred_pay, L):
        combs.append(list(subset))
        ## calculation for permutations of a given sample given it will be of 
        ## length L (i.e., using up all countries in entry)
        perms = fac(L)#/fac(L - L) 
        weights.append(perms)
        
## and the pooled values for each combination
group_vals = []
for i in combs:  
    #print(i)
    group_vals.append(pred_pay[i].sum(axis = 1))

for country in pred_pay.columns: 
    print(country)
    marg_contrib, summed = Shapley_calc(country, combs, payouts = pred_pay,     
                                    pricing = 'wang', code = 'wang_slim')
    gen_diff.loc[country, 'Shapley'] = summed[0]
    
    df_payouts = pd.DataFrame(pred_pay[country])
    df_payouts.columns = ['payout']
    gen_diff.loc[country, 'individ'] = wang_slim(df_payouts, contract = 'put', 
                                             premOnly = True)
    gen_diff.loc[country, 'avg_pay'] = df_payouts.mean()[0]
    

## IF WE JUST USE SHAPLEY CALC TO GET LOADING PER CAP THEN WE NEED TO ADD BACK THE 
## AVG PAY TO GET THE FULL PREMIUM

# gen_diff['loading_individ'] = ((gen_diff['individ'] + gen_diff['avg_pay'])/gen_diff['avg_pay'])*100
# gen_diff['loading_pool'] = ((gen_diff['Shapley'] + gen_diff['avg_pay'])/gen_diff['avg_pay'])*100
# gen_diff['pct_diff'] = 100*(gen_diff['individ'] - gen_diff['Shapley'])/gen_diff['individ']
# gen_diff['pct_diff_load'] = 100*(gen_diff['loading_individ'] - \
#                                  gen_diff['loading_pool'])/gen_diff['loading_individ']
gen_diff['loading_individ'] = ((gen_diff['individ'] + gen_diff['avg_pay'])/gen_diff['avg_pay'])*100
gen_diff['loading_pool'] = ((gen_diff['Shapley'])/gen_diff['avg_pay'])*100
gen_diff['pct_diff'] = 100*(gen_diff['individ'] - gen_diff['Shapley'])/gen_diff['individ']
gen_diff['pct_diff_load'] = 100*(gen_diff['loading_individ'] - \
                                 gen_diff['loading_pool'])/gen_diff['loading_individ']

    
portfolio = gen_diff.Shapley.sum() - gen_diff.avg_pay.sum()
individ_sum_contracts = gen_diff.individ.sum()
print(f"% diff with portfolio: {round((gen_diff.individ.sum() - portfolio)/gen_diff.individ.sum()*100,2):,}")
print(f"Portfolio: {round(portfolio):,}")
print(f"Individ Sum: {round(gen_diff.individ.sum()):,}")
load_pool = gen_diff.Shapley.sum()#gen_diff.avg_pay.sum() + gen_diff.Shapley.sum()
load_individ = gen_diff.avg_pay.sum() + gen_diff.individ.sum()
#print(f"% diff loading: {round((load_pool - load_individ)/load_individ*100,2):,}")
print(f"% diff loading: {round((load_pool - load_individ)/load_individ*100,2):,}")
 
## POOL 
df_payouts = pd.DataFrame(pred_pay.sum(axis = 1))
df_payouts.columns = ['payout']
pool_calc = wang_slim(df_payouts, contract = 'put', 
                                         premOnly = True)
print(f"Pool prem - shapley sum: {pool_calc - portfolio}")

###############################################################################
###############################################################################
## impact of index insurance INDIVIDUALLY 
## and then in POOL 
###############################################################################
###############################################################################
country_capacity.index = pd.to_numeric(country_capacity.index, errors = 'coerce')
country_capacity = country_capacity[country_capacity.index.isin(pred_pay.index)]
#country_capacity.reset_index(inplace = True)

individ_ins_val = pd.DataFrame(index = pred_pay.index, 
                        columns = pred_pay.columns)
ins_val = pd.DataFrame(index = pred_pay.index, 
                        columns = pred_pay.columns)


gen_data2 = id_hydro_countries(pct_gen = 25, year = 2015, 
                                      cap_fac = False)#, capacity = True)
gen_data2 = gen_data2.T
#country_gen.reset_index(inplace = True)
gen_data2.columns = gen_data2.iloc[0,:]
gen_data2 = gen_data2.iloc[3:-4,:]
gen_data2 = gen_data2.rename(columns = {'Country':'year'})
gen_data2 = gen_data2.rename(columns = {'Country/area':'year'})

for c in gen_data2.columns: 
    gen_data2[c] = pd.to_numeric(gen_data2[c], errors = 'coerce') 
gen_data2.index = pd.to_numeric(gen_data2.index, errors = 'coerce') 
gen_data2 = gen_data2.loc[pred_pay.index, pred_pay.columns]

if use_tariffs == True:
    gen_val = gen_data2.copy()
    for c in gen_data2.columns:
        cost = tariffs[tariffs['Country Name'] == country]['Tariff ($/kWh)']
        gen_val[c] = gen_val[c] * pow(10,9) * cost.values[0]         
else:
    gen_val = gen_data2 * pow(10,9) * LCOE ## billion kwh to $$ 
    
    
for country in pred_pay.columns: 
    individ_ins_val[country] = gen_val[country] + pred_pay[country] + \
        gen_diff.loc[country, 'individ']
    ins_val[country] = gen_val[country] + pred_pay[country] + \
        gen_diff.loc[country, 'Shapley']

plt.rcParams.update({
    'axes.titlesize': 14,  # Set the font size for axes titles
    'axes.labelsize': 14,  # Set the font size for x and y labels
    'xtick.labelsize': 12,  # Set the font size for x tick labels
    'ytick.labelsize': 12,  # Set the font size for y tick labels
    'legend.fontsize': 16,  # Set the font size for the legend
    'legend.title_fontsize': 12  # Set the font size for the legend title
})
# Determine the number of rows and columns for the subplot grid
num_cols = len(individ_ins_val.columns)
num_plots_per_row = int(np.ceil(np.sqrt(num_cols + 1)))  # +1 for the legend box
num_rows = int(np.ceil((num_cols + 1) / num_plots_per_row))

# Create the subplots including an extra one for the legend
fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(15, 10))
# Flatten the axes array for easy iteration
axes = axes.flatten()
# List to store plot handles for the legend
line_labels = ['Uninsured Outcome', 'Insured Outcome, Individual', "Insured Outcome, Pool"]
line_handles = []

individ_ins_val = individ_ins_val[sorted(individ_ins_val)]
for i, country in enumerate(individ_ins_val.columns):     
    r2 = round(all_sig[all_sig['Country Name'] == country]['r2'].iloc[0],2)
    loc = np.percentile(ins_val[country], 100)*1.05
    
    l1, = axes[i].plot(gen_val[country], label = "Uninsured Outcome", color = 'orange', 
             linewidth = 1.5, alpha = 0.8)
    l2, = axes[i].plot(individ_ins_val[country], label = "Insured Outcome, Individual", color = 'lightblue', 
             linewidth = 1.5, alpha = 0.8, linestyle = 'dashed')
    l3, = axes[i].plot(ins_val[country], label = "Insured Outcome, Pool", color = 'purple', 
             linewidth = 1.5, alpha = 0.6, linestyle = 'dashdot')
    #axes[i].set_ylim(ymin = gen_val[country].min()*.85, ymax = loc*1.1)
    axes[i].set_xlabel("Year") 
    axes[i].set_ylabel("Hydro Value ($)")
    axes[i].annotate(f"r2: {r2}", \
                     xy = (2000, loc))
    
    if i == 6:
       line_handles.extend([l1, l2, l3])
    axes[i].set_title(f"{country}")

legend_ax = axes[len(individ_ins_val.columns)]
legend_ax.axis('off')  # Turn off the axis

# Create the legend in the extra subplot
legend_ax.legend(handles=line_handles, labels=line_labels, \
                 loc='center', \
                 frameon = False)

# Remove any remaining empty subplots
for j in range(len(individ_ins_val.columns) + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#plt.savefig(f"Results_all/Comparisons/grid_all_ins" + name + ".png")
gen_diff.to_csv(f"Results_all/shapley_vals_actual_{name}.csv")

names = sorted(gen_diff.index)
corr_data = gen_data.loc[:, names]
# plt.figure()
# mask2 = np.triu(np.ones_like(corr_data.corr()))
# sns.heatmap(corr_data.corr(), mask = mask2, \
#             annot_kws={"fontsize":10}, cmap = 'PuOr', vmin=1, vmax=-1)

corrs = corr_data.corr()
corrs[corrs != 1].mean().mean()

corrs_all = cap_fac2.corr()
corrs_all[corrs_all != 1].mean().mean()


# ###########################
    
portfolio = gen_diff.Shapley.sum()
individ_sum_contracts = gen_diff.individ.sum()
print(f"% diff with portfolio: {round((gen_diff.individ.sum() - portfolio)/gen_diff.individ.sum()*100,2):,}")
print(f"Portfolio: {round(portfolio):,}")
print(f"Sum of shaps (double check): {round(gen_diff.Shapley.sum()):,}")




































































































