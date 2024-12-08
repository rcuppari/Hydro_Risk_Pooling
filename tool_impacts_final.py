# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:48:30 2024

@author: rcuppari
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

## import custom funcs
#from analysis_plotting_funcs import id_hydro_countries
from cleaning_funcs2 import id_hydro_countries

cap_fac = True ## use capacity factor or capacity?
strike_val = 20 ## strike as percent capacity factor (i.e., VaR)
strike_val2 = 5 ## strike as percent capacity factor (i.e., VaR)
payout_cap = .5 ## maximum payout as percent capacity factor (i.e., VaR)
LCOE = 0.044 # LCOE is 0.044/kWh, used to translate generation to $$
use_tariffs = False ## choice to use LCOE or observed tariff data if available
cvar = True
name = 'cap_fac995_final' ## suffix for reading in files
## potential yield of  "risk-free" investment (low risk, high liquidity) and
oc_rate = 6.03 ## as percent

    
pred_pay = pd.read_csv(f"Results_all/predicted_payouts_{name}.csv")#.iloc[:,1:]
pred_pay.set_index('year', inplace = True)
all_sig = pd.read_csv(f"Results_all/all_sig_{name}.csv")
gen_diff = pd.read_csv("Results_all/shapley_vals_actual_" + name + ".csv")
gen_diff.set_index(gen_diff.iloc[:,0], inplace = True)
gen_diff = gen_diff.iloc[:,1:]
gen_diff['Shapley_prem'] = gen_diff['Shapley'] - gen_diff['avg_pay']
gen_diff['individ_loading'] = gen_diff['individ'] + gen_diff['avg_pay']
gen_diff['pct_diff_load'] = (gen_diff['individ_loading'] - gen_diff['Shapley'])/ \
                            gen_diff['individ_loading']
cvs = pred_pay.std()/pred_pay.mean()
all_sig = all_sig[all_sig['Country Name'].isin(gen_diff.index)]
pred_pay = pred_pay.loc[:, gen_diff.index]

################# tariffs 
tariffs = pd.read_csv("Other_data/elec_prices_global_petrol.csv").iloc[:-1,:]
tariffs['Tariff ($/kWh)'] = pd.to_numeric(tariffs['Tariff ($/kWh)'], errors = 'coerce')
tariffs['$/MWh'] = tariffs['Tariff ($/kWh)']  * 1000 ## convert to MWh

################# hydropower generation capacity
country_capacity = id_hydro_countries(pct_gen = 25, year = 2015, 
                                      cap_fac = True, capacity = True)
country_capacity = country_capacity.T
country_capacity.reset_index(inplace = True)
country_capacity.columns = country_capacity.iloc[0,:]
country_capacity = country_capacity.iloc[3:-5,:]
country_capacity = country_capacity.rename(columns = {'Country':'year'})
country_capacity = country_capacity.rename(columns = {'Country/area':'year'})

cap_data = id_hydro_countries(pct_gen = 25, year = 2015, 
                                      cap_fac = True, capacity = False)
cap_data = cap_data.T
cap_data.reset_index(inplace = True)
cap_data.columns = cap_data.iloc[0,:]
cap_data = cap_data.iloc[3:-4,:]
cap_data = cap_data.rename(columns = {'Country':'year'})
cap_data = cap_data.rename(columns = {'Country/area':'year'})
cap_data.set_index('year', inplace = True)
cap_data.index = pd.to_numeric(cap_data.index)

country_capacity2 = country_capacity.copy()
## using capacity from the last year available for each country
for c in pred_pay.columns:
    country_capacity[c] = pd.to_numeric(country_capacity[c], errors = 'coerce')
    country_capacity2[c] = pd.to_numeric(country_capacity2[c], errors = 'coerce')
    cap_data[c] = pd.to_numeric(cap_data[c], errors = 'coerce')
    #country_capacity[c] = country_capacity[c].mean()
    country_capacity[c] = country_capacity[c].iloc[-1]
    
################################### regular ###################################
## as in, generation 
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

###############################################################################
###############################################################################
## pulled directly from the all sig shapley code 
## impact of index insurance INDIVIDUALLY 
## and then in POOL 
###############################################################################
###############################################################################
country_capacity.year = pd.to_numeric(country_capacity.year, errors = 'coerce')
country_capacity = country_capacity[country_capacity.year.isin(pred_pay.index)]
country_capacity.set_index('year', inplace = True)

individ_ins_val = pd.DataFrame(index = pred_pay.index, 
                        columns = pred_pay.columns)
ins_val = pd.DataFrame(index = pred_pay.index, 
                        columns = pred_pay.columns)


################# hydropower capacity factor data (or generation if cap_fac = False)
countries = id_hydro_countries(pct_gen = 25, year = 2015,
                                      cap_fac = True, capacity = False)

gen_data = countries.T
gen_data.reset_index(inplace = True)
## drop the rows with the column names 
gen_data.columns = gen_data.iloc[0,:]
gen_data = gen_data.iloc[3:-4,:]
gen_data = gen_data.rename(columns = {'Country/area':'year'})
gen_data = gen_data.rename(columns = {'Country':'year'})

for c in gen_data.columns: 
    gen_data[c] = pd.to_numeric(gen_data[c], errors = 'coerce')
gen_data.set_index('year', inplace = True)

gen_val = country_capacity.loc[pred_pay.index, pred_pay.columns] * \
          gen_data.loc[pred_pay.index, pred_pay.columns] * \
              8760 * 1000 #* LCOE

if use_tariffs == True:
   for c in gen_val.columns: 
       cost = tariffs[tariffs['Country Name'] == c]['$/MWh']
       print(f"Cost in {c}: {cost}/MWh")
       print()
       gen_val[c] = gen_val[c] * cost.values
else:
    gen_val = gen_val * LCOE

for country in pred_pay.columns: 
    individ_ins_val[country] = gen_val[country] + pred_pay[country] + \
        gen_diff.loc[country, 'individ']
        
    ins_val[country] = gen_val[country] + pred_pay[country] + \
        gen_diff.loc[country, 'Shapley_prem']
    
shap = gen_diff.copy()

labelsize = 14
plt.rcParams.update({
    'axes.titlesize': labelsize+2,  # Set the font size for axes titles
    'axes.labelsize': labelsize,  # Set the font size for x and y labels
    'xtick.labelsize': labelsize-2,  # Set the font size for x tick labels
    'ytick.labelsize': labelsize-2,  # Set the font size for y tick labels
    'legend.fontsize': labelsize-2,  # Set the font size for the legend
})

############################################### 
## for each country find the opp cost 
## how many reserves would need to hold to cover an equivalent amount of losses,
###############################################################################
## extract loading on individual and pooled contracts 
shap_load = pd.DataFrame(index = [0], columns = gen_diff.index)
individ_load = pd.DataFrame(index = [0], columns = gen_diff.index)

for country in gen_diff.index: 
#    shap_prem = gen_diff.loc[country, 'Shapley']
    individ_prem = gen_diff.loc[country, 'individ']
    avg_pay = gen_diff.loc[country, 'avg_pay']
    
    ## take the absolute value of the two to get the loading expressed as a 
    ## positive value. Donked up how I listed the two 
    shap_load.loc[:,country] = abs(gen_diff.loc[country, 'Shapley'])
    individ_load.loc[:,country] = abs(individ_prem + avg_pay)

###############################################################################
## calculate opportunity cost of reserves up to max insurance payout
opp_costs_res = pd.DataFrame(index = gen_diff.index, 
                             columns = ['opp_cost', 'opp_cost_pct', 
                                        'VaR'])

## want the opportunity costs for each country over the 17 years or so 
for country in gen_diff.index:     
    ## just compare based on payout cap for insurance
    cap_data3 = pd.DataFrame(country_capacity2[country])
    cap_data3.set_index(pd.to_numeric(country_capacity2.year), inplace = True)
    
    merge = pd.DataFrame(cap_data[country]).merge(cap_data3, 
                                        left_index = True, right_index = True)    
    merge.columns = ['cap_fac', 'capacity']

    mean = merge['cap_fac'].mean()
    payout_cap_cf = np.percentile(merge['cap_fac'], payout_cap)
    add_val_reserves = (mean - payout_cap_cf) * merge.iloc[-1,1] \
                        * 8760 * 1000 * LCOE
#    print(f"Reserves for VaR: {add_val_reserves/pow(10,6)}")
#    print()
    opp_costs_res.loc[country,'VaR'] = add_val_reserves
    
    if add_val_reserves < 0: 
        print(f"WOOPS! Check on {country}")
    
    opp_costs_res.loc[country,'opp_cost'] = add_val_reserves * \
                            (oc_rate/100)
    opp_costs_res.loc[country,'opp_cost_pct'] = \
            opp_costs_res.loc[country,'opp_cost'] / gen_diff.loc[country, 'avg_pay']

opp_cost_pool = (opp_costs_res.VaR.sum()) * (oc_rate/100)

##################################################################
labelsize = 20
plt.rcParams.update({
    'axes.titlesize': labelsize+2,  # Set the font size for axes titles
    'axes.labelsize': labelsize,  # Set the font size for x and y labels
    'xtick.labelsize': labelsize-2,  # Set the font size for x tick labels
    'ytick.labelsize': labelsize-2,  # Set the font size for y tick labels
    'legend.fontsize': labelsize-2,  # Set the font size for the legend
})

red_pool_res = (shap_load.sum(axis = 1)/opp_costs_res['opp_cost'].sum())[0]
red_individ_res = (individ_load.sum(axis = 1)/opp_costs_res['opp_cost'].sum())[0]

print(f"Pool prem as % reserve opp cost: {red_pool_res}")
print(f"Individ prem as % reserves opp cost: {red_individ_res}")
print(f"Pool savings (wrt res opp cost): {(1-red_pool_res)*100}")
print(f"Individ prem savings (wrt res opp cost): {(1-red_individ_res)*100}")

individ_res_savings = 100*(opp_costs_res.loc[:,'opp_cost']-individ_load.T.iloc[:,0])/opp_costs_res.loc[:,'opp_cost']

val_floor = ins_val.min() - gen_val.min() 

######################### TABLE DATA 
strike_val2 = 1
comp_table = pd.DataFrame(index = pred_pay.columns, 
                          columns = ['sq_mean', 'ins_mean',  
                                      'sq_99VaR','ins_99VaR',
                                      'sq_80VaR','ins_80VaR', 
                                      'sq_cvar', 'ins_cvar', 
                                      'sq_min', 'ins_min'])
comp_table.index = sorted(comp_table.index)
for c in pred_pay.columns: 
    comp_table.loc[c, 'sq_mean'] = gen_val[c].mean()
    comp_table.loc[c, 'ins_mean'] = ins_val[c].mean()
    
    comp_table.loc[c, 'sq_99VaR'] = np.percentile(gen_val[c], 1)
    comp_table.loc[c, 'ins_99VaR']  = np.percentile(ins_val[c], 1)

    comp_table.loc[c, 'sq_80VaR'] = np.percentile(gen_val[c], strike_val)
    comp_table.loc[c, 'ins_80VaR']  = np.percentile(ins_val[c], strike_val)

    comp_table.loc[c, 'sq_min'] = gen_val[c].min()
    comp_table.loc[c, 'ins_min']  = ins_val[c].min()
    
    comp_table.loc[c, 'sq_cvar'] = \
        (np.percentile(gen_val[c], strike_val) - \
         np.percentile(gen_val[c], strike_val2)).mean()
    comp_table.loc[c, 'ins_cvar']  = \
        (np.percentile(ins_val[c], strike_val) - \
         np.percentile(ins_val[c], strike_val2)).mean()
    
    
for c in comp_table.columns: 
    comp_table[c] = pd.to_numeric(comp_table[c])

comp_table['change_mean'] = comp_table['ins_mean'] - comp_table['sq_mean'] 

comp_table['change_cvar'] = comp_table['ins_cvar'] - comp_table['sq_cvar'] 
comp_table['change_80VaR'] = comp_table['ins_80VaR'] - comp_table['sq_80VaR'] 
comp_table['change_99VaR'] = comp_table['ins_99VaR'] - comp_table['sq_99VaR'] 
comp_table['change_min'] = comp_table['ins_min'] - comp_table['sq_min'] 

comp_table['change_cvar_pct'] = comp_table['change_cvar']/comp_table['sq_cvar'] 
comp_table['change_80VaR_pct'] = comp_table['change_80VaR']/comp_table['sq_80VaR'] 
comp_table['change_99VaR_pct'] = comp_table['change_99VaR']/comp_table['sq_99VaR'] 
comp_table['change_min_pct'] = comp_table['change_min']/comp_table['sq_min'] 


names = sorted(gen_diff.index)
corr_data = gen_data.loc[:, names]
corrs = corr_data.corr()
corrs[corrs != 1].mean().mean()
corrs[corrs != 1].max()
corrs.min()

############################### last figure ###################################
sum_costs = opp_costs_res.merge(individ_load.T, left_index = True, 
                                right_index = True)
sum_costs = sum_costs.merge(shap_load.T, left_index = True, 
                                right_index = True)
sum_costs['sum'] = sum_costs[['opp_cost', '0_x', '0_y']].max(axis = 1)
sum_costs = sum_costs['sum'].sort_values()

#opp_costs_res2 = opp_costs_res.iloc[:,0]
#opp_costs_res2 = opp_costs_res2.sort_values()

opp_costs_res2 = opp_costs_res.loc[sum_costs.index, 'opp_cost']
individ_load = individ_load.loc[:,opp_costs_res2.index]

shap_load = shap_load.loc[:,opp_costs_res2.index]

names = opp_costs_res2.index

width = 1
fig, (ax1, ax2) = plt.subplots(2,1,
                        gridspec_kw={'height_ratios': [1, 2]})
plt.subplots_adjust(left=0.17, right=0.93, top=0.98, bottom=0.12)

ax1.barh(-width/2, opp_costs_res2.sum(), 
         color='darkorange', hatch='\\\\\\', 
         edgecolor='white', linewidth=3, height=width/2.5)

ax1.barh(0, individ_load.iloc[0,:].sum(), 
         color='olivedrab', hatch='xxx', 
         edgecolor='white', linewidth=3, height=width/2.5)

ax1.barh(width/2, shap_load.iloc[0,:].sum(), 
         color='purple', edgecolor='white', linewidth=3, height=width/2.5)

ax1.set_xlabel("$M")
ax1.annotate("A", xy = (1.63*pow(10,8), 0.52), weight = "bold", fontsize = 35)
plt.legend(frameon = False, loc = 'upper left')
ax1.set_yticks([-width/2, 0, width/2], ['Opportunity Cost, \nReserves', 
            'Contract Loading, \nIndividual Insurance', 
            'Contract Loading, \nInsurance Pooling'])
ax1.set_xticks([0,  0.4*pow(10,8), 0.8*pow(10,8),
                1.2*pow(10,8), 1.6*pow(10,8)],
           ['0', '40', '80', '120', '160'])

width = 0.4
offset = width/2
ax2.bar(np.arange(0, len(names)) + offset, opp_costs_res2,
        color='darkorange', hatch='\\\\\\',
        edgecolor = 'white', linewidth = 3, 
        label = 'Opportunity Cost, Reserves', width = width)
    
ax2.bar(np.arange(0, len(names)) - offset, individ_load.iloc[0,:], 
        color = 'olivedrab', hatch = 'xxx', edgecolor = 'white', 
        linewidth = 3, label = 'Contract Loading, \nIndividual Insurance', 
        width = width)

ax2.bar(np.arange(0, len(names)) - offset, shap_load.iloc[0,:], 
        color = 'purple', edgecolor = 'white', linewidth = 3, 
        label = 'Contract Loading, \nInsurance Pooling', width = width)

         
ax2.set_xticks(np.arange(0, len(names)), names, rotation = 45)

ax2.set_ylabel("$M")
ax2.annotate("B", (15.65, 28.5*pow(10,6)), weight = "bold", fontsize = 35)
plt.legend(frameon = False, loc = 'upper left')

ax2.set_yticks([0, 10*pow(10,6), 20*pow(10,6), 30*pow(10,6)],
           ['0', '10', '20', '30'])

