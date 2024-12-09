# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:44:27 2024

@author: rcuppari
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
## import custom funcs
from cleaning_funcs2 import id_hydro_countries

###############################################################################
## this script evaluates the opportunity cost of holding reserves relative 
## to purchasing insurance (either individually or as part of a pool) 
## using two different sets of returns
## in this script, the two sets are the average of 10-year Treasury bonds
## between 2000-2024 and 1981-1999 and the average of 3-month Treasury bills 
## across the same time periods 
###############################################################################
cap_fac = True ## use capacity factor or capacity?
strike_val = 20 ## strike as percent capacity factor (i.e., VaR)
strike_val2 = 5 ## strike as percent capacity factor (i.e., VaR)
payout_cap = 0.5 ## maximum payout as percent capacity factor (i.e., VaR)
LCOE = 0.044 # LCOE is 0.044/kWh, used to translate generation to $$
use_tariffs = False ## choice to use LCOE or observed tariff data if available
name = 'cap_fac995_final' ## suffix for reading in files
alt_rate99 = 8.5#8.48 ## rate of alternative investment (low risk, lower liquidity) as % 1981- 1999
alt_rate24 = 3.28 ## rate of alternative investment (low risk, lower liquidity) as % 2000 - 2024

############### read in results files 
pred_pay = pd.read_csv(f"Results_all/predicted_payouts_{name}.csv")#.iloc[:,1:]
pred_pay.set_index('year', inplace = True)
all_sig = pd.read_csv(f"Results_all/all_sig_{name}.csv")
strikes = pd.read_csv(f"Results_all/strikes2_{name}.csv")
strikes.set_index(strikes.iloc[:,0], inplace = True)
strikes = strikes.iloc[:,1:]
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

country_capacity2 = country_capacity.copy()
## using capacity from the last year available for each country (2020)
for c in pred_pay.columns:
    country_capacity[c] = pd.to_numeric(country_capacity[c], errors = 'coerce')
    country_capacity2[c] = pd.to_numeric(country_capacity2[c], errors = 'coerce')
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
              8760 * 1000 * LCOE

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
## extract loading on individual and pooled contracts 

shap_load = pd.DataFrame(index = [0], columns = gen_diff.index)
individ_load = pd.DataFrame(index = [0], columns = gen_diff.index)

for country in gen_diff.index: 
    shap_prem = gen_diff.loc[country, 'Shapley_prem']
    individ_prem = gen_diff.loc[country, 'individ']
    avg_pay = gen_diff.loc[country, 'avg_pay']
    
    ## take the absolute value of the two to get the loading expressed as a 
    ## positive value
    shap_load.loc[:,country] = abs(shap_prem + avg_pay)
    individ_load.loc[:,country] = abs(individ_prem + avg_pay)
    
## for each country find the opp cost 
## how many reserves would need to hold to cover an equivalent amount of losses,
## i.e., the addition $$ 80% and 99%, since we can assume that 
## up to the 80% is going to be covered anyway 
strikes2 = strikes[strikes.index.isin(gen_diff.index)]

strikes2['99%'] = 0
strikes2['80%'] = 0
strikes2['$99%'] = 0
strikes2['$80%'] = 0

for country in strikes2.index: 
    data = gen_data[country]
    pct_99 = np.percentile(data, payout_cap)
    pct_80 = np.percentile(data, 20)
    mean =   data.mean()

    strikes2.loc[country, '99%'] = mean - pct_99   
    strikes2.loc[country, '80%'] = mean - pct_80

    if use_tariffs == True:
        cost = tariffs[tariffs['Country Name'] == country]['$/MWh']
        print(f"Cost in {c}: {cost}/MWh")
        print()

        strikes2.loc[country, '$99%'] = strikes2.loc[country, '99%'] * \
            int(country_capacity.loc[2020, country]) * cost.values * 8760
        strikes2.loc[country, '$80%'] = strikes2.loc[country, '80%'] * \
            int(country_capacity.loc[2020, country]) * cost.values * 8760
        
    else: 
        strikes2.loc[country, '$99%'] = strikes2.loc[country, '99%'] * \
            int(country_capacity.loc[2020, country]) * LCOE * 1000 * 8760
        strikes2.loc[country, '$80%'] = strikes2.loc[country, '80%'] * \
            int(country_capacity.loc[2020, country]) * LCOE * 1000 * 8760
            
###############################################################################
############################### 1981 - 1999 ###################################
############################################################################### 
opp_costs_res99 = pd.DataFrame(index = [0], columns = gen_diff.index)

strikes2['difference'] = strikes2['$99%'] - strikes2['$80%']
## want the opportunity costs for each country over the 17 years or so 
for country in gen_diff.index: 
   
    cap_data3 = pd.DataFrame(country_capacity[country])
    cap_data3.set_index(pd.to_numeric(country_capacity.index), inplace = True)
    
    merge = pd.DataFrame(gen_data[country]).merge(cap_data3, 
                                        left_index = True, right_index = True)    
    merge.columns = ['cap_fac', 'capacity']

    mean = merge['cap_fac'].mean()
    payout_cap_cf = np.percentile(merge['cap_fac'], payout_cap)
    add_val_reserves = (mean - payout_cap_cf) * merge.iloc[-1,1] \
                        * 8760 * 1000 * LCOE
    print(f"other: {add_val_reserves/pow(10,6)}")
    print()
 #   opp_costs_res99.loc[country,'VaR'] = add_val_reserves
        
    if add_val_reserves < 0: 
        print(f"WOOPS! Check on {country}")
    
    opp_costs_res99.loc[:,country] = add_val_reserves * alt_rate99 / 100


add_val_res_pool99 = np.percentile(pred_pay.sum(axis = 1), 100-payout_cap)  

################################################## we can do this on average... 

labelsize = 20
plt.rcParams.update({
    'axes.titlesize': labelsize+2,  # Set the font size for axes titles
    'axes.labelsize': labelsize,  # Set the font size for x and y labels
    'xtick.labelsize': labelsize-2,  # Set the font size for x tick labels
    'ytick.labelsize': labelsize-2,  # Set the font size for y tick labels
    'legend.fontsize': labelsize-2,  # Set the font size for the legend
})

red_pool_res99 = shap_load.mean(axis = 1)/opp_costs_res99.mean(axis = 1)
red_individ_res99 = individ_load.mean(axis = 1)/opp_costs_res99.mean(axis = 1)
print(f"Pool prem as % reserves: {red_pool_res99}")
print(f"Individ prem as % reserves: {red_individ_res99}")

red_pool_res99 = shap_load.iloc[0,:].mean()/opp_costs_res99.iloc[0,:].mean()
red_individ_res99 = individ_load.iloc[0,:].mean()/opp_costs_res99.iloc[0,:].mean()
print(f"Pool prem as % reserves: {red_pool_res99*100}")
print(f"Individ prem as % reserves: {red_individ_res99*100}")
print(f"Pool savings: {(1-red_pool_res99)*100}")
print(f"Individ prem savings: {(1-red_individ_res99)*100}")

########################### 2000 - 2024 data ##################################
opp_costs_shap24 = pd.DataFrame(index = [0], 
                                columns = gen_diff.index)
opp_costs_individ24 = pd.DataFrame(index = [0], 
                                   columns = gen_diff.index)
opp_costs_res24 = pd.DataFrame(index = [0], 
                               columns = gen_diff.index)

## want the opportunity costs for each country over the 17 years or so 
for country in gen_diff.index: 
   
    cap_data3 = pd.DataFrame(country_capacity[country])
    cap_data3.set_index(pd.to_numeric(country_capacity.index), inplace = True)
    
    merge = pd.DataFrame(gen_data[country]).merge(cap_data3, 
                                        left_index = True, right_index = True)    
    merge.columns = ['cap_fac', 'capacity']

    mean = merge['cap_fac'].mean()
    payout_cap_cf = np.percentile(merge['cap_fac'], payout_cap)
    add_val_reserves = (mean - payout_cap_cf) * merge.iloc[-1,1] \
                        * 8760 * 1000 * LCOE
    print(f"other: {add_val_reserves/pow(10,6)}")
    print()
#    opp_costs_res24.loc[country,'VaR'] = add_val_reserves
        
    if add_val_reserves < 0: 
        print(f"WOOPS! Check on {country}")

    if add_val_reserves < 0: 
        print(f"WOOPS! Check on {country}")
    
    opp_costs_res24.loc[:,country] = add_val_reserves * alt_rate24 / 100    


add_val_res_pool24 = np.percentile(pred_pay.sum(axis = 1), 100-payout_cap)

################################################## we can do this on average... 

red_pool_res24 = shap_load.mean(axis = 1)/opp_costs_res24.mean(axis = 1)
red_individ_res24 = individ_load.mean(axis = 1)/opp_costs_res24.mean(axis = 1)
print(f"Pool prem as % reserves: {red_pool_res24}")
print(f"Individ prem as % reserves: {red_individ_res24}")

red_pool_res24 = shap_load.iloc[0,:].mean()/opp_costs_res24.iloc[0,:].mean()
red_individ_res24 = individ_load.iloc[0,:].mean()/opp_costs_res24.iloc[0,:].mean()
print(f"Pool prem as % reserves: {red_pool_res24*100}")
print(f"Individ prem as % reserves: {red_individ_res24*100}")
print(f"Pool savings: {(1-red_pool_res24)*100}")
print(f"Individ prem savings: {(1-red_individ_res24)*100}")

############################### last figure ###################################
max_costs99 = opp_costs_res99.T.merge(individ_load.T, left_index = True, 
                                right_index = True)
max_costs99 = max_costs99.merge(shap_load.T, left_index = True, 
                                right_index = True)
max_costs99['sum'] = max_costs99[['0_x', '0_y', 0]].max(axis = 1)
max_costs99 = max_costs99['sum'].sort_values()

opp_costs_res299 = opp_costs_res99.loc[:, max_costs99.index]
#opp_costs_res299.sort_values(inplace = True)


max_costs24 = opp_costs_res24.T.merge(individ_load.T, left_index = True, 
                                right_index = True)
max_costs24 = max_costs24.merge(shap_load.T, left_index = True, 
                                right_index = True)
max_costs24['sum'] = max_costs24[['0_x', '0_y', 0]].max(axis = 1)
max_costs24 = max_costs24['sum'].sort_values()

opp_costs_res224 = opp_costs_res24.loc[:, max_costs24.index]
#opp_costs_res224.sort_values(inplace = True)

names99 = opp_costs_res299.columns
names24 = opp_costs_res224.columns


width = 0.5
fig, axs = plt.subplots(2,2, sharex = 'row', sharey = 'row',
                        gridspec_kw={'height_ratios': [1, 2]})  # Top row is thinner)
plt.subplots_adjust(wspace=0.1) 
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.17)

axs[0,0].barh(width, shap_load.iloc[0,:].sum(), 
         color='purple', edgecolor='white', 
         linewidth=3, height=width*.75)

axs[0,0].barh(0, individ_load.iloc[0,:].sum(), 
         color='olivedrab', hatch='xxx', 
         edgecolor='white', linewidth=3, height=width*.75)

axs[0,0].barh(-width, opp_costs_res99.iloc[0,:].sum(), 
         color='darkorange', hatch='\\\\\\',
         edgecolor='white', linewidth=3, height=width*.75)

axs[0,0].set_xlabel("$M")
axs[0,0].legend(frameon = False, loc = 'upper left')
axs[0,0].set_yticks([-width, 0, width], 
            ['Opp. Cost, \nReserves', 
            'Loading, \nIndivid. Ins.', 
            'Loading, \nPooled Ins.'], rotation = 0)
axs[0,0].set_xticks([0,  0.5*pow(10,8), 1*pow(10,8), 
            1.5*pow(10,8), 2*pow(10,8), 2.5*pow(10,8)],
           ['0', '50', '100', '150', '200', '250'])
axs[0,0].annotate("A", (235*pow(10,6), width*.9), weight = "bold", fontsize = 35)

axs[0,1].barh(0, individ_load.iloc[0,:].sum(), 
         color='olivedrab', hatch='xxx', 
         edgecolor='white', linewidth=3, height=width*.75)

axs[0,1].barh(width, shap_load.iloc[0,:].sum(), 
         color='purple', edgecolor='white', 
         linewidth=3, height=width*.75)

axs[0,1].barh(-width, opp_costs_res24.iloc[0,:].sum(), 
         color='darkorange', hatch='\\\\\\',
         edgecolor='white', linewidth=3, height=width*.75)

axs[0,1].set_xlabel("$M")
axs[0,1].set_xticks([0,  0.5*pow(10,8), 1*pow(10,8), 
            1.5*pow(10,8), 2*pow(10,8), 2.5*pow(10,8)],
           ['0', '50', '100', '150', '200', '250'])
axs[0,1].annotate("B", (235*pow(10,6), width*.9), weight = "bold", fontsize = 35)

############################ now country level ################################
offset = width/2
axs[1,0].bar(np.arange(0, len(names99)) + offset, opp_costs_res299.iloc[0,:],
        color='darkorange', hatch='\\\\\\',
        edgecolor = 'white', linewidth = 3, 
        label = 'Opportunity Cost, Reserves', 
        width = width)
    
axs[1,0].bar(np.arange(0, len(names99)) - offset, individ_load.loc[0, max_costs24.index], 
        color = 'olivedrab', hatch = 'xxx', edgecolor = 'white', 
        linewidth = 3, label = 'Contract Loading, \nIndividual Insurance', 
        width = width)

axs[1,0].bar(np.arange(0, len(names99)) - offset, shap_load.loc[0, max_costs24.index], 
        color = 'purple', edgecolor = 'white', linewidth = 3, 
        label = 'Contract Loading, \nInsurance Pooling', 
        width = width)

         
axs[1,0].set_xticks(np.arange(0, len(names99)), names99, rotation = 45)

#axs[1,0].set_xlabel("Country")
axs[1,0].set_ylabel("$M")
axs[1,0].annotate("C", (15.6, 40*pow(10,6)), weight = "bold", fontsize = 35)


axs[1,0].set_yticks([0, 10*pow(10,6), 20*pow(10,6), 30*pow(10,6), 40*pow(10,6)],
           ['0', '10', '20', '30', '40'])
axs[1,0].set_xlim(0,len(names24)+0.7)

axs[1,1].bar(np.arange(0, len(names24)) + offset, opp_costs_res224.iloc[0,:],
        color='darkorange', hatch='\\\\\\',
        edgecolor = 'white', linewidth = 3, 
        label = 'Opportunity Cost, Reserves', width = width)
    
axs[1,1].bar(np.arange(0, len(names24)) - offset, individ_load.loc[0, max_costs24.index], 
        color = 'olivedrab', hatch = 'xxx', edgecolor = 'white', 
        linewidth = 3, label = 'Contract Loading, Individual Insurance', width = width)

axs[1,1].bar(np.arange(0, len(names24)) - offset, shap_load.loc[0, max_costs24.index], 
        color = 'purple', edgecolor = 'white', linewidth = 3, 
        label = 'Contract Loading, Insurance Pooling', width = width)

         
axs[1,1].set_xticks(np.arange(0, len(names24))-offset, names24, rotation = 45)

#axs[1,1].set_xlabel("Country")
axs[1,1].set_ylabel("$M")
axs[1,1].annotate("D", (15.6, 40*pow(10,6)), weight = "bold", fontsize = 35)
#axs[1,1].legend(frameon = False, loc = 'upper left')

axs[1,1].set_yticks([0, 10*pow(10,6), 20*pow(10,6), 30*pow(10,6), 40*pow(10,6)],
           ['0', '10', '20', '30', '40'])
axs[1,1].set_xlim(0,len(names24)+0.7)

## floating legend 
#axs[1,0].legend(frameon = False, loc = 'upper left')
h, l = axs[1,1].get_legend_handles_labels()
axs[1,0].legend(h,l, frameon = False, loc = 'upper center', ncol = 3,
                bbox_to_anchor = (1.03, -0.23))

fig.text(0.3, 0.96, '1974-1999 Interest Rates', ha='center', fontsize = 20)
fig.text(0.76, 0.96, '2000-2024 Interest Rates', ha='center', fontsize = 20)











