# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:44:27 2024

@author: rcuppari
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
## import custom funcs
from analysis_plotting_funcs import id_hydro_countries

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
payout_cap = 1 ## maximum payout as percent capacity factor (i.e., VaR)
LCOE = 0.044 # LCOE is 0.044/kWh, used to translate generation to $$
use_tariffs = False ## choice to use LCOE or observed tariff data if available
name = 'cap_fac99_loading' ## suffix for reading in files
alt_rate99 = 8.48 ## rate of alternative investment (low risk, lower liquidity) as % 1981- 1999
safe_rate99 = 6.47 ## rate of "risk-free" investment (lower risk, high liquidity) as % 1981 - 1999
alt_rate24 = 3.28 ## rate of alternative investment (low risk, lower liquidity) as % 2000 - 2024
safe_rate24 = 1.83 ## rate of "risk-free" investment (lower risk, high liquidity) as % 2000 - 2024
time_horizon = 10

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

## SHOULD JUST USE MEAN CAPACITY?? OTHERWISE HARD TO HAVE A GOOD COMPARISON...
for c in pred_pay.columns:
    country_capacity[c] = pd.to_numeric(country_capacity[c], errors = 'coerce')
    country_capacity[c] = country_capacity[c].mean()
    
################################### regular ###################################
## as in, generation 
gen_data2 = id_hydro_countries(pct_gen = 25, year = 2015, 
                                      cap_fac = False)#, capacity = True)
gen_data2 = gen_data2.T
#country_gen.reset_index(inplace = True)
gen_data2.columns = gen_data2.iloc[0,:]
gen_data2 = gen_data2.iloc[3:-5,:]
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
gen_data = gen_data.iloc[3:-5,:]
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
def cpv(df, discount = 5, time = 20, rate = 'na'): 
    
    cpvs = pd.DataFrame(index = np.arange(0,time), columns = df.index) 
    for c, country in enumerate(df.index):
        print(country)
#        cpv.loc[c,country] = pd.DataFrame(index = np.arange(0, time), columns = ['cpv'])

        for r in range(0, time):
            new_yr = df[c]/pow((1 + discount/100),r)
            
            if r == 0: 
                cpvs.loc[r, country] = new_yr 
            else: 
                cpvs.loc[r, country] = cpvs.loc[r-1, country] + new_yr    
    avg = cpvs.sum(axis = 1)
    
    return avg, cpvs

labelsize = 14
plt.rcParams.update({
    'axes.titlesize': labelsize+2,  # Set the font size for axes titles
    'axes.labelsize': labelsize,  # Set the font size for x and y labels
    'xtick.labelsize': labelsize-2,  # Set the font size for x tick labels
    'ytick.labelsize': labelsize-2,  # Set the font size for y tick labels
    'legend.fontsize': labelsize-2,  # Set the font size for the legend
})

time = len(pred_pay)
discount = 3
cpvs = {}
cols = ['avg_pay', 'Shapley', 'individ']
avgs = pd.DataFrame(index = np.arange(0, time), columns = cols)
for col in cols: 
    avgs.loc[:,col], cpvs[col] = cpv(df = shap[col])
    
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
    
    
###############################################
def compound_interest(principal, rate, time): 
    # Calculates compound interest
    ## rate here is in % form so we divide by 100 
    Amount = principal * (pow((1 + rate / 100), time))
    return Amount

def pv_opp_cost(money, time_horizon, safe_rate, 
                alt_rate, discount = 2.5):
    opp_cost2 = []
    cum_opp_cost = 0
    
    for i in range(0, len(money)): 
        val_safe = compound_interest(money[i], safe_rate, time_horizon)
        val_alt = compound_interest(money[i], alt_rate, time_horizon)
    
        opp_cost_ann = (val_alt - val_safe)/pow((1+discount/100),i)
               
        if i > 0: 
            opp_cost2.append(opp_cost_ann + opp_cost2[i-1])
        else: 
            opp_cost2.append(opp_cost_ann)
            
        cum_opp_cost += opp_cost_ann   

    return opp_cost2
    
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
            int(country_capacity.loc[2019, country]) * cost.values * 8760
        strikes2.loc[country, '$80%'] = strikes2.loc[country, '80%'] * \
            int(country_capacity.loc[2019, country]) * cost.values * 8760
        
    else: 
        strikes2.loc[country, '$99%'] = strikes2.loc[country, '99%'] * \
            int(country_capacity.loc[2019, country]) * LCOE * 1000 * 8760
        strikes2.loc[country, '$80%'] = strikes2.loc[country, '80%'] * \
            int(country_capacity.loc[2019, country]) * LCOE * 1000 * 8760
            
###############################################################################
############################### 1981 - 1999 ###################################
############################################################################### 
opp_costs_res99 = pd.DataFrame(index = np.arange(0,time_horizon), columns = gen_diff.index)

strikes2['difference'] = strikes2['$99%'] - strikes2['$80%']
## want the opportunity costs for each country over the 17 years or so 
for country in gen_diff.index: 
    avg_pay = gen_diff.loc[country, 'avg_pay']
    
    ## just compare as maximum insurance payout
    add_val_reserves =  pred_pay[country].max()#strikes2.loc[country, '$99%']
        
    if add_val_reserves < 0: 
        print(f"WOOPS! Check on {country}")
    
    opp_costs_res99.loc[:,country] = add_val_reserves * \
                            (alt_rate99 - safe_rate99)/100 * time_horizon  


add_val_res_pool99 = np.percentile(pred_pay.sum(axis = 1), 100-payout_cap)

opp_costs_res_pool99 = pd.DataFrame(pv_opp_cost(np.repeat(add_val_res_pool99, time_horizon),
                        time_horizon = time_horizon, 
                        safe_rate = safe_rate99, alt_rate = alt_rate99))    

################################################## we can do this on average... 
opp_costs_res99 = opp_costs_res99.loc[[0, 4, 9],:]#, 14], :]

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
opp_costs_shap24 = pd.DataFrame(index = np.arange(0, time_horizon), columns = gen_diff.index)
opp_costs_individ24 = pd.DataFrame(index = np.arange(0, time_horizon), columns = gen_diff.index)
opp_costs_res24 = pd.DataFrame(index = np.arange(0, time_horizon), columns = gen_diff.index)

## want the opportunity costs for each country over the 17 years or so 
for country in gen_diff.index: 
    ## just compare as maximum insurance payout
    add_val_reserves =  pred_pay[country].max()#strikes2.loc[country, '$99%']
        
    if add_val_reserves < 0: 
        print(f"WOOPS! Check on {country}")
    
    opp_costs_res24.loc[:,country] = add_val_reserves * \
                            (alt_rate24 - safe_rate24)/100 * time_horizon   


add_val_res_pool24 = np.percentile(pred_pay.sum(axis = 1), 100-payout_cap)

opp_costs_res_pool24 = pd.DataFrame(pv_opp_cost(np.repeat(add_val_res_pool24, time_horizon),
                        time_horizon = time_horizon, 
                        safe_rate = safe_rate24, alt_rate = alt_rate24))    

################################################## we can do this on average... 
opp_costs_res24 = opp_costs_res24.loc[[0, 4, 9],:]#, 14], :]


labelsize = 20
plt.rcParams.update({
    'axes.titlesize': labelsize+2,  # Set the font size for axes titles
    'axes.labelsize': labelsize,  # Set the font size for x and y labels
    'xtick.labelsize': labelsize-2,  # Set the font size for x tick labels
    'ytick.labelsize': labelsize-2,  # Set the font size for y tick labels
    'legend.fontsize': labelsize-2,  # Set the font size for the legend
})

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
opp_costs_res299 = opp_costs_res99.loc[0,:]
opp_costs_res299.sort_values(inplace = True)

opp_costs_res224 = opp_costs_res24.loc[0,:]
opp_costs_res224.sort_values(inplace = True)

names = opp_costs_res299.index

width = 1
fig = plt.figure()

plt.barh(-width, opp_costs_res99.iloc[0,:].sum(), 
         color='tomato', hatch='\\\\', 
         edgecolor='white', linewidth=3, height=width/2.5)

plt.barh(-width/2, opp_costs_res24.iloc[0,:].sum(), 
         color='darkorange', hatch='\\\\\\', 
         edgecolor='white', linewidth=3, height=width/2.5)

plt.barh(0, individ_load.iloc[0,:].sum(), 
         color='olivedrab', hatch='xxx', 
         edgecolor='white', linewidth=3, height=width/2.5)

plt.barh(width/2, shap_load.iloc[0,:].sum(), 
         color='purple', edgecolor='white', linewidth=3, height=width/2.5)

plt.xlabel("$M")
plt.legend(frameon = False, loc = 'upper left')
plt.yticks([-width, -width/2, 0, width/2], 
           ['Opportunity Cost, \nReserves \n(1981-1999 rates)',  
            'Opportunity Cost, \nReserves \n(2000-2024 rates)', 
            'Loading, \nIndividual', 'Loading, \nPooled'])
plt.xticks([0,  0.5*pow(10,8), 1*pow(10,8), 
            1.5*pow(10,8), 2*pow(10,8), 2.5*pow(10,8), 3*pow(10,8), 3.5*pow(10,8)],
           ['0', '50', '100', '150', '200', '250', '300', '350'])


