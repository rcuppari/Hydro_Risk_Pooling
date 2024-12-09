# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:27:14 2024

@author: rcuppari
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import geopandas as gpd
## import custom funcs
from analysis_plotting_funcs1 import id_hydro_countries

###############################################################################
## this script produces the figure on changes in the coefficient of variation
## and payouts as the size of the pool increases
## it includes a few variations on the figure as well 
###############################################################################

name = 'cap_fac995_final'
strike_val = 20
var_cap = 0.5
cap_fac = False
LCOE = 0.044 ## per kWh

## read in countries that are in the pool
payouts = pd.read_csv(f"Results_all/predicted_payouts_{name}.csv").iloc[:,1:]
## exclude countries with no payouts
payouts = payouts.loc[:, payouts.any()]

################# starting plotting parameters
labelsize = 15
plt.rcParams.update({
    'axes.titlesize': labelsize+4,  # Set the font size for axes titles
    'axes.labelsize': labelsize+2,  # Set the font size for x and y labels
    'xtick.labelsize': labelsize,  # Set the font size for x tick labels
    'ytick.labelsize': labelsize,  # Set the font size for y tick labels
    'legend.fontsize': labelsize+2,  # Set the font size for the legend
})

################# hydropower generation #######################
countries2 = id_hydro_countries(pct_gen = 25, year = 2015, cap_fac = False)
gen_bkwh = countries2.T
gen_bkwh.reset_index(inplace = True)
## drop the rows with the column names 
gen_bkwh.columns = gen_bkwh.iloc[0,:]
gen_bkwh = gen_bkwh.iloc[1:-4,:]
gen_bkwh = gen_bkwh.rename(columns = {'Country':'year'})

for c in gen_bkwh.columns: 
    gen_bkwh[c] = pd.to_numeric(gen_bkwh[c], errors = 'coerce')

gen_bkwh.set_index('year', inplace = True)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['name','pop_est','iso_a3', 'gdp_md_est', 'geometry']]
world.columns = ['country', 'pop', 'Country Code', 'gdp', 'geometry']
## for some strange reason, France is listed as -99 instead of FRA... replace
world.iloc[43,2] = 'FRA'

pct_hydro = pd.read_csv("Other_data/wb_hydro%.csv")[['Country Name', 'Country Code', '2015']]

hydro_gdf = pd.merge(world, pct_hydro, on = 'Country Code')


################# RESERVES NEEDED, WITHOUT ANY INSURANCE? #####################
############################## FIGURE XX ######################################
# JUST THINKING ABOUT WHAT WE WOULD NEED TO COVER THEIR HISTORICAL VARIABILITY#
## want to keep it consistent with capacity factor though, so using that here 
## looking at how much would need to be held in order to cover up to the 
## 10th% of lost value within the pool versus individually
## IMPORTANT: there is no negative value in generation. So, need to subtract 
## from whatever we deem a bad value (we will use a strike) to get 'losses'
## here we use the mean 
##########################
## and to make this generalizable, let's look at every combination 
## of 1, 2, 3, ... n participants and take the average differences
## Will use IRENA 2021 LCOE data https://sciwheel.com/work/item/12749865/resources/14115980/pdf
## could also use https://www.iea.org/reports/projected-costs-of-generating-electricity-2020
gen_vals = id_hydro_countries(pct_gen = .01, year = 2015)
gen_vals = gen_vals.T
gen_vals.reset_index(inplace = True)
## drop the rows with the column names 
gen_vals.columns = gen_vals.iloc[0,:]
gen_vals = gen_vals.iloc[1:-4,:]
gen_vals = gen_vals.rename(columns = {'Country':'year'})

for country in gen_vals.columns[1:]:
    print(country)
    
    gen_vals[country] = pd.to_numeric(gen_vals[country], errors = 'coerce')
    gen_vals[country] = gen_vals[country] * LCOE *pow(10,9)## billion kWh * $/billion kWh

avg_val = pd.DataFrame(gen_vals.mean()).iloc[1:,:]
avg_val.columns = ['avg_val']
avg_val.reset_index(inplace = True)
avg_val.columns = ['Country Name', 'avg_val']

merged_vals = hydro_gdf.merge(avg_val, on = 'Country Name', how = 'inner')

for r in range(0, len(merged_vals)): 
    if merged_vals['avg_val'].iloc[r] == 0: 
        merged_vals['avg_val'].iloc[r] = 'na'

merged_vals['avg_val'] = pd.to_numeric(merged_vals['avg_val'], errors = 'coerce')
        

gen_vals2 = gen_vals.iloc[:,1:]
gen_vals2 = gen_vals2.loc[:, payouts.columns]

## find combinations of participants
## randomly choose x of each length 
## (would take too long to get every combination of every length)

combs = []
for i in range(1, len(gen_vals2.columns)+1):
    print(i)
    for x in range(400):
        col_ilocs = np.random.choice(np.arange(0, len(gen_vals2.columns)), i, replace=False)
        combs.append(list(gen_vals2.columns[col_ilocs].values))

import itertools
#combs.sort()
combs = list(combs for combs,_ in itertools.groupby(combs))
print('combs done: ' + str(len(combs)))

## need to drop or fill nas when combining (sadly) 
gen_vals2 = gen_vals2.dropna(axis = 0)
gen_vals2.reset_index(inplace = True)
gen_vals2 = gen_vals2.iloc[:, 1:]

##  need to get individual vars for each 
sum_individ = []
## and the pooled values for each combination
group_vals = []
## and the means
means = []
## and the coefficient of variation 
cv_vals = []
## and the stds 
stds = []

for i in combs:  
    #print(i)
    if len(i) == 1: 
        sum_val = gen_vals2[i]
    else:         
        sum_val = gen_vals2[i].sum(axis = 1)
    
    var_val = np.percentile(sum_val, var_cap)

    if len(i) == 1: 
        var_i = (sum_val.mean() - np.percentile(sum_val, var_cap))[0]
    else: 
        var_i = (sum_val.mean() - np.percentile(sum_val, var_cap))
    group_vals.append(var_i)
    
    vals = 0
    
    mean = payouts[i].sum(axis = 1).mean()
    std = payouts[i].sum(axis = 1).std()
    means.append(mean)
    cv_vals.append(std/mean)
    stds.append(std)
    
    for country in i: 
        #print(country)
        gen_vals3 = gen_vals2[country].dropna()
        gen_vals3 = gen_vals3[gen_vals3 > 0]
        
        var_val = np.percentile(gen_vals3, var_cap)

        ## use 95% VaR
        vals += gen_vals3.mean() - var_val
        #print(f"{country} in i, vals = {vals}")

    sum_individ.append(vals)
print('done for loop #1')

## now find the average based on the length of each pool

group_vals2 = pd.DataFrame(group_vals).fillna(method = 'ffill')
group_vals = list(group_vals2.iloc[:,0])

vals_by_group_size = pd.DataFrame(index = np.arange(1, \
                                  len(gen_vals2.columns)+2), 
                                  columns = ['var_group', 'mean', 
                                             'std', 'cv'])

for n in range(1, len(gen_vals2.columns)+1): 
    print(n)
    ## need to retrieve the combinations of a certain value 
    ## and then match them to the values 
    ids = [index for index, i in enumerate(combs) if len(i) == n]
    subsets = [group_vals[i] for i in ids]
    vals_by_group_size.loc[n,'var_group'] = sum(subsets)/len(subsets)
    
    subsets_mean = [means[i] for i in ids]
    subsets_std = [stds[i] for i in ids]
    subsets_cv = [cv_vals[i] for i in ids]
    
    ## cv = ratio of std to mean 
    vals_by_group_size.loc[n,'mean'] = sum(subsets_mean)/len(subsets_mean)
    vals_by_group_size.loc[n,'std'] = sum(subsets_std)/len(subsets_std)
    vals_by_group_size.loc[n,'cv'] = sum(subsets_cv)/len(subsets_cv)
print('done for loop #2')

## okay, let's group them by size of each group
vals_by_group_size['sum_individ'] = 0
for n in range(1, len(gen_vals2.columns)+1): 
      print(n)
      ## need to retrieve the combinations of a certain value 
      ## and then match them to the values 
      ids = [index for index, i in enumerate(combs) if len(i) == n]
      subsets = [sum_individ[i] for i in ids] 
      vals_by_group_size['sum_individ'].iloc[n-1] = sum(subsets)/len(subsets)

vals_by_group_size['diff'] = vals_by_group_size['sum_individ']  - \
                            vals_by_group_size['var_group'] 
vals_by_group_size = vals_by_group_size.iloc[1:, :]

vals_by_group_size = vals_by_group_size.iloc[:-1,:]
vals_by_group_size['pct_diff'] = vals_by_group_size['diff'] / vals_by_group_size['sum_individ']
print(f"Avg pct difference in cvars between group and summed individual: {100*vals_by_group_size['pct_diff'].mean():,}")

###############################################################################
## plot with bars 
ind = np.arange(len(vals_by_group_size))#[:14]
width = np.min(np.diff(ind))/3

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                round(height/(1*pow(10,9)),1),#'%d' % int(height),
                ha='center', va='bottom', fontsize = 12) 

#fig, (ax, ax2) = plt.subplots(2, 1)
fig, ax = plt.subplots()
rects2 = ax.bar(ind, vals_by_group_size['sum_individ'], width = width, \
                label = 'Sum, Individual', color = 'orange',align='edge')
rects3 = ax.bar(ind+width, vals_by_group_size['var_group'], width = width, \
                label = 'Pool', color = 'purple',align='edge')
ax.set_xlabel("Number of Pool Participants", fontsize = 16) 
ax.set_ylabel("CVaR ($B)", fontsize = 16)
ax.set_xticks(ind + width/2, 
              1 + np.arange(0, len(ind)), 
            fontsize = 14)
ax.set_yticks([0, 1*pow(10,9), 2*pow(10,9), 3*pow(10,9)],#, 6*pow(10,9), 9*pow(10,9), 12*pow(10,9)],
            ['0', '1', '2', '3'],#, '6', '9', '12'], 
            fontsize = 14)
ax.set_xlim(0,len(ind))
ax.legend(frameon = False, fontsize = 14, loc = 'upper left')        
autolabel(rects2)
autolabel(rects3)
plt.tight_layout()
plt.draw()

################## CV panel with  
## difference in reserves with individual + pool 
width = 0.8
fig, (ax, ax1) = plt.subplots(2,1, sharex = True)
rects1 = ax.bar(vals_by_group_size.index, vals_by_group_size['cv'], 
                color = 'orange', alpha = 0.8,
                width = width, label = 'Coefficient of Variation')
ax.set_ylabel("Coefficient of Variation")
ax.set_xlabel("Number Pool Participants")
ax.set_xticks(np.arange(1,15), np.arange(1,15))

ax.set_yticks(np.arange(0,2.5, 0.5), np.arange(0, 2.5, 0.5))
## have mean separately as line
ax2 = ax.twinx()
lns = ax2.plot(vals_by_group_size['mean'], linewidth = 3, \
        color = 'gray', label = 'Average Payout')
ax2.set_yticks([0, 0.5*pow(10,8), 1*pow(10,8), 1.5*pow(10,8), 
                2*pow(10,8), 2.5*pow(10,8)],
               [0, 50, 100, 150, 200, 250])
ax2.set_ylabel("Average Annual Payout ($M)")

lns = [rects1] + lns
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper center', ncol = 2, frameon = False)

## now show difference in reserves by pool size 
ax1.bar(vals_by_group_size.index, vals_by_group_size['diff'], \
        color = 'purple', width = width, alpha = 0.8)
ax1.set_xlabel("Number Pool Participants")
ax1.set_ylabel("Savings ($M)")
ax1.set_yticks([0, 2*pow(10,8), 4*pow(10,8), 6*pow(10,8), 8*pow(10,8), \
                10*pow(10,8), 12*pow(10,8)],
               ['0', '200', '400', '600', '800', '1000', '1200'])
ax1.set_xticks(np.arange(0,15), np.arange(0,15))

ax.set_xlim(1, 15)
ax2.set_xlim(1, 15)
ax1.set_xlim(1, 15)

################# CV means single panel 

fig, ax1 = plt.subplots()

# Primary axis (left)
rects1 = ax1.bar(vals_by_group_size.index, vals_by_group_size['cv'], 
                 color='orange', alpha=0.8, label='Coefficient of Variation', zorder=1)
ax1.set_xlabel("Number Pool Participants")
ax1.set_xticks(np.arange(1, 17), np.arange(0, 16))
ax1.set_yticks(np.arange(0, 2.5, 0.5), np.arange(0, 2.5, 0.5))
ax1.set_xlim(1, 16.5)
ax1.yaxis.tick_right()

# Secondary axis (right)
ax2 = ax1.twinx()
lns = ax2.plot(vals_by_group_size.index, vals_by_group_size['diff'], 
               color='purple', linewidth=4, alpha=0.8, label='Reduction in VaR', zorder=2)
ax2.set_yticks([0, 2*pow(10,8), 4*pow(10,8), 6*pow(10,8), 8*pow(10,8), \
                10*pow(10,8), 12*pow(10,8), 14*pow(10,8), 16*pow(10,8)],
               ['0', '200', '400', '600', '800', '1000', '1200', '1400', '1600'])
ax2.set_xticks(np.arange(2, 17), np.arange(2, 17))


## swap ticks and corresponding labels 
ax2.yaxis.tick_left()
ax1.yaxis.tick_right()
ax1.set_ylabel("Reduction in VaR ($M)", labelpad = 45)
ax2.set_ylabel("Coefficient of Variation", labelpad = 45)

# Combine legends
lns = [rects1] + lns
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper center', ncol=2, frameon=False)
plt.show()
