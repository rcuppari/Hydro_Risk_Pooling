# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:13:22 2023

@author: rcuppari
"""

from itertools import chain, combinations
import itertools
from math import factorial as fac
from wang_transform import wang_slim
from wang_transform import wang 
import pandas as pd
import numpy as np 

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_rename_list(i, payouts, name = 'payout'):
    x = pd.DataFrame(payouts[i].sum(axis = 1))
    x.columns = [name]
#    print(payouts[x])
    return x

## 1) identify all combinations that include the country 
## 2) match these combinations to the group payouts/E[losses]
## note: this might be useless to do, because the marginal contribution 
## is always the value of the loss? Doesn't really offset? Unless
## it becomes the average payout per country? (doing that by using ".sum()")  
    
## a nice breakdown of what the individual variables mean 
## https://math.stackexchange.com/questions/3583957/understanding-shapley-values-formula
## https://medium.com/the-modern-scientist/what-is-the-shapley-value-8ca624274d5a

def Shapley_calc(country, combs, payouts, pricing = 'wang', loading_factor = 1.3, 
                 code = 'wang'): 
    ## find expected losses where country is in the subset 
    ids = [combs.index(i) for i in combs if(country in i)]
    country_subsets = [combs[i] for i in ids]
    exclud_subsets = [list(c) for c in country_subsets]
#    loss_subsets = [group_loss[i] for i in ids]
    loss_subsets = []
    loss_subsets_exc = []
    
    ## but also need to find those same subsets *without the country* 
    ## and make sure remove the right country.  
    for i, names in enumerate(country_subsets):
#        print(names)
        exclud_subsets[i].remove(country)
        names_exc = exclud_subsets[i]
        
        if pricing == 'wang': 
    
            if code == 'wang': 
                new_payouts = get_rename_list(names, payouts, name = 'asset')
                wang_val = wang(new_payouts, 
                                              contractType = 'put', lam = 0.25,\
                                              premOnly = True)
                loading_contrib = wang_val - new_payouts['asset'].mean()
                loss_subsets.append(loading_contrib) #\
                               #           for i in country_subsets[1:]]#, from_user = True) \
            
                new_payouts_exc = get_rename_list(names_exc, payouts, name = 'asset')
                wang_val_exc = wang(new_payouts_exc, 
                                              contractType = 'put', lam = 0.25,\
                                              premOnly = True)
                loading_contrib_exc = wang_val_exc - new_payouts_exc['asset'].mean()
                loss_subsets_exc.append(loading_contrib_exc) 

                country_payouts = pd.DataFrame(payouts[country])
                country_payouts.columns = ['payout']
    
            else: 
                new_payouts = get_rename_list(names, payouts, name = 'payout')
                wang_val = wang_slim(new_payouts, 
                                contract = 'put', lam = 0.25,
                                premOnly = True)
                loading = new_payouts.mean() + wang_val
                loss_subsets.append(loading) 
                
                new_payouts_exc = get_rename_list(names_exc, payouts, name = 'payout')            
                wang_exc = wang_slim(new_payouts_exc, 
                                     contract = 'put', lam = 0.25,\
                                         premOnly = True)
                loading_exc = new_payouts_exc.mean() + wang_exc
                loss_subsets_exc.append(loading_exc) 
    
                ## get the contribution when it is just the single country 
                country_payouts = pd.DataFrame(payouts[country])
                country_payouts.columns = ['payout']
        else: 
            loss_subsets = [pd.DataFrame(payouts[i]).sum(axis = 1).mean()*loading_factor \
                            for i in country_subsets[1:]]        
                
            loss_subsets_exc = [pd.DataFrame(payouts[i]).sum(axis = 1).mean()*loading_factor \
                                for i in exclud_subsets[1:]]
    
    ## flipping from standard equation because talking about losses and not gains
    marg_contrib = (pd.DataFrame(loss_subsets) - pd.DataFrame(loss_subsets_exc))#.mean()
        
    marg_contrib.columns = ['marg_contrib']

    # ###########################################################################
    # # https://medium.com/the-modern-scientist/what-is-the-shapley-value-8ca624274d5a
    # ## the "avg" term 
    # num_countries = len(combs[-1])
    # ## NOTE: because order does not matter for us, we only want to use the 
    # ## *combination* not the *permutation* (i.e., not just a factorial)

    summed = 0 
    for i, names in enumerate(loss_subsets):
        C = len(country_subsets[i])
        n = len(combs[-1])
        num_term1 = fac(C - 1) * fac(n - C)
        den_term1 = fac(n)
        term2 = loss_subsets[i] - loss_subsets_exc[i]

        ## weigh the value by the number of times that it will occur
        summed += ((num_term1/den_term1) * term2) 

#         print(f"num1: {num_term1}")
#         print(f"deno1: {den_term1}")
#         print(f"term2: {term2}")
#         print(f"summed: {summed}")
#         print()

    return marg_contrib, summed
