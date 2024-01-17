# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:47:31 2024

@author: rcuppari
"""



def predict_countries(country, hybas2, reg_inputs): 
    if 'watershed' in hybas2:
        hybas = 'watersheds'
    else: hybas = hybas2
    
    try: 
        inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data.csv").iloc[:,1:]
    except: 
        try: 
            inputs = pd.read_csv(f"Results_all/Inputs_{hybas}/{country}_input_data_{hybas}.csv").iloc[:,1:]
        except: 
            try: 
                inputs = pd.read_csv(f"Results_all/Inputs_{hybas2}/{country}_input_data_{hybas}.csv").iloc[:,1:]
            except: 
                inputs = 'failed'
                print(f"failed to find {country}, {hybas}")
            
    if isinstance(inputs, pd.DataFrame): 
        inputs.set_index('year', inplace = True)
        inputs = inputs.merge(country_gen, on = 'year')
    
#    if hybas2 != 'watersheds': 
#        for i in np.arange(0,len(list(reg_inputs.split(","))):
#            print(reg_inputs[i])
    try:         
        reg_inputs2 = inputs[[i for i in reg_inputs[1:]]] ## skip constant
        reg_inputs2 = sm.add_constant(reg_inputs2)
              
        y = inputs[country] ## outcome should be second column
        
        est = sm.OLS(y, reg_inputs2)
        est2 = est.fit()
        
        predicted = pd.DataFrame(est2.predict(reg_inputs2))
        predicted.columns = [country]
        
        predicted[predicted < 0] = 0
    #    predictions = pd.concat([predictions, predicted], axis = 1)  
        #predictions.index = inputs.year
        
        #gen_data2 = make_numeric(gen_data)
        #gen_data2 = gen_data2[(gen_data2['year'] <= inputs.year.max()) & (gen_data2['year'] >= inputs.year.min())]
        #gen_data2.set_index('year', inplace = True)
    #    error_pct = ((predictions - gen_data2)/gen_data2).dropna(axis =1)
    #    error = (predictions - gen_data2).dropna(axis =1)
        error_pct = ((predicted.iloc[:,0] - inputs.iloc[:,-1])/inputs.iloc[:,-1]).mean()*100
        error = (predicted.iloc[:,0] - inputs.iloc[:,-1])
    
        print(f"{country} error pct: {error_pct}")
    except: 
        predicted = pd.DataFrame(np.repeat(-9,20))
        predicted.columns = [country]
    
    return predicted

def make_scatters(predictions, name = ''): 
    ncols = 3
    nrows = math.ceil(len(predictions.columns)/ncols)
    
    fig = plt.figure()
    gs = plt.GridSpec(nrows, ncols, wspace = 0.2, hspace = .4) 
    axs = []
    for c, num in zip(predictions.columns, range(1, len(predictions.columns)+1)):
        print(c)
        print(num)
        axs.append(fig.add_subplot(gs[num-1]))
        im = axs[-1].scatter(predictions[c], gen_data2[c], c = r2[c], vmin = .3, vmax = 1)
        axs[-1].plot([gen_data2[c].min(), gen_data2[c].max()],
                    [gen_data2[c].min(), gen_data2[c].max()], 
                    color = 'black', linestyle = '--')
        axs[-1].set_xlabel("Predicted", fontsize = 14)
        axs[-1].set_ylabel("Observed", fontsize = 12)
        axs[-1].set_title(f"{c}", fontsize = 14)
        #plt.tight_layout()
    
    axs.append(fig.add_subplot(gs[nrows-1, 2:len(predictions.columns)]))
    fig.colorbar(im, cax = axs[-1], orientation = 'horizontal')
    axs[-1].set_xlabel("r2",fontsize = 14)
    fig.savefig(fname = f"Figures/indices{name}.jpg")
