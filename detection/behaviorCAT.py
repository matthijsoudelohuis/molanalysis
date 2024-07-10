#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches
import pandas as pd
import seaborn as sns


from scipy import special
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def clean_data(ses,filter_engaged):
    trial=ses.trialdata.copy()
    behavior=ses.behaviordata.copy()
    video=ses.videodata.copy()
    
    #delete 1st trials
    tstart=trial.loc[trial['lickResponse']==1].iloc[1]['tStart']
    trial=trial.loc[trial['tStart']>=tstart].reset_index(drop=True)
    behavior=behavior.loc[behavior['ts']>tstart].reset_index(drop=True)
    video=video.loc[video['ts']>tstart]
    
    # delete pupil w/ -5<z-score>5
    z=stats.zscore(video['pupil_area'], nan_policy='omit')
    video=video.drop(z[(z>5)|(z<-5)].index).reset_index(drop=True)

    if filter_engaged:
        trial = trial[trial['engaged']==1].reset_index(drop=True)
        tend=trial.iloc[-1]['tEnd']
        behavior=behavior.loc[behavior['ts']<tend].reset_index(drop=True)
        video=video.loc[video['ts']<tend].reset_index(drop=True)

    # make ts start at 0cm
    #video['ts']=(video['ts']-video.iloc[0]['ts'])
    #behavior['ts']=(behavior['ts']-behavior.iloc[0]['ts'])

    return trial,behavior,video


def psychometric_function(x, mu, sigma, lapse_rate, guess_rate):
    return guess_rate + (1 - guess_rate - lapse_rate) * 0.5 * (1 + special.erf((x - mu) / (np.sqrt(2) * sigma)))

def plot_psycurve(sessions,clean=True,filter_engaged=True):
    for ises,ses in enumerate(sessions):
        if clean:
            trialdata,behavior,video=clean_data(ses,filter_engaged)
        else:
            trialdata=ses.trialdata.copy()
            if filter_engaged:
                trialdata = trialdata[trialdata['engaged']==1].reset_index(drop=True)

        
        psydata = trialdata.groupby(['signal'])['lickResponse'].sum() / trialdata.groupby(['signal'])['lickResponse'].count()
        x = psydata.keys().to_numpy()
        y = psydata.to_numpy()
       
        X = trialdata['signal'] #Fit with actual trials, not averages per condition
        Y = trialdata['lickResponse']
        initial_guess           = [20, 15, 1-y[-1], y[0]]  # Initial guess for parameters (mu,sigma,lapse_rate,guess_rate)
        # set guess rate and lapse rate to be within 10% of actual response rates at catch and max trials:
        bounds                  = ([0,2,(1-y[-1])*0.9,y[0]*0.9-0.01],[100,40,(1-y[-1])*1.1+0.01,y[0]*1.1])
        # bounds                  = ([0,4,0,0],[100,40,0.5,0.5])
        
        # Fit the psychometric curve to the data using curve_fit
        # params, covariance      = curve_fit(psychometric_function, x, y, p0=initial_guess,bounds=bounds)
        params, covariance      = curve_fit(psychometric_function, X, Y, p0=initial_guess,bounds=bounds)
        
        ## Plot the results
        fig, ax = plt.subplots()
        ax.scatter(x, y, label='data',c='k')
        x_highres = np.linspace(np.min(x),np.max(x),1000)
        ax.plot(x_highres, psychometric_function(x_highres, *params), label='fit', color='blue')
        ax.set_xlabel('Stimulus Intensity')
        ax.set_ylabel('Probability of Response')
        ax.legend(loc='lower right')
        ax.set_xlim([np.min(x),np.max(x)])
        ax.set_ylim([0,1])
        ax.set_title(ses.sessiondata['session_id'][0])

 # Print the fitted parameters
        print(f"Fitted Parameters ({ses.trialdata['session_id'][0]}):")
        print("mu:", '%2.2f' % params[0])
        print("sigma:", '%2.2f' % params[1])
        print("lapse_rate:", '%2.2f' % params[2])
        print("guess_rate:", '%2.2f' % params[3])
        
    return fig, params



def filtered_psycurve(sess, filter, n, clean=True, filter_engaged=True): 
    if n not in range(1,6): # n is the number of psy curves in the single plot, should be between 1 and 5
        raise ValueError("n is not between 1 and 5")
    
    for ises,ses in enumerate(sess):
        fig, ax = plt.subplots()
        colors = cm.get_cmap(cmap(filter), n+1)
        
        for i in range(n):
            trialdata, name = filter(ses, n, i, clean, filter_engaged)
            
            psydata = trialdata.groupby(['signal'])['lickResponse'].sum() / trialdata.groupby(['signal'])['lickResponse'].count()

            x = psydata.keys().to_numpy()
            y = psydata.to_numpy()

            X = trialdata['signal']
            Y = trialdata['lickResponse']
            #initial_guess           = [20, 15, 0.1, 0.1]  # Initial guess for parameters (mu,sigma,lapse_rate,guess_rate)
            #bounds                  = ([0,4,0,0],[100,40,0.5,0.5])
            initial_guess           = [20, 15, 1-y[-1], y[0]]  # Initial guess for parameters (mu,sigma,lapse_rate,guess_rate)
            # set guess rate and lapse rate to be within 10% of actual response rates at catch and max trials:
            bounds                  = ([0,2,(1-y[-1])*0.9,y[0]*0.9-0.01],[100,40,(1-y[-1])*1.1+0.01,y[0]*1.1])
            
            params, covariance      = curve_fit(psychometric_function, X, Y, p0=initial_guess,bounds=bounds)
            
            color = colors(i+1)
            # Plot the results
            ax.scatter(x, y, label=f'{name} {filter.__name__}',color=color)
            x_highres = np.linspace(np.min(x),np.max(x),1000)
            ax.plot(x_highres, psychometric_function(x_highres, *params), color=color)
             
        # make legend 
        ax.set_xlabel('Stimulus Intensity')
        ax.set_ylabel('Probability of Response')
        ax.legend(loc='lower right')
        ax.set_xlim([0,100])
        ax.set_ylim([0,1])
        ax.set_title(ses.trialdata['session_id'][0])
        #fig.savefig(ses.trialdata['session_id'][0])


def trial_number(ses, n, q, clean, filter_engaged):
    if clean:
        trial,behavior,video=clean_data(ses,filter_engaged)
    else:
        trial=ses.trialdata.copy()
        if filter_engaged:
            trial = trial[trial['engaged']==1].reset_index(drop=True)

    lowlim=round(len(trial.index)/n*q)
    highlim=round(len(trial.index)/n*(q+1))
    data=trial.iloc[lowlim:highlim]
    
    return data, (level(n, q))


def run_speed(ses, n, q, clean, filter_engaged):
    if clean:
        trial,behavior,video=clean_data(ses,filter_engaged)
    else:
        trial=ses.trialdata.copy()
        behavior=ses.behaviordata.copy()
        if filter_engaged:
            trial = trial[trial['engaged']==1].reset_index(drop=True)

    # 1. add mean runspeed for zpos in [-20, 5] to trialdata
    trial['meanSpeed'] = pd.Series(dtype=float)
    for i in range(1, (trial['trialNumber'].max()+1)):
        if i not in trial['trialNumber'].values:
            continue
        stimstart = trial[trial['trialNumber']==i]['stimStart_k'].values[0]
        zbefore = stimstart -20
        zafter  = stimstart +5
        meanspeed=behavior.loc[((behavior['trialNumber']==i) & behavior['zpos_k'].between(zbefore, zafter)), 'runspeed'].mean()
        trial.loc[trial['trialNumber']==i, 'meanSpeed']=meanspeed
    
    # 2. return trials
    highlim=np.quantile(trial['meanSpeed'].to_numpy(), (q+1)/n)
    lowlim=np.quantile(trial['meanSpeed'].to_numpy(), q/n)
    #print(f'highlim: {highlim}, lowlim: {lowlim}')
    subset=trial.loc[trial['meanSpeed'].between(lowlim, highlim)]
    return subset, (level(n, q))

def pupil_size(ses, n, q, clean, filter_engaged):
    if clean:
        trial,behavior,video=clean_data(ses,filter_engaged)
    else:
        trial=ses.trialdata.copy()
        behavior=ses.behaviordata.copy()
        video=ses.videodata.copy()
        if filter_engaged:
            trial = trial[trial['engaged']==1].reset_index(drop=True)
    
    # 1. add mean pupil size for zpos in [-20, 5] to trialdata
    trial['meanPupil'] = pd.Series(dtype=float)
    for i in range(1, (trial['trialNumber'].max()+1)):
        if i not in trial['trialNumber'].values:
            continue
        stimstart = trial[trial['trialNumber']==i]['stimStart_k'].values[0]
        tbefore = behavior.loc[((behavior['trialNumber']==i) & (behavior['zpos_k']>(stimstart -20))), 'ts'].min() #Not exactly -20, but almost
        tafter  = behavior.loc[((behavior['trialNumber']==i) & (behavior['zpos_k']<(stimstart +5))), 'ts'].max()
        meanpupil=video.loc[video['ts'].between(tbefore, tafter), 'pupil_area'].mean()
        trial.loc[trial['trialNumber']==i, 'meanPupil']=meanpupil

    trial=trial.dropna(subset=['meanPupil'])
    
    # 2. return trials
    highlim = np.quantile(trial['meanPupil'].to_numpy(), (q+1)/n)
    lowlim  = np.quantile(trial['meanPupil'].to_numpy(), q/n)
    subset = trial.loc[trial['meanPupil'].between(lowlim, highlim)]
    return subset, (level(n, q))

def level(n, i):   
    if n<4:
        if i==0:
            return "low"
        elif n-i==1:
            return "high"
        elif n-i==2:
            return "medium"
    else:
        if i==0:
            return "very low"
        elif i==1:
            return "low"
        elif n-i==1:
            return "very high"
        elif n-i==2:
            return "high"
        elif n-i==3:
            return "medium"

def cmap(filter):
    if filter==pupil_size:
        return 'YlOrRd'
    if filter==run_speed:
        return 'YlGnBu'
    if filter==trial_number:
        return 'RdPu'

def compute_dprime_c(signal,response):
    
    n=len(signal)
    hit_rate            = sum((signal == 100) & (response == 1)) / sum(signal == 100)
    falsealarm_rate     = sum((signal == 0) & (response == 1)) / sum(signal == 0)
    
    # avoiding inf values
    hit_rate = np.clip(hit_rate, 0.5/n, 1 - 0.5/n)
    falsealarm_rate = np.clip(falsealarm_rate, 0.5/n, 1 - 0.5/n)

    dprime              = stats.norm.ppf(hit_rate) - stats.norm.ppf(falsealarm_rate)
    criterion           = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(falsealarm_rate))
    return dprime,criterion


def filtered_psyparams_d_c(ses, filter, n, clean, filter_engaged): 
    if n not in range(1,6): # n is the number of psy curves in the single plot, should be between 1 and 5
        raise ValueError("n is not between 1 and 5")
    
    min_R=0.2 # sessions with at least 1 fit with R-squared below this value are excluded

    names = []
    values = []
    rs = []


    for i in range(n):
        trialdata, name = filter(ses, n, i, clean, filter_engaged)
        
        psydata = trialdata.groupby(['signal'])['lickResponse'].sum() / trialdata.groupby(['signal'])['lickResponse'].count()

        x = psydata.keys().to_numpy()
        y = psydata.to_numpy()

        X = trialdata['signal']
        Y = trialdata['lickResponse']
        #initial_guess           = [20, 15, 0.1, 0.1]  # Initial guess for parameters (mu,sigma,lapse_rate,guess_rate)
        #bounds                  = ([0,4,0,0],[100,40,0.5,0.5])
        initial_guess           = [20, 15, 1-y[-1], y[0]]  # Initial guess for parameters (mu,sigma,lapse_rate,guess_rate)
        # set guess rate and lapse rate to be within 10% of actual response rates at catch and max trials:
        bounds                  = ([0,2,(1-y[-1])*0.9,y[0]*0.9-0.01],[100,40,(1-y[-1])*1.1+0.01,y[0]*1.1])
        
        #params, covariance      = curve_fit(psychometric_function, x, y, p0=initial_guess,bounds=bounds)
        params, covariance      = curve_fit(psychometric_function, X, Y, p0=initial_guess,bounds=bounds)
        r=r2_score(y,psychometric_function(x,*params))
        d,c=compute_dprime_c(trialdata['signal'],trialdata['lickResponse'])

        names.append(name)
        values.append(np.append(params,[d,c]))
        if r<min_R:
            rs.append(r)
    
    if len(rs)==0:
        return values, names, ses.trialdata['session_id'][0]


def store_params(sess,filter,clean,filter_engaged):
    nparams=6
    nsplits=3
    param_list=[]
    
    for ises,ses in enumerate(sess):
        try:
            params,names,session=filtered_psyparams_d_c(ses,filter,nsplits,clean,filter_engaged)
        except TypeError:
            continue
        param_list.append(params)

    store=np.empty([len(param_list),nparams,nsplits])
    for ises,params in enumerate(param_list):
        for i in range(nsplits):
            store[ises,:,i] = params[i] 
    
    #print("Number of sessions:", store.shape[0])

    return store

# store_params[:,0,:] --> all the mu 
# store_params[:,1,:] --> all the sigma
# store_params[:,2,:] --> all the lapse_rate
# store_params[:,3,:] --> all the guess_rate
# store_params[:,4,:] --> all the d-prime
# store_params[:,5,:] --> all the criterion
# store_params[:,:,0] --> low 'filter'
# store_params[:,:,1] --> medium 'filter'
# store_params[:,:,2] --> high 'filter'


def storedf(sess,filter,clean,filter_engaged):
    nsplits=3
    
    store=pd.DataFrame(columns=['session','level','mu','sigma','lapse_rate','guess_rate','dprime','criterion'])
    for ises,ses in enumerate(sess):  
        try:
            params,names,session=filtered_psyparams_d_c(ses,filter,nsplits,clean,filter_engaged)
        except TypeError:
            continue
        
        for i in range(nsplits):
            row = [session, names[i]] + params[i].tolist()
            store.loc[len(store.index)] = row
    
    #nSessions = len(pd.unique(store['session']))
    #print("Number of sessions:", nSessions)

    return store

# nb of rows = nSessions*nSplits(3)

def plot_params(sess,filter,clean=True,filter_engaged=True):
    store=storedf(sess,filter,clean,filter_engaged)
    param_names=['mu','sigma','lapse_rate','guess_rate','dprime','criterion']
    colors = cm.get_cmap(cmap(filter), 9)

    fig, axs = plt.subplots(3,2, figsize=(8,13))
    ps=[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    for ip,p in enumerate(param_names):
        ax=axs[ps[ip]]
        sns.lineplot(ax=ax, data=store, x='level', y=p, hue='session', palette='gray', linewidth=.6) 
        sns.pointplot(ax=ax, data=store, x='level', y=p, capsize=.1, errorbar="sd", color=colors(ip+3))
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(p)
        ax.legend().remove()
    fig.suptitle(filter.__name__, y=0.92, fontsize='xx-large')



def get_pvalues(sess,filter,clean=True,filter_engaged=True):
    store=store_params(sess,filter,clean,filter_engaged)
    alpha = 0.05

    for i,param in enumerate(['mu', 'sigma', 'lapse_rate', 'guess_rate','dprime','criterion']):
        p_value = stats.f_oneway(store[:,i,0],store[:,i,1],store[:,i,2])[1]
        print(p_value)
        if p_value < alpha:
            tukey=stats.tukey_hsd(store[:,i,0],store[:,i,1],store[:,i,2])
            print(param + ':')
            print(f"low vs medium: p = {tukey.pvalue[0,1]}")
            print(f"medium vs high: p = {tukey.pvalue[1,2]}")
            print(f"low vs high: p = {tukey.pvalue[0,2]}\n")
            
from statsmodels.stats.anova import AnovaRM

def RM_pvalues(sess,filter,clean=True,filter_engaged=True):
    store=storedf(sess,filter,clean,filter_engaged)
    alpha = 0.05

    for i,param in enumerate(['mu', 'sigma', 'lapse_rate', 'guess_rate','dprime','criterion']):
        res = AnovaRM(data=store, depvar=param, subject='session', within=['level']).fit()
        print(param,':')
        p_value=res.anova_table.loc['level', 'Pr > F']
        print(f"p = {'%.5f' % p_value}")
        if p_value < alpha:
            high,medium,low=(store[store['level'] == level][param] for level in ['high', 'medium', 'low'])
            print("low vs medium: p = ", '%.4f' % stats.ttest_rel(low,medium).pvalue*3)
            print("medium vs high: p = ", '%.4f' % stats.ttest_rel(medium,high).pvalue*3)
            print("low vs high: p = ", '%.4f' % stats.ttest_rel(low,high).pvalue*3)
        print()


#     #   # # # #     # # #   # # # #
#     #      #      #            #
#     #      #      #            #
# # # #      #        # #        #
#     #      #            #      #
#     #      #            #      #
#     #   # # # #   # # #        #


from matplotlib.ticker import MultipleLocator

def filtered_hist_means(sess,filter,clean=True,filter_engaged=True, devided=True):
    for ises,ses in enumerate(sess):
        n=3 # Number of percentiles
        df_list = []
        for i in range(n):
            trial, name = filter(ses, n, i, clean, filter_engaged)
            trial[filter.__name__] = pd.Series(dtype=str)            
            trial[filter.__name__] = name
            df_list.append(trial)

        result = pd.concat(df_list)

        fig, ax = plt.subplots()
        colors = cm.get_cmap(cmap(filter), n+1)

        if filter==run_speed: 
            color='royalblue'
            xticks=10 # or binsize=..
            units='cm/s'
        elif filter==pupil_size:
            color='orangered'
            xticks=20 # or binsize=..
            units='pixels'
        
        if devided:
            darkened_colors = [colors(i, alpha=0.8) for i in range(1,n+1)]
            sns.histplot(data=result, x=result.columns[-2], ax=ax, bins=30, hue=filter.__name__, palette=darkened_colors, multiple="stack") #binwidth=binsize, 
            ax.xaxis.set_major_locator(MultipleLocator(xticks)) # the more ticks, the more sparse the values are
        else:
            sns.histplot(data=result, x=result.columns[-2], bins=30, ax=ax, color=color, multiple="stack") #binwidth=binsize, 

        ax.set_xlabel(f"{filter.__name__.replace('_',' ')} ({units})")
        ax.set_title(f'Mean {filter.__name__.replace('_',' ')} per trial ({ses.trialdata['session_id'][0]})')
        
        #fig.savefig(ses.trialdata['session_id'][0])
        

def histogram(sess, filter,clean=False, filter_engaged=False): #filter should be pupil_size or run_speed
    meanlist=[]
    for ises,ses in enumerate(sess):
        if clean:
            trial,behavior,video=clean_data(ses,filter_engaged)
        else:
            trial=ses.trialdata.copy()
            behavior=ses.behaviordata.copy()
            video=ses.videodata.copy()
            if filter_engaged:
                trial = trial[trial['engaged']==1].reset_index(drop=True)
                tend=trial.iloc[-1]['tEnd']
                behavior=behavior.loc[behavior['ts']<tend].reset_index(drop=True)
                video=video.loc[video['ts']<tend].reset_index(drop=True)
        
        fig, ax = plt.subplots()
        if filter==run_speed:
            #n, bins, patches = ax.hist(ses.behaviordata['runspeed'], 50, density=True)
            sns.histplot(data=behavior, x='runspeed', ax=ax, color='royalblue')
            ax.set_xlabel('Run speed (cm/s)')
            #ax.set_ylabel('Frequency')
            mean=behavior['runspeed'].mean()
            meanlist.append(mean)

        if filter==pupil_size:
            #n, bins, patches = ax.hist(ses.videodata['pupil_area'], 50, colors='orangered', density=True)
            sns.histplot(data=video, x='pupil_area', ax=ax, color='orangered')
            ax.set_xlabel('Pupil size (pixels)')
            #ax.set_ylabel('Frequency')
            meanlist.append(video['pupil_area'].mean())
        ax.set_title(f'{ses.trialdata['session_id'][0]}')
    print('Max mean:',max(meanlist))
    print('Min mean:',min(meanlist))
        




  # # #     # # #     # # #     # # # 
#         #       #   #     #   #     #   
#         #       #   #     #   #     #   
#         #       #   # # #     # # #
#         #       #   #  #      #  #
#         #       #   #    #    #    #
  # # #     # # #     #     #   #     #

def means_per_trial(trial,behavior,video):
    start=-20
    end=5
    
    trial['meanSpeed'] = pd.Series(dtype=float)
    trial['meanPupil'] = pd.Series(dtype=float)
    trial['meanMotion'] = pd.Series(dtype=float)
    for i in range((trial['trialNumber'].min()), (trial['trialNumber'].max()+1)):
        if i not in trial['trialNumber'].values:
            continue
        stimstart = trial.loc[trial['trialNumber']==i]['stimStart_k'].values[0]
        zbefore = stimstart + start
        zafter  = stimstart + end
        meanspeed=behavior.loc[((behavior['trialNumber']==i) & behavior['zpos_k'].between(zbefore, zafter)), 'runspeed'].mean()
        trial.loc[trial['trialNumber']==i, 'meanSpeed']=meanspeed
        tbefore = behavior.loc[((behavior['trialNumber']==i) & (behavior['zpos_k']>(stimstart -20))), 'ts'].min() #Not exactly -20, but almost
        tafter  = behavior.loc[((behavior['trialNumber']==i) & (behavior['zpos_k']<(stimstart +5))), 'ts'].max()
        meanpupil=video.loc[video['ts'].between(tbefore, tafter), 'pupil_area'].mean()
        meanmotion=video.loc[video['ts'].between(tbefore, tafter), 'motionenergy'].mean()
        trial.loc[trial['trialNumber']==i, 'meanPupil']=meanpupil
        trial.loc[trial['trialNumber']==i, 'meanMotion']=meanmotion

    return trial

# correlation between trial number, mean pupil size and mean running speed per trial
def corr_means_allsess(sess,clean=True,filter_engaged=True):
    list_trials=[]
    for ises,ses in enumerate(sess):
        if clean:
            trial,behavior,video=clean_data(ses,filter_engaged)
        else:
            trial=ses.trialdata.copy()
            behavior=ses.behaviordata.copy()
            video=ses.videodata.copy()
            if filter_engaged:
                trial = trial[trial['engaged']==1].reset_index(drop=True)

        trial=means_per_trial(trial,behavior,video)
        
        trial=trial.dropna(subset=['meanPupil','meanSpeed'])
        trial['meanPupil']=stats.zscore(trial['meanPupil'])
        trial['meanSpeed']=stats.zscore(trial['meanSpeed'])
        list_trials.append(trial)

    all_trials = pd.concat(list_trials)
    sns.pairplot(data=all_trials[['trialNumber','meanPupil','meanSpeed','trialOutcome']], hue="trialOutcome", 
                 palette={'HIT':'g','CR':'b','FA':'r','MISS':'orange'}, hue_order=['HIT','CR','FA','MISS'], 
                 plot_kws=dict(marker="o", s=5))
    
    fig,ax=plt.subplots()
    corr= all_trials[['trialNumber','meanPupil','meanSpeed']].corr(numeric_only=True)
    print(corr)
    print('pup vs sp',stats.pearsonr(all_trials['meanPupil'],all_trials['meanSpeed']),
    'tn vs sp',stats.pearsonr(all_trials['meanSpeed'],all_trials['trialNumber']),
    'pup vs tn',stats.pearsonr(all_trials['meanPupil'],all_trials['trialNumber']))
    sns.heatmap(corr,vmin=-1,vmax=1,ax=ax,cmap='bwr')


def plot_corr_pupil_trial(sess,clean=True,filter_engaged=True):
    list_trials=[]
    for ises,ses in enumerate(sess):
        if clean:
            trial,behavior,video=clean_data(ses,filter_engaged)
        else:
            trial=ses.trialdata.copy()
            behavior=ses.behaviordata.copy()
            video=ses.videodata.copy()
            if filter_engaged:
                trial = trial[trial['engaged']==1].reset_index(drop=True)

        trial=means_per_trial(trial,behavior,video)
        
        trial=trial.dropna(subset='meanPupil')
        trial['meanPupil']=stats.zscore(trial['meanPupil'])
        list_trials.append(trial)

    all_trials = pd.concat(list_trials)

    lm=sns.lmplot(data=all_trials, x='trialNumber', y='meanPupil', 
                  scatter_kws={'s': 7, 'color':'tomato'},line_kws={'color':'crimson'})
    lm.fig.suptitle(f"Correlation between Trial Number and Pupil Size")
    lm.set_axis_labels('Trial Number', 'Mean pupil size per trial (z-score)')
    plt.subplots_adjust(top=0.93) #antes: 95


    res = stats.linregress(all_trials['trialNumber'], all_trials['meanPupil'])
    print(f'Slope: {res.slope}')
    print(f'Correlation Coefficient (r_value): {res.rvalue}')
    print(f'p-value: {res.pvalue}')
    print(f'Standard Error: {res.stderr}')


from sklearn.linear_model import LogisticRegression

def log_reg(x,y):
    
    # Create linear regression object
    regr = LogisticRegression() # C=0.05

    # Fit the model to dataset
    regr.fit(x, y)
    
    coefs=regr.coef_
    ypred=regr.predict(x)
    r=regr.score(x,y)
    # print("Coefficients: \n", regr.coef_)
    # print("R-squared: \n", regr.score(x,y))
    return coefs, r, ypred
    

def plot_reg_coefs(sess,clean=True,filter_engaged=True):
    predictors=['signal','trialNumber','meanPupil','meanSpeed','meanMotion']
    coefs_df=pd.DataFrame(columns=['session_id']+ predictors)
    r_list=[]
    for ises,ses in enumerate(sess):
        if clean:
            trial,behavior,video=clean_data(ses,filter_engaged)
        else:
            trial=ses.trialdata.copy()
            behavior=ses.behaviordata.copy()
            video=ses.videodata.copy()
            if filter_engaged:
                trial = trial[trial['engaged']==1].reset_index(drop=True)
        
        trial=means_per_trial(trial,behavior,video)
        noise=trial.loc[(trial['signal']!=100) & (trial['signal']!=0)].copy()
        
        noise=noise.dropna(subset=['meanPupil','meanMotion','meanSpeed'], ignore_index=True)

        for predictor in predictors:
            if predictor == 'trialNumber':
                continue
            noise[predictor]  = stats.zscore(noise[predictor])
        
        noise['trialNumber']=(noise['trialNumber']-noise['trialNumber'].min())/noise['trialNumber'].max()
        
        x= noise[predictors]
        y= noise['lickResponse']
        coefs, r, ypred = log_reg(x,y) # coefs --> ndarray of shape (1, n_features)
        #print(ypred)
        prediction = pd.concat([x,y,pd.DataFrame(ypred,columns=['predResponse'])],axis=1)

        coefs_df.loc[len(coefs_df.index)] = [trial['session_id'][0]]+coefs[0].tolist()
        r_list.append(r)

    df_long = pd.melt(coefs_df, id_vars=['session_id'], var_name='parameter', value_name='coefficient')

    plt.figure(figsize=(6, 6))

    # Bar plot
    sns.barplot(x='parameter', y='coefficient', data=df_long, 
                errorbar='sd', capsize=0.1, 
                palette='Set2', hue='parameter', legend=False)

    # Strip plot (data points)
    sns.stripplot(x='parameter', y='coefficient', data=df_long, 
                  color='black',hue='session_id', palette='RdPu', alpha=0.7, jitter=True)

    plt.axhline(0, color='black', linewidth=0.5)

    # Add title and labels
    plt.title('Multiple Logistic Regression Coefficients')
    plt.xlabel('Predictor')
    plt.ylabel('Coefficient')
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x='parameter', y='coefficient', data=df_long)
    print('Mean R-squared:',np.mean(r_list))
    

def plot_r_squared_differences(sess, clean=True, filter_engaged=True):
    predictors = ['signal', 'trialNumber', 'meanPupil', 'meanSpeed', 'meanMotion']
    r_diff_df = pd.DataFrame(columns=['session_id'] + predictors)
    
    for ises, ses in enumerate(sess):
        if clean:
            trial, behavior, video = clean_data(ses, filter_engaged)
        else:
            trial = ses.trialdata.copy()
            behavior = ses.behaviordata.copy()
            video = ses.videodata.copy()
            if filter_engaged:
                trial = trial[trial['engaged'] == 1].reset_index(drop=True)
        
        trial = means_per_trial(trial, behavior, video)
        noise = trial.loc[(trial['signal'] != 100) & (trial['signal'] != 0)].copy()
        noise = noise.dropna(subset=['meanPupil', 'meanMotion', 'meanSpeed'], ignore_index=True)
        
        for predictor in predictors:
            if predictor == 'trialNumber':
                continue
            noise[predictor] = stats.zscore(noise[predictor])
        
        noise['trialNumber'] = (noise['trialNumber'] - noise['trialNumber'].min()) / noise['trialNumber'].max()
        
        x = noise[predictors]
        y = noise['lickResponse']
        
        _, r_full, _ = log_reg(x, y)  # R-squared with all predictors
        
        r_diffs = []
        for predictor in predictors:
            x_partial = x.drop(columns=[predictor])
            _, r_partial, _ = log_reg(x_partial, y)
            r_diff = r_full - r_partial
            r_diffs.append(r_diff)
        
        r_diff_df.loc[len(r_diff_df.index)] = [trial['session_id'][0]] + r_diffs
    
    df_long = pd.melt(r_diff_df, id_vars=['session_id'], var_name='parameter', value_name='r_squared_difference')
    
    plt.figure(figsize=(6, 6))
    
    sns.barplot(x='parameter', y='r_squared_difference', data=df_long, 
                errorbar='sd', capsize=0.1, 
                palette='Set2', hue='parameter', legend=False)
    
    sns.stripplot(x='parameter', y='r_squared_difference', data=df_long, 
                  color='black', alpha=0.7, jitter=True)
    
    plt.axhline(0, color='black', linewidth=0.5)
    
    plt.title('R-squared Difference by Predictor Removal')
    plt.xlabel('Removed Predictor')
    plt.ylabel('R-squared Difference')



# # # # #   # # #     
    #       #     #
    #       #     #
    #       # # #
    #       #   #
    #       #     #

# plot pupil and speed through whole session
def traces(sess,clean=True,filter_engaged=True):
    for ises,ses in enumerate(sess):
        if clean:
            trial,behavior,video=clean_data(ses,filter_engaged)
        else:
            trial=ses.trialdata.copy()
            behavior=ses.behaviordata.copy()
            video=ses.videodata.copy()
            if filter_engaged:
                trial = trial[trial['engaged']==1].reset_index(drop=True)
                tend=trial.iloc[-1]['tEnd']
                behavior=behavior.loc[behavior['ts']<tend].reset_index(drop=True)
                video=video.loc[video['ts']<tend].reset_index(drop=True)

            # make ts start at 0c
            video['ts']=(video['ts']-video.iloc[0]['ts'])
            behavior['ts']=(behavior['ts']-behavior.iloc[0]['ts'])

        
        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(9, 5))


        x1=behavior['ts'].to_numpy()
        y1=behavior['runspeed'].to_numpy()
        ax1.plot(x1,y1, 'royalblue', linewidth=0.5)
        ax1.set_ylabel('Run speed (cm/s)')
        ax1.set_title(ses.trialdata['session_id'][0])

        #z=stats.zscore(video['pupil_area'], nan_policy='omit')
        
        x2=video['ts'].to_numpy() 
        y2=video['pupil_area'].to_numpy()
        ax2.plot(x2,y2, 'orangered', linewidth=0.5)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('Pupil size')
        ax2.set_ylim([0,np.nanmax(y2)*1.1])
        ax2.set_xlim([np.min(x2),np.max(x2)])

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        #fig.savefig(ses.trialdata['session_id'][0])


def expand_traces(sess,clean=True,filter_engaged=True):
    window=[830,860]
    for ises,ses in enumerate(sess):
        if clean:
            trial,behavior,video=clean_data(ses,filter_engaged)
        else:
            trial=ses.trialdata.copy()
            behavior=ses.behaviordata.copy()
            video=ses.videodata.copy()
            if filter_engaged:
                trial = trial[trial['engaged']==1].reset_index(drop=True)
                tend=trial.iloc[-1]['tEnd']
                behavior=behavior.loc[behavior['ts']<tend].reset_index(drop=True)
                video=video.loc[video['ts']<tend].reset_index(drop=True)

            # make ts start at 0c
            video['ts']=(video['ts']-video.iloc[0]['ts'])
            behavior['ts']=(behavior['ts']-behavior.iloc[0]['ts'])

        
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9, 5))

        speed_data=behavior.loc[behavior['ts'].between(*window), ['ts','runspeed']]
        x1=speed_data['ts'].to_numpy()
        y1=speed_data['runspeed'].to_numpy()
        ax1.plot(x1,y1, 'royalblue')
        ax1.set_ylabel('Run speed (cm/s)')
        ax1.set_title(ses.trialdata['session_id'][0])

        pupil_data=video.loc[video['ts'].between(*window), ['ts','pupil_area']]
        x2=pupil_data['ts'].to_numpy() 
        y2=pupil_data['pupil_area'].to_numpy()
        ax2.plot(x2,y2, 'orangered')
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('Pupil size (% of mean)')
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        #fig.savefig(ses.trialdata['session_id'][0])


def trace_motion(sess,clean=True,filter_engaged=True):
    for ises,ses in enumerate(sess):
        if clean:
            trial,behavior,video=clean_data(ses,filter_engaged)
        else:
            trial=ses.trialdata.copy()
            video=ses.videodata.copy()
            if filter_engaged:
                trial = trial[trial['engaged']==1].reset_index(drop=True)
                tend=trial.iloc[-1]['tEnd']
                video=video.loc[video['ts']<tend].reset_index(drop=True)

            # make ts start at 0c
            video['ts']=(video['ts']-video.iloc[0]['ts'])

        fig, ax = plt.subplots(figsize=(10, 3))

        x2=video['ts'].to_numpy() 
        y2=video['motionenergy'].to_numpy()
        ax.plot(x2,y2, 'orangered', linewidth=0.5)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Motion Energy')
        ax.set_ylim([0,np.nanmax(y2)*1.1])
        ax.set_xlim([np.min(x2),np.max(x2)])

        #fig.savefig(ses.trialdata['session_id'][0])



# # #        # # #    # # # # #    #       #
#     #    #              #        #       #
#     #    #              #        #       #
# # #        # #          #        # # # # #
#                #        #        #       #  
#                #        #        #       #
#          # # #          #        #       #


def calc_pupilPSTH(ses, filter_engaged, zscore, s_pre = -80, s_post = 60, binsize = 5):
    trial=ses.trialdata.copy()
    video=ses.videodata.copy()
    if filter_engaged:
        trial = trial[trial['engaged']==1].reset_index(drop=True)

    # pupil z-score
    if zscore:
        video['zscore']=stats.zscore(video['pupil_area'], nan_policy='omit')
    
    binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(s_pre,s_post+binsize,binsize)

    ntrials     = len(trial)
    pupilPSTH   = np.empty(shape=(ntrials, len(bincenters)))

    # Add trial Number to videodata
    video['trialNumber']=pd.Series(dtype=int)
    for i in range(ntrials):
        tstart=trial['tStart'][i]
        tend=trial['tEnd'][i]
        video.loc[video['ts'].between(tstart,tend), 'trialNumber']=i+1

    
    for itrial in range(ntrials):
        idx = np.logical_and(itrial-1 <= video['trialNumber'], video['trialNumber'] <= itrial+2)
        pupilPSTH[itrial,:] = stats.binned_statistic(video['zpos'][idx]-trial['stimStart'][itrial],
                                                    video['zscore'][idx], statistic='mean', bins=binedges)[0]
        
    return pupilPSTH, bincenters


def plot_pupil_corridor_outcome(sess, filter_engaged=True, zscore=True):
    ### Plot run speed as a function of trial type:
    for ises,ses in enumerate(sess):
        
        trial=ses.trialdata.copy()
        if filter_engaged:
            trial = trial[trial['engaged']==1].reset_index(drop=True)

        pupilPSTH,bincenters=calc_pupilPSTH(ses, filter_engaged,zscore)
        
        fig, ax = plt.subplots()
        
        ttypes = pd.unique(trial['trialOutcome'])
        colors = {'HIT':'g','MISS':'orange','FA':'r','CR':'blue'}

        for i,ttype in enumerate(ttypes):
            idx = trial['trialOutcome']==ttype
            data_mean = np.nanmean(pupilPSTH[idx,:],axis=0)
            data_error = np.nanstd(pupilPSTH[idx,:],axis=0)# / math.sqrt(sum(idx))
            ax.plot(bincenters,data_mean,label=ttype,color=colors[ttype],linewidth=2)
            ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[ttype])

        rewzonestart = np.mean(trial['rewardZoneStart'] - trial['stimStart'])
        rewzonelength = np.mean(trial['rewardZoneEnd'] - trial['rewardZoneStart'])

        ax.legend()
        ax.set_ylim(0,1)
        ax.set_xlim(bincenters[0],rewzonestart+rewzonelength)
        ax.set_xlabel('Position rel. to stimulus onset (cm)')
        ax.set_ylabel('Pupil size (normalized per trial)')
        ax.set_title(f"{trial['session_id'][0]}")
        ax.add_patch(matplotlib.patches.Rectangle((0,0),20,1, 
                                fill = True, alpha=0.2,
                                color = "blue",
                                linewidth = 0))
        ax.add_patch(matplotlib.patches.Rectangle((rewzonestart,0),rewzonelength,1, 
                                fill = True, alpha=0.2,
                                color = "green",
                                linewidth = 0))

        plt.text(5, 0.9, 'Stim',fontsize=12)
        plt.text(25, 0.9, 'Rew',fontsize=12)

        #return fig
        fig.savefig(trial['session_id'][0])

def plot_pupil_corridor_outcome_allsess(sess, filter_engaged=True):
    ### Plot run speed as a function of trial type:
    list_trials = []
    list_PSTH   = []
    for ises,ses in enumerate(sess):
        
        trial=ses.trialdata.copy()
        if filter_engaged:
            trial = trial[trial['engaged']==1].reset_index(drop=True)

        pupilPSTH,bincenters=calc_pupilPSTH(ses, filter_engaged, zscore=True)
        unit='z-score'
    
        list_trials.append(trial)
        list_PSTH.append(pupilPSTH)

    all_trials = pd.concat(list_trials, ignore_index=True)
    all_PSTH   = np.concatenate(list_PSTH, axis=0)


    fig, ax = plt.subplots()
    
    ttypes = pd.unique(trial['trialOutcome'])
    colors = {'HIT':'g','MISS':'orange','FA':'r','CR':'blue'}

    for i,ttype in enumerate(ttypes):
        idx = all_trials['trialOutcome']==ttype
        data_mean = np.nanmean(all_PSTH[idx,:],axis=0)
        data_error = np.nanstd(all_PSTH[idx,:],axis=0)# / math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype,color=colors[ttype],linewidth=2)
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[ttype])

    rewzonestart = 25 # or np.mean(all_trials['rewardZoneStart'] - all_trials['stimStart']) 
    rewzonelength = 20 # or np.mean(all_trials['rewardZoneEnd'] - all_trials['rewardZoneStart']) 

    ax.legend()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(bincenters[0],bincenters[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel(f'Pupil size ({unit})')
    ax.set_title(f"All sessions")
    ax.add_patch(matplotlib.patches.Rectangle((0,-1.5),20,3, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((rewzonestart,-1.5),rewzonelength,3, 
                            fill = True, alpha=0.2,
                            color = "green",
                            linewidth = 0))

    plt.text(5, 1.3, 'Stim',fontsize=12)
    plt.text(30, 1.3, 'Reward',fontsize=12)



def calc_normmotionPSTH(ses, filter_engaged,s_pre = -80, s_post = 60, binsize = 5):
    trial=ses.trialdata.copy()
    video=ses.videodata.copy()
    if filter_engaged:
        trial = trial[trial['engaged']==1].reset_index(drop=True)

    
    ## Parameters for spatial binning
    # s_pre       #pre cm
    # s_post      #post cm
    # binsize     #spatial binning in cm
    binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(s_pre,s_post+binsize,binsize)

    
    ntrials     = len(trial)
    pupilPSTH   = np.empty(shape=(ntrials, len(bincenters)))

    # Normalize motion energy and add trial number to videodata
    video['trialNumber']=pd.Series(dtype=int)
    for i in range(ntrials):
        tstart=trial['tStart'][i]
        tend=trial['tEnd'][i]
        trial_motion=video.loc[video['ts'].between(tstart,tend), 'motionenergy']
        video.loc[video['ts'].between(tstart,tend), 'motionenergy']=(trial_motion-trial_motion.min())/trial_motion.max()
        video.loc[video['ts'].between(tstart,tend), 'trialNumber']=i+1
        

    for itrial in range(ntrials):
        idx = np.logical_and(itrial-1 <= video['trialNumber'], video['trialNumber'] <= itrial+2)
        pupilPSTH[itrial,:] = stats.binned_statistic(video['zpos'][idx]-trial['stimStart'][itrial],
                                                    video['motionenergy'][idx], statistic='mean', bins=binedges)[0]
        
        
    return pupilPSTH, bincenters


def calc_zmotionPSTH(ses, filter_engaged,s_pre = -80, s_post = 60, binsize = 5):
    trial=ses.trialdata.copy()
    video=ses.videodata.copy()
    if filter_engaged:
        trial = trial[trial['engaged']==1].reset_index(drop=True)

    # z-score
    video['zscore']=stats.zscore(video['motionenergy'], nan_policy='omit')

    binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(s_pre,s_post+binsize,binsize)

    
    ntrials     = len(trial)
    motionPSTH   = np.empty(shape=(ntrials, len(bincenters)))

    video['trialNumber']=pd.Series(dtype=int)
    for i in range(ntrials):
        tstart=trial['tStart'][i]
        tend=trial['tEnd'][i]
        video.loc[video['ts'].between(tstart,tend), 'trialNumber']=i+1

    
    for itrial in range(ntrials):
        idx = np.logical_and(itrial-1 <= video['trialNumber'], video['trialNumber'] <= itrial+2)
        motionPSTH[itrial,:] = stats.binned_statistic(video['zpos'][idx]-trial['stimStart'][itrial],
                                                    video['zscore'][idx], statistic='mean', bins=binedges)[0]
        
    return motionPSTH, bincenters



def plot_motion_corridor_outcome_allsess(sess, filter_engaged=True,normalized=False):
    ### Plot run speed as a function of trial type:
    list_trials = []
    list_PSTH   = []
    for ises,ses in enumerate(sess):
        
        trial=ses.trialdata.copy()
        if filter_engaged:
            trial = trial[trial['engaged']==1].reset_index(drop=True)

        if normalized:
            motionPSTH,bincenters=calc_normmotionPSTH(ses, filter_engaged)
            unit='normalized per trial'
        else:
            motionPSTH,bincenters=calc_zmotionPSTH(ses, filter_engaged)
            unit='z-score'
        
        list_trials.append(trial)
        list_PSTH.append(motionPSTH)

    all_trials = pd.concat(list_trials, ignore_index=True)
    all_PSTH   = np.concatenate(list_PSTH, axis=0)


    fig, ax = plt.subplots()
    
    ttypes = pd.unique(trial['trialOutcome'])
    colors = {'HIT':'g','MISS':'orange','FA':'r','CR':'blue'}

    for i,ttype in enumerate(ttypes):
        idx = all_trials['trialOutcome']==ttype
        data_mean = np.nanmean(all_PSTH[idx,:],axis=0)
        data_error = np.nanstd(all_PSTH[idx,:],axis=0)# / math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype,color=colors[ttype],linewidth=2)
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[ttype])

    rewzonestart = np.mean(all_trials['rewardZoneStart'] - all_trials['stimStart']) # or just 25
    rewzonelength = np.mean(all_trials['rewardZoneEnd'] - all_trials['rewardZoneStart']) # or just 20

    ax.legend()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(bincenters[0],bincenters[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel(f'Motion Energy ({unit})')
    ax.set_title(f"All sessions")
    ax.add_patch(matplotlib.patches.Rectangle((0,-1.5),20,3, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((rewzonestart,-1.5),rewzonelength,3, 
                            fill = True, alpha=0.2,
                            color = "green",
                            linewidth = 0))

    plt.text(5, 1.3, 'Stim',fontsize=12)
    plt.text(30, 1.3, 'Reward',fontsize=12)

 


  # # #   # # # #     # # #
#         #         #       
#         #         #
  # #     # # #       # #
      #   #               #
      #   #               #
# # #     # # # #   # # #

from loaddata.session_info import filter_sessions,load_sessions

#################################################

#session_list        = np.array([['LPE12013','2024_04_22'],['LPE11998','2024_04_22'],['LPE11622','2024_02_19']])
#sessions,nSessions  = load_sessions(protocol = 'DP',session_list=session_list,load_behaviordata=True, 
#                                    load_calciumdata=False, load_videodata=True)

sessions,nSessions  = filter_sessions(protocols = ['DP','DN'],load_behaviordata=True, min_trials=200,
                                    load_calciumdata=False, load_videodata=True,has_pupil=True)

