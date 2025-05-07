
import numpy as np
import copy
from scipy.stats import zscore
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LOGR
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm as SVM
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from scipy.stats import zscore, wilcoxon, ranksums, ttest_rel
from matplotlib.lines import Line2D

from utils.dimreduc_lib import *
from utils.rf_lib import filter_nearlabeled
from utils.plot_lib import *

def find_optimal_lambda(X,y,model_name='LOGR',kfold=5,clip=False):
    if model_name == 'LogisticRegression':
        model_name = 'LOGR'
    assert len(X.shape)==2, 'X must be a matrix of samples by features'
    assert len(y.shape)==1, 'y must be a vector'
    assert X.shape[0]==y.shape[0], 'X and y must have the same number of samples'
    # assert model_name in ['LOGR','SVM','LDA'], 'regularization not supported for model %s' % model_name
    assert model_name in ['LOGR','SVM','LDA','Ridge','Lasso','LinearRegression'], 'regularization not supported for model %s' % model_name

    # Define the k-fold cross-validation object
    kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

    # Initialize an array to store the decoding performance for each fold
    fold_performance = np.zeros((kfold,))

    # Find the optimal regularization strength (lambda)
    lambdas = np.logspace(-4, 4, 20)
    cv_scores = np.zeros((len(lambdas),))
    for ilambda, lambda_ in enumerate(lambdas):
        
        if model_name == 'LOGR':
            model = LOGR(penalty='l1', solver='liblinear', C=lambda_)
            score_fun = 'accuracy'
        elif model_name == 'SVM':
            model = SVM.SVC(kernel='linear', C=lambda_)
            score_fun = 'accuracy'
        elif model_name == 'LDA':
            n_components = np.unique(y).size-1
            model = LDA(n_components=n_components,solver='eigen', shrinkage=np.clip(lambda_,0,1))
            score_fun = 'accuracy'
        elif model_name in ['Ridge', 'Lasso']:
            model = getattr(sklearn.linear_model,model_name)(alpha=lambda_)
            score_fun = 'r2'
        elif model_name in ['ElasticNet']:
            model = getattr(sklearn.linear_model,model_name)(alpha=lambda_,l1_ratio=0.9)
            score_fun = 'r2'

        scores = cross_val_score(model, X, y, cv=kf, scoring=score_fun)
        cv_scores[ilambda] = np.mean(scores)
    optimal_lambda = lambdas[np.argmax(cv_scores)]
    # print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))
    if clip:
        optimal_lambda = np.clip(optimal_lambda, 0.03, 166)
    # optimal_lambda = 1
    return optimal_lambda

def my_decoder_wrapper(Xfull,Yfull,model_name='LOGR',kfold=5,lam=None,subtract_shuffle=True,
                          scoring_type=None,norm_out=False,n_components=None):
    if model_name == 'LogisticRegression':
        model_name = 'LOGR'
    assert len(Xfull.shape)==2, 'Xfull must be a matrix of samples by features'
    assert len(Yfull.shape)==1, 'Yfull must be a vector'
    assert Xfull.shape[0]==Yfull.shape[0], 'Xfull and Yfull must have the same number of samples'
    assert model_name in ['LOGR','SVM','LDA','Ridge','Lasso','LinearRegression'], 'regularization not supported for model %s' % model_name
    assert lam is None or lam > 0
    
    if lam is None:
        lam = find_optimal_lambda(Xfull,Yfull,model_name=model_name,kfold=kfold)

    if model_name == 'LOGR':
        model = LOGR(penalty='l1', solver='liblinear', C=lam)
    elif model_name == 'SVM':
        model = SVM.SVC(kernel='linear', C=lam)
    elif model_name == 'LDA':
        if n_components is None: 
            n_components = np.unique(Yfull).size-1
        model = LDA(n_components=n_components,solver='eigen', shrinkage=np.clip(lam,0,1))
        # model = LDA(n_components=n_components,solver='svd')
    elif model_name == 'GBC': #Gradient Boosting Classifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=10, random_state=0,max_features='sqrt')
    elif model_name in ['Ridge', 'Lasso']:
        model = getattr(sklearn.linear_model,model_name)(alpha=lam)
    elif model_name in ['ElasticNet']:
        model = getattr(sklearn.linear_model,model_name)(alpha=lam,l1_ratio=0.9)

    if scoring_type is None:
        scoring_type = 'accuracy_score' if model_name in ['LOGR','SVM','LDA','GBC'] else 'r2_score'
    score_fun           = getattr(sklearn.metrics,scoring_type)

    # Define the number of folds for cross-validation
    kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

    # Initialize an array to store the decoding performance
    performance         = np.full((kfold,), np.nan)
    performance_shuffle = np.full((kfold,), np.nan)
    # weights             = np.full((kfold,np.shape(Xfull)[1]), np.nan) #deprecated, estimate weights from all data, not cv
    projs               = np.full((np.shape(Xfull)[0]), np.nan)

    # Loop through each fold
    for ifold, (train_index, test_index) in enumerate(kf.split(Xfull)):
        # Split the data into training and testing sets
        X_train, X_test = Xfull[train_index], Xfull[test_index]
        y_train, y_test = Yfull[train_index], Yfull[test_index]

        # Train a classification model on the training data with regularization
        model.fit(X_train, y_train)

        # weights[ifold,:] = model.coef_ #deprecated, estimate weights from all data, not cv

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate the decoding performance for this fold
        performance[ifold] = score_fun(y_test, y_pred)
        projs[test_index] = y_pred

        if subtract_shuffle:
            # Shuffle the labels and calculate the decoding performance for this fold
            np.random.shuffle(y_train)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            performance_shuffle[ifold] = score_fun(y_test, y_pred)

    if subtract_shuffle: # subtract the shuffling performance from the average perf
        performance_avg = np.mean(performance - performance_shuffle)
    else: # Calculate the average decoding performance across folds
        performance_avg = np.mean(performance)
    if norm_out: # normalize to maximal range of performance (between shuffle and 1)
        performance_avg = performance_avg / (1-np.mean(performance_shuffle))

    #Estimate the weights from the entire dataset:
    model.fit(Xfull,Yfull)
    weights = model.coef_.ravel()

    if np.shape(Xfull)[1] == np.shape(weights)[0]:
    # if len(np.unique(Yfull)) == 2:
        ev      = var_along_dim(Xfull,weights)
    else:
        ev = None
        # ev      = var_along_dim(Xfull,weights)

    return performance_avg,weights,projs,ev

def prep_Xpredictor(X,y):
    X           = zscore(X, axis=0,nan_policy='omit')
    idx_nan     = ~np.all(np.isnan(X),axis=1)
    X           = X[idx_nan,:]
    y           = y[idx_nan]
    X[:,np.all(np.isnan(X),axis=0)] = 0
    X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
    y           = np.nan_to_num(y,nan=np.nanmean(y,axis=0,keepdims=True))
    return X,y,idx_nan

def balance_trial(X,y,sample_min_trials=20):
    N0,N1 = np.sum(y==0),np.sum(y==1)
    mintrials =  np.min([N0,N1])
    if mintrials < sample_min_trials:
        idx0 = np.random.choice(np.where(y==0)[0],size=sample_min_trials,replace=True)
        idx1 = np.random.choice(np.where(y==1)[0],size=sample_min_trials,replace=True)
        yb = np.concatenate((y[idx0],y[idx1]))
        Xb = np.concatenate((X[idx0,:],X[idx1,:]))
    else: 
        idx0 = np.random.choice(np.where(y==0)[0],size=mintrials,replace=False)
        idx1 = np.random.choice(np.where(y==1)[0],size=mintrials,replace=False)
        yb  = np.concatenate((y[idx0],y[idx1]))
        Xb  = np.concatenate((X[idx0,:],X[idx1,:]))
    return Xb,yb    

# def prep_Xpredictor(X,y):
#     X           = X[:,~np.all(np.isnan(X),axis=0)] #
#     idx_nan     = ~np.all(np.isnan(X),axis=1)
#     X           = X[idx_nan,:]
#     y           = y[idx_nan]
#     X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
#     X           = zscore(X, axis=0)
#     X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
#     return X,y

######  #######  #####     #     #    #    ######     #     # ######     #    ######  ######  ####### ######  
#     # #       #     #    #     #   # #   #     #    #  #  # #     #   # #   #     # #     # #       #     # 
#     # #       #          #     #  #   #  #     #    #  #  # #     #  #   #  #     # #     # #       #     # 
#     # #####   #          #     # #     # ######     #  #  # ######  #     # ######  ######  #####   ######  
#     # #       #           #   #  ####### #   #      #  #  # #   #   ####### #       #       #       #   #   
#     # #       #     #      # #   #     # #    #     #  #  # #    #  #     # #       #       #       #    #  
######  #######  #####        #    #     # #     #     ## ##  #     # #     # #       #       ####### #     # 

# Binary classification decoder from e.g. V1 and PM labeled and unlabeled neurons separately
def decvar_from_arealabel_wrapper(sessions,sbins,arealabels,var='noise',nmodelfits=20,kfold=5,lam=0.08,nmintrialscond=10,
                                   nminneurons=10,nsampleneurons=20,model_name='LOGR',filter_nearby=False):
    np.random.seed(49)

    nSessions       = len(sessions)
    narealabels     = len(arealabels)

    dec_perf   = np.full((len(sbins),narealabels,nSessions), np.nan)
    # Loop through each session
    for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
        neuraldata = copy.deepcopy(ses.stensor)
        if var=='noise':
            idx_T       = np.all((np.isin(ses.trialdata['stimcat'],['C','N']),
                                    # np.isin(ses.trialdata['trialOutcome'],['HIT','CR']),
                                    ses.trialdata['engaged']==1),axis=0)
            y_idx_T           = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
        elif var=='max':
            idx_T       = np.all((np.isin(ses.trialdata['stimcat'],['C','M']),
                                    # np.isin(ses.trialdata['trialOutcome'],['HIT','CR']),
                                    ses.trialdata['engaged']==1),axis=0)
            y_idx_T           = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
        elif var=='choice':
            for isig,sig in enumerate(np.unique(ses.trialdata['signal'])):
                idx_T = ses.trialdata['signal']==sig
                neuraldata[:,idx_T,:] -= np.nanmean(neuraldata[:,idx_T,:],axis=1,keepdims=True)

            #Correct setting: stimulus trials during engaged part of the session:
            idx_T       = np.all((ses.trialdata['engaged']==1,np.isin(ses.trialdata['stimcat'],['N'])), axis=0)
            y_idx_T     = ses.trialdata['lickResponse'][idx_T].to_numpy()

        for ial, arealabel in enumerate(arealabels):
            
            if filter_nearby:
                idx_nearby  = filter_nearlabeled(ses,radius=50)
            else:
                idx_nearby = np.ones(len(ses.celldata),dtype=bool)

            idx_N       = np.all((ses.celldata['arealabel']==arealabel,
                                    ses.celldata['noise_level']<20,	
                                    idx_nearby),axis=0)

            if np.sum(y_idx_T==0) >= nmintrialscond and np.sum(y_idx_T==1) >= nmintrialscond and np.sum(idx_N) >= nminneurons:
                for ibin, bincenter in enumerate(sbins):            # Loop through each spatial bin
                    temp = np.empty(nmodelfits)
                    for i in range(nmodelfits):
                        y   = copy.deepcopy(y_idx_T)

                        if np.sum(idx_N) >= nsampleneurons:
                            # idx_Ns = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                            idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                        else:
                            idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=True)

                        X       = neuraldata[np.ix_(idx_Ns,idx_T,sbins==bincenter)].squeeze()
                        X       = X.T

                        X,y,_   = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                        Xb,yb           = balance_trial(X,y,sample_min_trials=nmintrialscond)
                        # temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
                        temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name=model_name,kfold=kfold,lam=lam,
                                                                subtract_shuffle=False,norm_out=False)
                    dec_perf[ibin,ial,ises] = np.nanmean(temp)

    return dec_perf

# Decode from subpopulations separately, split by hit/miss e.g. V1 and PM labeled and unlabeled neurons
def decvar_hitmiss_from_arealabel_wrapper(sessions,sbins,arealabels,var='noise',nmodelfits=20,kfold=5,lam=0.08,nmintrialscond=10,
                                   nminneurons=10,nsampleneurons=20,filter_nearby=False):
    np.random.seed(49)

    nSessions       = len(sessions)
    narealabels     = len(arealabels)
    nlickresponse   = 2

    dec_perf   = np.full((len(sbins),narealabels,nlickresponse,nSessions), np.nan)
    # Loop through each session
    for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
        neuraldata = copy.deepcopy(ses.stensor)

        for ial, arealabel in enumerate(arealabels):
            for ilr in range(nlickresponse):

                if var=='noise':
                    idx_cat     = np.logical_or(np.logical_and(ses.trialdata['stimcat']=='N',
                                                               ses.trialdata['lickResponse']==ilr),
                                        ses.trialdata['stimcat']=='C')
                    idx_T       = np.all((idx_cat,
                                  ses.trialdata['engaged']==1),axis=0)
                    y_idx_T           = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()

                elif var=='max':
                    idx_cat     = np.logical_or(np.logical_and(ses.trialdata['stimcat']=='M',
                                                               ses.trialdata['lickResponse']==ilr),
                                        ses.trialdata['stimcat']=='C')
                    idx_T       = np.all((idx_cat,
                                  ses.trialdata['engaged']==1),axis=0)
                    y_idx_T           = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()

                if filter_nearby:
                    idx_nearby  = filter_nearlabeled(ses,radius=50)
                else:
                    idx_nearby = np.ones(len(ses.celldata),dtype=bool)
                # idx_nearby  = filter_nearlabeled(ses,radius=50)
                idx_N       = np.all((ses.celldata['arealabel']==arealabel,
                                        ses.celldata['noise_level']<20,	
                                        idx_nearby),axis=0)

                if np.sum(y_idx_T==0) >= nmintrialscond and np.sum(y_idx_T==1) >= nmintrialscond and np.sum(idx_N) >= nminneurons:
                    for ibin, bincenter in enumerate(sbins):            # Loop through each spatial bin
                        temp = np.empty(nmodelfits)
                        for i in range(nmodelfits):
                            y   = copy.deepcopy(y_idx_T)

                            if np.sum(idx_N) >= nsampleneurons:
                                # idx_Ns = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                                idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                            else:
                                idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=True)

                            X       = neuraldata[np.ix_(idx_Ns,idx_T,sbins==bincenter)].squeeze()
                            X       = X.T

                            X,y,_   = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                            Xb,yb           = balance_trial(X,y,sample_min_trials=nmintrialscond)
                            # temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
                            temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,
                                                                    subtract_shuffle=False,norm_out=False)
                        dec_perf[ibin,ial,ilr,ises] = np.nanmean(temp)

    return dec_perf

# Binary classification decoder  from V1 and PM labeled and unlabeled neurons separately
def decvar_cont_from_arealabel_wrapper(sessions,sbins,arealabels,var='signal',nmodelfits=20,kfold=5,lam=0.08,nmintrialscond=10,
                                   nminneurons=10,nsampleneurons=20,model_name='Ridge',filter_nearby=False):
    np.random.seed(49)

    nSessions       = len(sessions)
    narealabels     = len(arealabels)

    dec_perf   = np.full((len(sbins),narealabels,nSessions), np.nan)
    # Loop through each session
    for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
        neuraldata = copy.deepcopy(ses.stensor)
        if var=='signal':
            idx_T       = np.all((np.isin(ses.trialdata['stimcat'],['N']),
                        ses.trialdata['engaged']==1),axis=0)
            y_idx_T           = ses.trialdata['signal'][idx_T].to_numpy()
        
        for ial, arealabel in enumerate(arealabels):
            if filter_nearby:
                idx_nearby  = filter_nearlabeled(ses,radius=50)
            else:
                idx_nearby = np.ones(len(ses.celldata),dtype=bool)
                
            idx_N       = np.all((ses.celldata['arealabel']==arealabel,
                                    ses.celldata['noise_level']<20,	
                                    idx_nearby),axis=0)
            
            if len(y_idx_T) >= nmintrialscond and np.sum(idx_N) >= nminneurons:
                for ibin, bincenter in enumerate(sbins):            # Loop through each spatial bin
                    temp = np.empty(nmodelfits)
                    for i in range(nmodelfits):
                        y   = copy.deepcopy(y_idx_T)

                        if np.sum(idx_N) >= nsampleneurons:
                            # idx_Ns = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                            idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                        else:
                            idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=True)

                        X       = neuraldata[np.ix_(idx_Ns,idx_T,sbins==bincenter)].squeeze()
                        X       = X.T

                        X,y,_   = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                        temp[i],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,lam=lam,
                                                                subtract_shuffle=True,norm_out=True)
                    dec_perf[ibin,ial,ises] = np.nanmean(temp)

    return dec_perf


# Binary classification decoder  from V1 and PM labeled and unlabeled neurons separately
def decvar_cont_hitmiss_from_arealabel_wrapper(sessions,sbins,arealabels,var='signal',nmodelfits=20,kfold=5,lam=0.08,nmintrialscond=10,
                                   nminneurons=10,nsampleneurons=20,model_name='Ridge',filter_nearby=False):
    np.random.seed(49)

    nSessions       = len(sessions)
    narealabels     = len(arealabels)
    nlickresponse   = 2

    dec_perf   = np.full((len(sbins),narealabels,nlickresponse,nSessions), np.nan)
    # Loop through each session
    for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
        neuraldata = copy.deepcopy(ses.stensor)

        for ilr in range(nlickresponse):

            if var=='signal':
                idx_T       = np.all((np.isin(ses.trialdata['stimcat'],['N']),
                                    ses.trialdata['lickResponse']==ilr,
                                    ses.trialdata['engaged']==1),axis=0)
                y_idx_T           = ses.trialdata['signal'][idx_T].to_numpy()
            
            for ial, arealabel in enumerate(arealabels):
                if filter_nearby:
                    idx_nearby  = filter_nearlabeled(ses,radius=50)
                else:
                    idx_nearby = np.ones(len(ses.celldata),dtype=bool)
                    
                idx_N       = np.all((ses.celldata['arealabel']==arealabel,
                                        ses.celldata['noise_level']<20,	
                                        idx_nearby),axis=0)
                
                if len(y_idx_T) >= nmintrialscond and np.sum(idx_N) >= nminneurons:
                    for ibin, bincenter in enumerate(sbins):            # Loop through each spatial bin
                        temp = np.empty(nmodelfits)
                        for i in range(nmodelfits):
                            y   = copy.deepcopy(y_idx_T)

                            if np.sum(idx_N) >= nsampleneurons:
                                idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                            else:
                                idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=True)

                            X       = neuraldata[np.ix_(idx_Ns,idx_T,sbins==bincenter)].squeeze()
                            X       = X.T

                            X,y,_   = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                            temp[i],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,lam=lam,
                                                                    subtract_shuffle=True,norm_out=True)
                        dec_perf[ibin,ial,ilr,ises] = np.nanmean(temp)

    return dec_perf



# Show the decoding performance across space for the different populations:
def plot_dec_perf_arealabel(dec_perf,sbins,arealabels,clrs_arealabels,testwin=[0,25]):

    ylow = 0.45 if np.nanmin(dec_perf)>.25 else -0.05
    ymax = 1.0 if np.nanmax(dec_perf)>.75 else 0.5
    ychance = .5 if np.nanmin(dec_perf)>.25 else 0.0

    fig,axes    = plt.subplots(1,4,figsize=(10,3),sharex=False,sharey=True,gridspec_kw={'width_ratios': [3,1,3,1]})
    markersize  = 30

    statdata    = np.nanmean(dec_perf[(sbins>=testwin[0]) & (sbins<=testwin[1]),:,:],axis=0)

    nSessions   = dec_perf.shape[2]
    # statdata    = np.nanmean(dec_perf[(sbins>=11) & (sbins<=20),:,:],axis=0)
    ax = axes[0]
    handles = []
    for ial, arealabel in enumerate(arealabels[:2]):
        handles.append(shaded_error(sbins,dec_perf[:,ial,:].T,color=clrs_arealabels[ial],alpha=0.5,linewidth=1.5,error='sem',ax=ax))
        for ises in range(nSessions):
            ax.plot(sbins,dec_perf[:,ial,ises],color=clrs_arealabels[ial],linewidth=0.2)

    ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles,labels=arealabels[:2])
    ax.set_ylabel('Decoding performance')
    ax.set_xlabel('Position relative to stim (cm)')
    add_stim_resp_win(ax)
    ax.set_xticks([-50,-25,0,25,50,75])
    ax.set_ylim([ylow,ymax])
    ax.set_xlim([-50,75])

    for ibin, bincenter in enumerate(sbins):
        t,pval = ttest_rel(dec_perf[ibin,0,:],dec_perf[ibin,1,:],nan_policy='omit')
        ax.text(bincenter, ymax-0.1, '%s' % get_sig_asterisks(pval,return_ns=False), color='k', ha='center', fontsize=12)

    ax = axes[1]
    ax.plot([0,1],statdata[:2,:],color='k',linewidth=0.25)
    ax.scatter(np.zeros(nSessions),statdata[0,:],color='k',marker='.',s=markersize)
    ax.scatter(np.ones(nSessions),statdata[1,:],color='k',marker='.',s=markersize)
    ax.scatter([0,1],np.nanmean(statdata[:2,:],axis=1),color=clrs_arealabels[:2],marker='o',s=markersize*2,zorder=10)
    ax.errorbar([0,1],np.nanmean(statdata[:2,:],axis=1),np.nanstd(statdata[:2,:],axis=1)/np.sqrt(nSessions),
                color='k',capsize=0,elinewidth=2,zorder=0)

    t,pval = ttest_rel(statdata[0,:],statdata[1,:],nan_policy='omit')
    ax.text(0.5, 0.9, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', transform=ax.transAxes,fontsize=12)
    ax.set_xticks([0,1],arealabels[:2])
    ax.axhline(ychance,color='k',linestyle='--',linewidth=1)
    ax.text(0.5, ychance-0.05, 'Chance', color='gray', ha='center', fontsize=8)
    ax.set_xlim([-0.3,1.3])

    ax = axes[2]
    handles = []
    for ial, arealabel in enumerate(arealabels[2:]):
        idx = ial + 2
        handles.append(shaded_error(sbins,dec_perf[:,idx,:].T,color=clrs_arealabels[idx],alpha=0.5,linewidth=1.5,error='sem',ax=ax))
        for ises in range(nSessions):
            ax.plot(sbins,dec_perf[:,idx,ises],color=clrs_arealabels[idx],linewidth=0.2)

    for ibin, bincenter in enumerate(sbins):
        t,pval = ttest_rel(dec_perf[ibin,2,:],dec_perf[ibin,3,:],nan_policy='omit')
        ax.text(bincenter, ymax-0.1, '%s' % get_sig_asterisks(pval,return_ns=False), color='k', ha='center', fontsize=12)
    ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles,labels=arealabels[2:])
    ax.set_xlabel('Position relative to stim (cm)')
    add_stim_resp_win(ax)
    ax.set_xticks([-50,-25,0,25,50,75])
    ax.set_xlim([-50,75])

    ax = axes[3]
    ax.plot([0,1],statdata[2:,:],color='k',linewidth=0.25)
    ax.scatter(np.zeros(nSessions),statdata[2,:],color='k',marker='.',s=markersize)
    ax.scatter(np.ones(nSessions),statdata[3,:],color='k',marker='.',s=markersize)
    ax.scatter([0,1],np.nanmean(statdata[2:,:],axis=1),color=clrs_arealabels[2:],marker='o',s=markersize*2,zorder=10)
    ax.errorbar([0,1],np.nanmean(statdata[2:,:],axis=1),np.nanstd(statdata[2:,:],axis=1)/np.sqrt(nSessions),
                color='k',capsize=0,elinewidth=2,zorder=0)

    t,pval = ttest_rel(statdata[2,:],statdata[3,:],nan_policy='omit')
    ax.text(0.5, 0.9, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', transform=ax.transAxes,fontsize=12)
    ax.set_xticks([0,1],arealabels[2:])
    ax.axhline(ychance,color='k',linestyle='--',linewidth=1)
    ax.text(0.5, ychance-0.05, 'Chance', color='gray', ha='center', fontsize=8)
    ax.set_xlim([-0.3,1.3])

    sns.despine(offset=3,trim=True)

    plt.tight_layout()
    return fig

from statsmodels.stats.anova import AnovaRM

# Show the decoding performance across space for the different populations:
def plot_dec_perf_area(dec_perf,sbins,arealabels,clrs_arealabels,testwin=[0,25]):

    ylow = 0.45 if np.nanmin(dec_perf)>.25 else -0.05
    ymax = 1.0 if np.nanmax(dec_perf)>.75 else 0.5
    ychance = .5 if np.nanmin(dec_perf)>.25 else 0.0
    narealabels = len(arealabels)

    fig,axes    = plt.subplots(1,2,figsize=(6,3),sharex=False,sharey=True,gridspec_kw={'width_ratios': [3,1]})
    markersize  = 30

    statdata    = np.nanmean(dec_perf[(sbins>=testwin[0]) & (sbins<=testwin[1]),:,:],axis=0)

    nSessions   = dec_perf.shape[2]
    # statdata    = np.nanmean(dec_perf[(sbins>=11) & (sbins<=20),:,:],axis=0)
    ax = axes[0]
    handles = []
    for ial, arealabel in enumerate(arealabels):
        handles.append(shaded_error(sbins,dec_perf[:,ial,:].T,color=clrs_arealabels[ial],alpha=0.5,linewidth=1.5,error='sem',ax=ax))
        for ises in range(nSessions):
            ax.plot(sbins,dec_perf[:,ial,ises],color=clrs_arealabels[ial],linewidth=0.2)

    ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles,labels=arealabels)
    ax.set_ylabel('Decoding performance')
    ax.set_xlabel('Position relative to stim (cm)')
    add_stim_resp_win(ax)
    ax.set_xticks([-50,-25,0,25,50,75])
    ax.set_ylim([ylow,ymax])
    ax.set_xlim([-50,75])

    # for ibin, bincenter in enumerate(sbins):
    #     t,pval = ttest_rel(dec_perf[ibin,0,:],dec_perf[ibin,1,:],nan_policy='omit')
    #     ax.text(bincenter, ymax-0.1, '%s' % get_sig_asterisks(pval,return_ns=False), color='k', ha='center', fontsize=12)

    ax = axes[1]
    ax.plot(np.arange(narealabels),statdata,color='k',linewidth=0.25)
    for ial, arealabel in enumerate(arealabels):
        ax.scatter(np.ones(nSessions)*ial,statdata[ial,:],color='k',marker='.',s=markersize)
        ax.scatter(ial,np.nanmean(statdata[ial,:]),color=clrs_arealabels[ial],marker='o',s=markersize*2,zorder=10)
    # ax.scatter([0,1],np.nanmean(statdata[:2,:],axis=1),color=clrs_arealabels[:2],marker='o',s=markersize*2,zorder=10)
        ax.errorbar(ial,np.nanmean(statdata[ial,:]),np.nanstd(statdata[ial,:])/np.sqrt(nSessions),
                    color='k',capsize=0,elinewidth=2,zorder=0)
    
    # df = pd.DataFrame({'dec_perf' : statdata.flatten(),
    #                    'arealabel': np.tile(arealabels,(nSessions,1)).flatten(),
    #                    'session'  : np.tile(np.arange(nSessions).reshape((-1, 1)),(1,narealabels)).flatten()})
    # df = df.dropna().reset_index(drop=True) #drop occasional missing data
    # aov = AnovaRM(data=df, depvar='dec_perf', subject='session', within=['arealabel']).fit()
    # Fval = aov.FValues['arealabel'][0]
   
    # ax.text(0.5, 0.9, f'F({narealabels-1},{narealabels-1}) = {aov.anova_table['F Value'][0]:.2f}\np = {pval:.4f}', color='k', ha='center', transform=ax.transAxes,fontsize=10)
    
    # t,pval = ttest_rel(statdata[0,:],statdata[1,:],nan_policy='omit')
    # ax.text(0.5, 0.9, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', transform=ax.transAxes,fontsize=12)
    ax.set_xticks(np.arange(narealabels),arealabels)
    ax.axhline(ychance,color='k',linestyle='--',linewidth=1)
    ax.text(0.5, ychance-0.05, 'Chance', color='gray', ha='center', fontsize=8)
    ax.set_xlim([-0.3,narealabels - 1 + 0.3])

    sns.despine(offset=3,trim=True)

    plt.tight_layout()
    return fig

# Show the decoding performance across space for the different populations:
def plot_dec_perf_hitmiss_arealabel(dec_perf,sbins,arealabels,clrs_arealabels,testwin=[0,25]):

    ylow = 0.45 if np.nanmin(dec_perf)>.25 else -0.05
    ymax = 1.0 if np.nanmax(dec_perf)>.75 else 0.5
    ychance = .5 if np.nanmin(dec_perf)>.25 else 0.0
    
    fig,axes    = plt.subplots(1,4,figsize=(10,3),sharex=False,sharey=True,gridspec_kw={'width_ratios': [3,1,3,1]})
    markersize  = 30

    statdata    = np.nanmean(dec_perf[(sbins>=testwin[0]) & (sbins<=testwin[1]),:,:,:],axis=0)

    nSessions   = dec_perf.shape[-1]
    # statdata    = np.nanmean(dec_perf[(sbins>=11) & (sbins<=20),:,:],axis=0)
    ax = axes[0]
    handles = []
    for ial, arealabel in enumerate(arealabels[:2]):
        for ilr in range(2):
            handles.append(shaded_error(sbins,dec_perf[:,ial,ilr,:].T,color=clrs_arealabels[ial],
                                    linestyle=['--','-'][ilr],alpha=0.25,linewidth=1.5,error='sem',ax=ax))
    firstlegend = ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles[1::2],labels=arealabels[:2])
    leg_lines = [Line2D([0], [0], color='black', linewidth=2, linestyle='-'),
                Line2D([0], [0], color='black', linewidth=2, linestyle='--')]
    leg_labels = ['Hits','Misses']
    ax.legend(leg_lines, leg_labels, loc='center left', frameon=False, fontsize=9)
    ax.add_artist(firstlegend)

    ax.set_ylabel('Decoding performance')
    ax.set_xlabel('Position relative to stim (cm)')
    add_stim_resp_win(ax)
    ax.set_xticks([-50,-25,0,25,50,75])
    ax.set_ylim([ylow,ymax])
    ax.set_xlim([-50,75])

    # for ibin, bincenter in enumerate(sbins):
    #     t,pval = ttest_rel(dec_perf[ibin,0,:],dec_perf[ibin,1,:],nan_policy='omit')
    #     ax.text(bincenter, 0.9, '%s' % get_sig_asterisks(pval,return_ns=False), color='k', ha='center', fontsize=12)
    
    ax = axes[1]
    ax.plot([0,1],statdata[:2,0,:],color='k',linewidth=0.25)
    ax.scatter(np.zeros(nSessions),statdata[0,0,:],color='k',marker='.',s=markersize)
    ax.scatter(np.ones(nSessions),statdata[1,0,:],color='k',marker='.',s=markersize)
    ax.scatter([0,1],np.nanmean(statdata[:2,0,:],axis=1),color=clrs_arealabels[:2],marker='o',s=markersize*2,zorder=10)
    ax.errorbar([0,1],np.nanmean(statdata[:2,0,:],axis=1),np.nanstd(statdata[:2,0,:],axis=1)/np.sqrt(nSessions),
                color='k',capsize=0,elinewidth=2,zorder=0)
    
    ax.plot([2,3],statdata[:2,1,:],color='k',linewidth=0.25)
    ax.scatter(np.zeros(nSessions)+2,statdata[0,1,:],color='k',marker='.',s=markersize)
    ax.scatter(np.ones(nSessions)+2,statdata[1,1,:],color='k',marker='.',s=markersize)
    ax.scatter([2,3],np.nanmean(statdata[:2,1,:],axis=1),color=clrs_arealabels[:2],marker='o',s=markersize*2,zorder=10)
    ax.errorbar([2,3],np.nanmean(statdata[:2,1,:],axis=1),np.nanstd(statdata[:2,1,:],axis=1)/np.sqrt(nSessions),
                color='k',capsize=0,elinewidth=2,zorder=0)

    t,pval = ttest_rel(statdata[0,0,:],statdata[1,0,:],nan_policy='omit')
    ax.text(0.5, ymax-0.1, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', fontsize=12)

    t,pval = ttest_rel(statdata[0,1,:],statdata[1,1,:],nan_policy='omit')
    ax.text(2.5, ymax-0.1, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', fontsize=12)

    # ax.set_xticks([0,1,2,3],np.tile(arealabels[:2],2),fontsize=7)
    ax.set_xticks([0.5,2.5],['Misses','Hits'],fontsize=7)

    ax.axhline(ychance,color='k',linestyle='--',linewidth=1)
    ax.text(1.5, ychance-0.05, 'Chance', color='gray', ha='center', fontsize=8)
    ax.set_xlim([-0.3,3.3])

    ax = axes[2]
    handles = []
    for ial, arealabel in enumerate(arealabels[2:]):
        idx = ial + 2
        for ilr in range(2):
            handles.append(shaded_error(sbins,dec_perf[:,idx,ilr,:].T,color=clrs_arealabels[idx],
                                    linestyle=['--','-'][ilr],alpha=0.25,linewidth=1.5,error='sem',ax=ax))
        #   for ises in range(nSessions):
            # ax.plot(sbins,dec_perf[:,idx,ises],color=clrs_arealabels[idx],linewidth=0.2)

    # for ibin, bincenter in enumerate(sbins):
    #     t,pval = ttest_rel(dec_perf[ibin,2,:],dec_perf[ibin,3,:],nan_policy='omit')
    #     ax.text(bincenter, 0.9, '%s' % get_sig_asterisks(pval,return_ns=False), color='k', ha='center', fontsize=12)
    firstlegend = ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles[1::2],labels=arealabels[2:])
    leg_lines = [Line2D([0], [0], color='black', linewidth=2, linestyle='-'),
                Line2D([0], [0], color='black', linewidth=2, linestyle='--')]
    leg_labels = ['Hits','Misses']
    ax.legend(leg_lines, leg_labels, loc='center left', frameon=False, fontsize=9)
    ax.add_artist(firstlegend)
    # ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles,labels=arealabels[2:])
    ax.set_xlabel('Position relative to stim (cm)')
    add_stim_resp_win(ax)
    ax.set_xticks([-50,-25,0,25,50,75])
    ax.set_xlim([-50,75])

    ax = axes[3]
    ax.plot([0,1],statdata[2:,0,:],color='k',linewidth=0.25)
    ax.scatter(np.zeros(nSessions),statdata[2,0,:],color='k',marker='.',s=markersize)
    ax.scatter(np.ones(nSessions),statdata[3,0,:],color='k',marker='.',s=markersize)
    ax.scatter([0,1],np.nanmean(statdata[2:,0,:],axis=1),color=clrs_arealabels[2:],marker='o',s=markersize*2,zorder=10)
    ax.errorbar([0,1],np.nanmean(statdata[2:,0,:],axis=1),np.nanstd(statdata[2:,0,:],axis=1)/np.sqrt(nSessions),
                color='k',capsize=0,elinewidth=2,zorder=0)
    
    ax.plot([2,3],statdata[2:,1,:],color='k',linewidth=0.25)
    ax.scatter(np.zeros(nSessions)+2,statdata[2,1,:],color='k',marker='.',s=markersize)
    ax.scatter(np.ones(nSessions)+2,statdata[3,1,:],color='k',marker='.',s=markersize)
    ax.scatter([2,3],np.nanmean(statdata[2:,1,:],axis=1),color=clrs_arealabels[2:],marker='o',s=markersize*2,zorder=10)
    ax.errorbar([2,3],np.nanmean(statdata[2:,1,:],axis=1),np.nanstd(statdata[2:,1,:],axis=1)/np.sqrt(nSessions),
                color='k',capsize=0,elinewidth=2,zorder=0)

    t,pval = ttest_rel(statdata[2,0,:],statdata[3,0,:],nan_policy='omit')
    ax.text(0.5, ymax-0.1, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', fontsize=12)

    t,pval = ttest_rel(statdata[2,1,:],statdata[3,1,:],nan_policy='omit')
    ax.text(2.5, ymax-0.1, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', fontsize=12)

    # ax.set_xticks([0,1,2,3],np.tile(arealabels[2:],2),fontsize=7)
    ax.set_xticks([0.5,2.5],['Misses','Hits'],fontsize=7)

    ax.axhline(ychance,color='k',linestyle='--',linewidth=1)
    ax.text(0.5, ychance-0.05, 'Chance', color='gray', ha='center', fontsize=8)
    ax.set_xlim([-0.3,3.3])

    sns.despine(offset=3,trim=True)

    plt.tight_layout()
    return fig


# Show the decoding performance across space for the different populations:
def plot_dec_perf_hitmiss_arealabel2(dec_perf,sbins,arealabels,clrs_arealabels,testwin=[0,25]):

    ylow = 0.45 if np.nanmin(dec_perf)>.25 else -0.05
    ymax = 1.0 if np.nanmax(dec_perf)>.75 else 0.5
    ychance = .5 if np.nanmin(dec_perf)>.25 else 0.0
    
    fig,axes    = plt.subplots(1,len(arealabels)*2,figsize=(2.5*len(arealabels)*2,3),sharex=False,sharey=True,gridspec_kw={'width_ratios': np.tile([3,1],len(arealabels))})
    markersize  = 30

    statdata    = np.nanmean(dec_perf[(sbins>=testwin[0]) & (sbins<=testwin[1]),:,:,:],axis=0)

    nSessions   = dec_perf.shape[-1]
    # statdata    = np.nanmean(dec_perf[(sbins>=11) & (sbins<=20),:,:],axis=0)
    for ial,arealabel in enumerate(arealabels):
        ax = axes[ial*2]
        handles = []
        for ilr in range(2):
            handles.append(shaded_error(sbins,dec_perf[:,ial,ilr,:].T,color=clrs_arealabels[ial],
                                linestyle=['--','-'][ilr],alpha=0.25,linewidth=1.5,error='sem',ax=ax))
        ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles,labels=['Hits','Misses'])
        # ax.legend(leg_lines, leg_labels, loc='center left', frameon=False, fontsize=9)
        # ax.add_artist(firstlegend)
        ax.set_title(arealabel)
        ax.set_ylabel('Decoding performance')
        ax.set_xlabel('Position relative to stim (cm)')
        add_stim_resp_win(ax)
        ax.set_xticks([-50,-25,0,25,50,75])
        ax.set_ylim([ylow,ymax])
        ax.set_xlim([-50,75])

        ax = axes[ial*2+1]
        ax.plot([0,1],statdata[ial,:,:],color='k',linewidth=0.25)
        ax.scatter(np.zeros(nSessions),statdata[ial,0,:],color='k',marker='.',s=markersize)
        ax.scatter(np.ones(nSessions),statdata[ial,1,:],color='k',marker='.',s=markersize)
        ax.scatter([0,1],np.nanmean(statdata[ial,:,:],axis=1),color=clrs_arealabels[ial],marker='o',s=markersize*2,zorder=10)
        ax.errorbar([0,1],np.nanmean(statdata[ial,:,:],axis=1),np.nanstd(statdata[ial,:,:],axis=1)/np.sqrt(nSessions),
                    color='k',capsize=0,elinewidth=2,zorder=0)

        t,pval = ttest_rel(statdata[ial,0,:],statdata[ial,1,:],nan_policy='omit')
        ax.text(0.5, ymax-0.1, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', fontsize=12)

        # ax.set_xticks([0,1,2,3],np.tile(arealabels[:2],2),fontsize=7)
        ax.set_xticks([0,1],['Misses','Hits'],fontsize=7)

        ax.axhline(ychance,color='k',linestyle='--',linewidth=1)
        ax.text(0.5, ychance-0.05, 'Chance', color='gray', ha='center', fontsize=8)
        ax.set_xlim([-0.3,1.3])

    sns.despine(offset=3,trim=True)

    plt.tight_layout()
    return fig

####### #     #  #####  ####### ######  ### #     #  #####  
#       ##    # #     # #     # #     #  #  ##    # #     # 
#       # #   # #       #     # #     #  #  # #   # #       
#####   #  #  # #       #     # #     #  #  #  #  # #  #### 
#       #   # # #       #     # #     #  #  #   # # #     # 
#       #    ## #     # #     # #     #  #  #    ## #     # 
####### #     #  #####  ####### ######  ### #     #  #####  

def get_enc_predictors(ses,ibin=0):
    X           = np.empty([len(ses.trialdata), 0])
    varnames    = np.array([], dtype=object)
    X           = np.c_[X, np.atleast_2d(ses.trialdata['trialNumber'].to_numpy()).T]
    varnames    = np.append(varnames, ['trialnumber'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signal_raw'])
    temp        = copy.deepcopy(ses.trialdata['signal_psy'].to_numpy())
    temp[ses.trialdata['signal'] == 0] = -3
    temp[ses.trialdata['signal'] == 100] = 10
    X           = np.c_[X, np.atleast_2d(temp).T]
    varnames    = np.append(varnames, ['signal_psy'])
    temp        = copy.deepcopy(ses.trialdata['signal_psy'].to_numpy())
    temp[ses.trialdata['signal'] == 0] = -3
    temp[ses.trialdata['signal'] == 100] = -3
    X           = np.c_[X, np.atleast_2d(temp).T]
    varnames    = np.append(varnames, ['signal_psy_noise'])
    X           = np.c_[X, np.atleast_2d((ses.trialdata['stimcat'] == 'M').to_numpy()).T]
    varnames    = np.append(varnames, ['stimcat_M'])
    X           = np.c_[X, np.atleast_2d((ses.trialdata['stimcat'] == 'N').to_numpy()).T]
    varnames    = np.append(varnames, ['stimcat_N'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['lickResponse'].to_numpy()).T]
    varnames    = np.append(varnames, ['lickresponse'])
    temp        = ses.trialdata['lickResponse'].to_numpy() * 2 - 1
    temp[ses.trialdata['stimcat'] != 'N'] = 0
    X           = np.c_[X, np.atleast_2d(temp).T]
    varnames    = np.append(varnames, ['lickresponse_noise'])
    temp        = ses.trialdata['lickResponse'].to_numpy() * 2 - 1
    temp[ses.trialdata['stimcat'] != 'N'] = 0
    temp[ses.trialdata['engaged'] != 1] = 0
    X           = np.c_[X, np.atleast_2d(temp).T]
    varnames    = np.append(varnames, ['lickresponse_noise_eng'])
    temp        = ses.trialdata['lickResponse'].to_numpy() * 2 - 1
    temp[ses.trialdata['engaged'] != 1] = 0
    X           = np.c_[X, np.atleast_2d(temp).T]
    varnames    = np.append(varnames, ['lickresponse_eng'])
    X           = np.c_[X, np.atleast_2d(ses.respmat_runspeed).T]
    varnames    = np.append(varnames, ['runspeed'])
    X           = np.c_[X, np.atleast_2d(ses.runPSTH[:,ibin]).T]
    varnames    = np.append(varnames, ['runbin'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['engaged'].to_numpy()).T]
    varnames    = np.append(varnames, ['engaged'])
    X           = np.c_[X, np.atleast_2d(np.random.normal(0,1,len(ses.trialdata))).T]
    varnames    = np.append(varnames, ['random'])
    X           = np.c_[X, np.atleast_2d(ses.respmat_runspeed.flatten() * ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signalxrun'])
    X           = np.c_[X, np.atleast_2d(-ses.respmat_runspeed.flatten() * ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signalxruninv'])
    X           = np.c_[X, np.atleast_2d(ses.runPSTH[:,ibin].flatten() * ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signalxrunbin'])
    X           = np.c_[X, np.atleast_2d(-ses.runPSTH[:,ibin].flatten() * ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signalxrunbininv'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['lickResponse'].to_numpy() * ses.trialdata['signal_psy'].to_numpy()).T]
    varnames    = np.append(varnames, ['signal_psyxhit'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['lickResponse'].to_numpy() * ses.trialdata['stimcat'] == 'M').T]
    varnames    = np.append(varnames, ['signal_maxxhit'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['rewardGiven'].to_numpy()).T]
    varnames    = np.append(varnames, ['reward'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['nLicks'].to_numpy()).T]
    varnames    = np.append(varnames, ['nlicks'])
    
    return X,varnames

def get_enc_predictors_from_modelversion(version='v1'):
    modelvars_dict = {
            'v1': ['trialnumber'],
            'v2': ['trialnumber','signal_raw'],
            'v3': ['trialnumber','signal_psy'],
            'v4': ['trialnumber','stimcat_M'],
            'v5': ['trialnumber','stimcat_N'],
            'v6': ['trialnumber','signal_psy_noise','stimcat_M'],
            'v7': ['trialnumber','signal_psy_noise','stimcat_M','runspeed'],
            'v8': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxrun'],
            'v9': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxruninv'],
            'v10': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxrun','lickresponse'],
            'v11': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxrun','lickresponse_noise'],
            'v12': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxrun','signal_psyxhit'],
            'v13': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxrun','signal_maxxhit'],
            'v14': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxrun','reward'],
            'v15': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxrun','nlicks'],
            'v20': ['trialnumber','signal_psy_noise','stimcat_M','runbin','signalxrunbin','reward']
            }


    return modelvars_dict[version]

def enc_model_stimwin_wrapper(ses,idx_N,idx_T,version='v1',modelname='Lasso',optimal_lambda=None,kfold=5,scoring_type = 'r2_score',
                              crossval=True,subtr_shuffle=False):
    """
    Wrapper function to calculate encoding model performance for all neurons in a session.
    
    Parameters
    ----------
    ses : Session object
        Session object containing data for one session.
    idx_N : array
        Array of neuron indices to use for encoding model.
    idx_T : array
        Array of trial indices to use for encoding model.
    modelname : string
        Name of the model to use. Options are 'LinearRegression', 'Lasso', 'Ridge', 'ElasticNet'.
    optimal_lambda : float, optional
        Optimal regularization strength for the model. If None, the function will find the optimal lambda using cross-validation.
    kfold : integer, optional
        Number of folds to use for cross-validation. Default is 5.
    scoring_type : string, optional
        Scoring type to use for cross-validation. Options are 'r2', 'mean_squared_error'. Default is 'r2'.
    crossval : boolean, optional
        Whether or not to use cross-validation. Default is True.
    subtr_shuffle : boolean, optional
        Whether or not to subtract the shuffling performance from the average performance. Default is False.
    
    Returns
    -------
    error : array
        Average encoding error across folds. Size is N x 1 where N is the number of neurons
    weights : array
        Weights for the encoding model. Size is N x 1 where N is the number of neurons
    y_hat : array
        Predicted values for the encoding model. Size is N x T where N is the number of neurons and 
        T is the number of trials
    """

    assert modelname in ['LinearRegression','Lasso','Ridge','ElasticNet'],'Unknown modelname %s' % modelname
    assert np.sum(idx_T) > 50, 'Not enough trials in session %d' % ses.sessiondata['session_id'][0]
    assert np.sum(idx_N) > 1, 'Not enough neurons in session %d' % ses.sessiondata['session_id'][0]

    modelvars   = get_enc_predictors_from_modelversion(version)

    V           = len(modelvars)
    K           = len(ses.trialdata)
    N           = len(ses.celldata)
    N_idx       = np.sum(idx_N)
    weights     = np.full((N,V),np.nan)
    error       = np.full((N),np.nan)
    error_var   = np.full((N,V),np.nan)
    y_hat       = np.full((N,K),np.nan)

    y           = ses.respmat[np.ix_(idx_N,idx_T)].T

    X,allvars   = get_enc_predictors(ses)               # get all predictors
    X           = X[:,np.isin(allvars,modelvars)] #get only predictors of interest
    X           = X[idx_T,:]                     #get only trials of interest
   
    X,y         = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

    if optimal_lambda is None:
        # Find the optimal regularization strength (lambda)
        lambdas = np.logspace(-4, 4, 20)
        cv_scores = np.zeros((len(lambdas),))
        for ilambda, lambda_ in enumerate(lambdas):
            model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)
            # model = ElasticNet(alpha=lambda_,l1_ratio=0.9)
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring_type.replace('_score',''))
            cv_scores[ilambda] = np.mean(scores)
        optimal_lambda = lambdas[np.argmax(cv_scores)]

    # Train a regression model on the training data with regularization
    # model = ElasticNet(alpha=optimal_lambda,l1_ratio=0.9)
    model = getattr(sklearn.linear_model,modelname)(alpha=optimal_lambda)
    score_fun = getattr(sklearn.metrics,scoring_type)

    if crossval:
        # Define the k-fold cross-validation object
        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        
        # Initialize an array to store the decoding performance for each fold
        fold_error             = np.zeros((kfold,N_idx))
        fold_error_var         = np.zeros((kfold,N_idx,V))
        # fold_r2_shuffle     = np.zeros((kfold,N))
        fold_weights        = np.zeros((kfold,N_idx,V))

        # Loop through each fold
        for ifold, (train_index, test_index) in enumerate(kf.split(X)):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)

            # Compute R2 on the test set
            y_pred                  = model.predict(X_test)
            fold_error[ifold,:]     = score_fun(y_test, y_pred, multioutput='raw_values')
            
            fold_weights[ifold,:,:]     = model.coef_
            y_hat[np.ix_(idx_N,test_index)] = y_pred.T
            
            for ivar in range(V):
                X_test_var              = copy.deepcopy(X_test)
                X_test_var[:,ivar] = 0
                y_pred                  = model.predict(X_test_var)
                fold_error_var[ifold,:,ivar] = fold_error[ifold,:] - score_fun(y_test, y_pred, multioutput='raw_values')

            if subtr_shuffle:
                print('Shuffling labels not yet implemented')
                # # Shuffle the labels and calculate the decoding performance for this fold
                # np.random.shuffle(y_train)
                # model.fit(X_train, y_train)
                # y_pred = model.predict(X_test)
                # fold_r2_shuffle[ifold] = accuracy_score(y_test, y_pred)
    
        # Calculate the average decoding performance across folds
        error[idx_N] = np.nanmean(fold_error, axis=0)
        error_var[idx_N,:] = np.nanmean(fold_error_var, axis=0)
        weights[idx_N,:] = np.nanmean(fold_weights, axis=0)

    else:   
        # Without cross-validation
        model.fit(X, y)
        y_pred = model.predict(X)
        error[idx_N] = r2_score(y, y_pred, multioutput='raw_values')
        y_hat[np.ix_(idx_N,idx_T)] = y_pred.T
        weights[idx_N,:] = model.coef_
    
    return error, weights, y_hat, error_var



def enc_model_spatial_wrapper(ses,sbins,idx_N,idx_T,version='v20',modelname='Lasso',optimal_lambda=None,kfold=5,scoring_type = 'r2',
                              crossval=True,subtr_shuffle=False):
    """
    Wrapper function to calculate encoding model performance for all neurons in a session.
    
    Parameters
    ----------
    ses : Session object
        Session object containing data for one session.
    sbins : array
        Array of spatial bin centers.
    idx_N : array
        Array of neuron indices to use for encoding model.
    idx_T : array
        Array of trial indices to use for encoding model.
    modelname : string
        Name of the model to use. Options are 'LinearRegression', 'Lasso', 'Ridge', 'ElasticNet'.
    optimal_lambda : float, optional
        Optimal regularization strength for the model. If None, the function will find the optimal lambda using cross-validation.
    kfold : integer, optional
        Number of folds to use for cross-validation. Default is 5.
    scoring_type : string, optional
        Scoring type to use for cross-validation. Options are 'r2', 'neg_mean_squared_error'. Default is 'r2'.
    crossval : boolean, optional
        Whether or not to use cross-validation. Default is True.
    subtr_shuffle : boolean, optional
        Whether or not to subtract the shuffling performance from the average performance. Default is False.
    
    Returns
    -------
    error : array
        Average encoding error across folds. Size is N x S where N is the number of neurons and S is the number of spatial bins
    weights : array
        Weights for the encoding model. Size is N x S where N is the number of neurons and S is the number of spatial bins
    y_hat : array
        Predicted values for the encoding model. Size is N x S x T where N is the number of neurons and S is the number of spatial bins and 
        T is the number of trials
    """

    assert modelname in ['LinearRegression','Lasso','Ridge','ElasticNet'],'Unknown modelname %s' % modelname
    assert np.sum(idx_T) > 50, 'Not enough trials in session %d' % ses.sessiondata['session_id'][0]
    assert np.sum(idx_N) > 1, 'Not enough neurons in session %d' % ses.sessiondata['session_id'][0]

    modelvars   = get_enc_predictors_from_modelversion(version)
    
    if optimal_lambda is None:
        y           = ses.respmat[np.ix_(idx_N,idx_T)].T

        X,allvars   = get_enc_predictors(ses)               # get all predictors
        X           = X[:,np.isin(allvars,modelvars)] #get only predictors of interest
        X           = X[idx_T,:]                     #get only trials of interest

        X,y         = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        # Find the optimal regularization strength (lambda)
        lambdas = np.logspace(-4, 4, 20)
        cv_scores = np.zeros((len(lambdas),))
        for ilambda, lambda_ in enumerate(lambdas):
            model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)
            # model = ElasticNet(alpha=lambda_,l1_ratio=0.9)
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring_type.replace('_score',''))
            cv_scores[ilambda] = np.median(scores)
        optimal_lambda = lambdas[np.argmax(cv_scores)]

    # Train a regression model on the training data with regularization
    # model = ElasticNet(alpha=optimal_lambda,l1_ratio=0.9)
    model       = getattr(sklearn.linear_model,modelname)(alpha=optimal_lambda)
    score_fun   = getattr(sklearn.metrics,scoring_type)

    S           = len(sbins)
    V           = len(modelvars)
    K           = len(ses.trialdata)
    N           = len(ses.celldata)
    N_idx       = np.sum(idx_N)
    weights     = np.full((N,S,V),np.nan)
    error       = np.full((N,S),np.nan)
    error_var   = np.full((N,S,V),np.nan)
    y_hat       = np.full((N,S,K),np.nan)

    for ibin, bincenter in enumerate(sbins):    # Loop over each spatial bin
        y = ses.stensor[np.ix_(idx_N,idx_T,[ibin])].squeeze().T # Get the neural response data for this bin

        X,allvars   = get_enc_predictors(ses,ibin)               # get all predictors
        X           = X[:,np.isin(allvars,modelvars)] #get only predictors of interest
        X           = X[idx_T,:]                     #get only trials of interest
    
        X,y         = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        if crossval:
            # Define the k-fold cross-validation object
            kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
            
            # Initialize an array to store the decoding performance for each fold
            fold_error             = np.zeros((kfold,N_idx))
            # fold_r2_shuffle     = np.zeros((kfold,N))
            fold_weights        = np.zeros((kfold,N_idx,V))
            fold_error_var      = np.zeros((kfold,N_idx,V))

            # Loop through each fold
            for ifold, (train_index, test_index) in enumerate(kf.split(X)):
                # Split the data into training and testing sets
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                model.fit(X_train, y_train)

                # Compute R2 on the test set
                y_pred                  = model.predict(X_test)
                # fold_error[ifold,:]     = r2_score(y_test, y_pred, multioutput='raw_values')
                fold_error[ifold,:]     = score_fun(y_test, y_pred, multioutput='raw_values')
                
                fold_weights[ifold,:,:]     = model.coef_
                y_hat[np.ix_(idx_N,[ibin],test_index)] = y_pred.T[:,np.newaxis,:]
                
                for ivar in range(V):
                    X_test_var              = copy.deepcopy(X_test)
                    # X_test_var[:,np.arange(V) != ivar] = 0
                    X_test_var[:,ivar] = 0
                    y_pred                  = model.predict(X_test_var)
                    # fold_error_var[ifold,:,ivar] = score_fun(y_test, y_pred, multioutput='raw_values')
                    fold_error_var[ifold,:,ivar] = fold_error[ifold,:] - score_fun(y_test, y_pred, multioutput='raw_values')

                if subtr_shuffle:
                    print('Shuffling labels not yet implemented')
                    # # Shuffle the labels and calculate the decoding performance for this fold
                    # np.random.shuffle(y_train)
                    # model.fit(X_train, y_train)
                    # y_pred = model.predict(X_test)
                    # fold_r2_shuffle[ifold] = accuracy_score(y_test, y_pred)
        
            # Calculate the average decoding performance across folds
            error[idx_N,ibin]       = np.nanmean(fold_error, axis=0)
            error_var[idx_N,ibin,:] = np.nanmean(fold_error_var, axis=0)
            weights[idx_N,ibin,:]   = np.nanmean(fold_weights, axis=0)

        else:   
            # Without cross-validation
            model.fit(X, y)
            y_pred = model.predict(X)
            error[idx_N,ibin] = r2_score(y, y_pred, multioutput='raw_values')
            y_hat[np.ix_(idx_N,[ibin],idx_T)] = y_pred.T[:,np.newaxis,:]
            weights[idx_N,ibin,:] = model.coef_
    
    return error, weights, y_hat, error_var

######  #######  #####  ####### ######  ### #     #  #####  
#     # #       #     # #     # #     #  #  ##    # #     # 
#     # #       #       #     # #     #  #  # #   # #       
#     # #####   #       #     # #     #  #  #  #  # #  #### 
#     # #       #       #     # #     #  #  #   # # #     # 
#     # #       #     # #     # #     #  #  #    ## #     # 
######  #######  #####  ####### ######  ### #     #  #####  


def get_dec_predictors(ses,ibin=0,nneuraldims=10):
    X           = np.empty([len(ses.trialdata), 0])
    varnames    = np.array([], dtype=object)
    X           = np.c_[X, np.atleast_2d(ses.trialdata['trialNumber'].to_numpy()).T]
    varnames    = np.append(varnames, ['trialnumber'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signal_raw'])
    temp        = copy.deepcopy(ses.trialdata['signal_psy'].to_numpy())
    temp[ses.trialdata['signal'] == 0] = -3
    temp[ses.trialdata['signal'] == 100] = 10
    X           = np.c_[X, np.atleast_2d(temp).T]
    varnames    = np.append(varnames, ['signal_psy'])
    temp        = copy.deepcopy(ses.trialdata['signal_psy'].to_numpy())
    temp[ses.trialdata['signal'] == 0] = -3
    temp[ses.trialdata['signal'] == 100] = -3
    X           = np.c_[X, np.atleast_2d(temp).T]
    varnames    = np.append(varnames, ['signal_psy_noise'])
    X           = np.c_[X, np.atleast_2d((ses.trialdata['stimcat'] == 'M').to_numpy()).T]
    varnames    = np.append(varnames, ['stimcat_M'])
    X           = np.c_[X, np.atleast_2d((ses.trialdata['stimcat'] == 'N').to_numpy()).T]
    varnames    = np.append(varnames, ['stimcat_N'])
    # X           = np.c_[X, np.atleast_2d(ses.trialdata['lickResponse'].to_numpy()).T]
    # varnames    = np.append(varnames, ['lickresponse'])
    # temp        = ses.trialdata['lickResponse'].to_numpy() * 2 - 1
    # temp[ses.trialdata['stimcat'] != 'N'] = 0
    # X           = np.c_[X, np.atleast_2d(temp).T]
    # varnames    = np.append(varnames, ['lickresponse_noise'])
    # temp        = ses.trialdata['lickResponse'].to_numpy() * 2 - 1
    # temp[ses.trialdata['stimcat'] != 'N'] = 0
    # temp[ses.trialdata['engaged'] != 1] = 0
    # X           = np.c_[X, np.atleast_2d(temp).T]
    # varnames    = np.append(varnames, ['lickresponse_noise_eng'])
    # temp        = ses.trialdata['lickResponse'].to_numpy() * 2 - 1
    # temp[ses.trialdata['engaged'] != 1] = 0
    # X           = np.c_[X, np.atleast_2d(temp).T]
    # varnames    = np.append(varnames, ['lickresponse_eng'])
    X           = np.c_[X, np.atleast_2d(ses.respmat_runspeed).T]
    varnames    = np.append(varnames, ['runspeed'])
    X           = np.c_[X, np.atleast_2d(ses.runPSTH[:,ibin]).T]
    varnames    = np.append(varnames, ['runbin'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['engaged'].to_numpy()).T]
    varnames    = np.append(varnames, ['engaged'])
    X           = np.c_[X, np.atleast_2d(np.random.normal(0,1,len(ses.trialdata))).T]
    varnames    = np.append(varnames, ['random'])
    X           = np.c_[X, np.atleast_2d(ses.respmat_runspeed.flatten() * ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signalxrun'])
    X           = np.c_[X, np.atleast_2d(-ses.respmat_runspeed.flatten() * ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signalxruninv'])
    X           = np.c_[X, np.atleast_2d(ses.runPSTH[:,ibin].flatten() * ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signalxrunbin'])
    X           = np.c_[X, np.atleast_2d(-ses.runPSTH[:,ibin].flatten() * ses.trialdata['signal'].to_numpy()).T]
    varnames    = np.append(varnames, ['signalxrunbininv'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['lickResponse'].to_numpy() * ses.trialdata['signal_psy'].to_numpy()).T]
    varnames    = np.append(varnames, ['signal_psyxhit'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['lickResponse'].to_numpy() * ses.trialdata['stimcat'] == 'M').T]
    varnames    = np.append(varnames, ['signal_maxxhit'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['rewardGiven'].to_numpy()).T]
    varnames    = np.append(varnames, ['reward'])
    X           = np.c_[X, np.atleast_2d(ses.trialdata['nLicks'].to_numpy()).T]
    varnames    = np.append(varnames, ['nlicks'])

    #Add individual cells neural data:
    idx_N       = np.ones(len(ses.celldata), dtype=bool)
    X           = np.c_[X, ses.respmat[idx_N,:].T]
    # varnames    = np.append(varnames, ses.celldata['cell_id'][idx_N])
    varnames    = np.append(varnames, np.repeat('allcells',len(ses.celldata['cell_id'][idx_N])))
    
    areas = ['V1', 'PM', 'AL', 'RSP']
    for iarea,area in enumerate(areas):
        idx_N       = ses.celldata['roi_name']==area
        X           = np.c_[X, ses.respmat[idx_N,:].T]
        varnames    = np.append(varnames, np.repeat(area,len(ses.celldata['cell_id'][idx_N])))

    # Add first nPCs from all neurons, or individual areas:
    idx_N       = np.ones(len(ses.celldata), dtype=bool)
    pcadata     = ses.respmat[idx_N,:].T
    pcadata[np.isnan(pcadata)] = 0
    X           = np.c_[X, PCA(n_components=nneuraldims).fit_transform(pcadata)]
    varnames    = np.append(varnames, ['PC{}_all'.format(i) for i in np.arange(nneuraldims)])
    
    areas = ['V1', 'PM', 'AL', 'RSP']
    for iarea,area in enumerate(areas):
        idx_N       = ses.celldata['roi_name']==area
        if np.sum(idx_N) > nneuraldims:
            pcadata     = ses.respmat[idx_N,:].T
            pcadata[np.isnan(pcadata)] = 0
            X           = np.c_[X, PCA(n_components=nneuraldims).fit_transform(pcadata)]
        else: 
            X           = np.c_[X, np.zeros((np.shape(X)[0],nneuraldims))]
        varnames    = np.append(varnames, ['PC{}_{}'.format(i,area) for i in np.arange(nneuraldims)])

    for iarea,area in enumerate(areas):
        idx_T           = np.ones(len(ses.trialdata), dtype=bool)
        idx_N           = ses.celldata['roi_name']==area
        proj            = get_signal_dim(ses,idx_T,idx_N)[1] if np.sum(idx_N) > 10 else np.zeros((np.shape(X)[0],1))
        X               = np.c_[X, proj]
        varnames        = np.append(varnames, 'SS_signal_{}'.format(area))

    for iarea,area in enumerate(areas):
        idx_N           = ses.celldata['roi_name']==area
        idx_T           = np.isin(ses.trialdata['stimcat'],['C','M'])
        proj            = get_signal_dim(ses,idx_T,idx_N)[1] if np.sum(idx_N) > 10 else np.zeros((np.shape(X)[0],1))

        X               = np.c_[X, proj]
        varnames        = np.append(varnames, 'SS_signal_max_{}'.format(area))

    for iarea,area in enumerate(areas):
        idx_N           = ses.celldata['roi_name']==area
        idx_T           = ses.trialdata['stimcat'] == 'N'
        proj            = get_signal_dim(ses,idx_T,idx_N)[1] if np.sum(idx_N) > 10 else np.zeros((np.shape(X)[0],1))

        X               = np.c_[X, proj]
        varnames        = np.append(varnames, 'SS_signal_noise_{}'.format(area))

    # Add signal dimension from all neurons:
    idx_T           = np.isin(ses.trialdata['stimcat'],['C','M'])
    idx_N           = np.ones(len(ses.celldata), dtype=bool)
    proj            = get_signal_dim(ses,idx_T,idx_N)[1] if np.sum(idx_N) > 10 else np.zeros((np.shape(X)[0],1))
    X               = np.c_[X, proj]
    varnames        = np.append(varnames, 'SS_signal_max_all')

     # Add signal dimension from all neurons:
    idx_T           = ses.trialdata['stimcat'] == 'N'
    idx_N           = np.ones(len(ses.celldata), dtype=bool)
    proj            = get_signal_dim(ses,idx_T,idx_N)[1] if np.sum(idx_N) > 10 else np.zeros((np.shape(X)[0],1))
    X               = np.c_[X, proj]
    varnames        = np.append(varnames, 'SS_signal_noise_all')

    assert np.shape(X)[1] == np.shape(varnames)[0], 'X and varnames must have the same number of columns'

    return X,varnames

def get_signal_dim(ses,idx_T,idx_N):
    # idx_N is the subset of cells used to estimate the signal dimension
    # idx_T is the subset of trials used to estimate the signal dimension
    # the output is the projection of all trials on the signal dimension

    model_name      = 'Ridge'
    scoring_type    = 'r2_score'
    lam             = None
    kfold           = 5

    X_all  = ses.respmat[idx_N,:].T
    X      = ses.respmat[np.ix_(idx_N,idx_T)].T
    y      = ses.trialdata['signal'][idx_T]

    X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
    
    _,weights,_,_ = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,lam=lam,
                                                scoring_type=scoring_type,norm_out=False,subtract_shuffle=False) 
    X_all = zscore(X_all,axis=0,nan_policy='omit')
    proj = np.dot(X_all,weights)
    
    return weights,proj


def get_dec_predictors_from_modelversion(version='v1',nneuraldims=10):
    modelvars_dict = {
            'v1': ['trialnumber'],
            'v2': ['trialnumber','signal_raw'],
            'v3': ['trialnumber','signal_psy'],
            'v4': ['trialnumber','stimcat_M'],
            'v5': ['trialnumber','stimcat_N'],
            'v6': ['trialnumber','signal_psy_noise','stimcat_M'],
            'v7': ['trialnumber','signal_psy_noise','stimcat_M','runspeed'],
            'v8': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','signalxrun'],
            'v9': ['signal_psy_noise','stimcat_M'],
            'v10': ['trialnumber','signal_psy_noise','stimcat_M','runspeed'] + ['PC{}_{}'.format(i,'all') for i in np.arange(nneuraldims)],
            'v11': ['trialnumber','signal_psy_noise','stimcat_M','runspeed'] + ['PC{}_{}'.format(i,'V1') for i in np.arange(nneuraldims)],
            'v12': ['trialnumber','signal_psy_noise','stimcat_M','runspeed'] + ['PC{}_{}'.format(i,'PM') for i in np.arange(nneuraldims)],
            'v13': ['trialnumber','signal_psy_noise','stimcat_M','runspeed'] + ['PC{}_{}'.format(i,'AL') for i in np.arange(nneuraldims)],
            'v14': ['trialnumber','signal_psy_noise','stimcat_M','runspeed'] + ['PC{}_{}'.format(i,'RSP') for i in np.arange(nneuraldims)],
            'v15': ['trialnumber','signal_psy_noise','stimcat_M','runspeed', 'SS_signal_V1'],
            'v16': ['trialnumber','signal_psy_noise','stimcat_M','runspeed', 'SS_signal_PM'],
            'v17': ['trialnumber','signal_psy_noise','stimcat_M','runspeed', 'SS_signal_AL'],
            'v18': ['trialnumber','signal_psy_noise','stimcat_M','runspeed', 'SS_signal_RSP'],
            'v19': ['trialnumber','signal_psy_noise','stimcat_M','runspeed', 'SS_signal_noise_V1'],
            'v20': ['trialnumber','signal_psy_noise','stimcat_M','runspeed', 'SS_signal_noise_PM'],
            'v21': ['trialnumber','signal_psy_noise','stimcat_M','runspeed', 'SS_signal_noise_AL'],
            'v22': ['trialnumber','signal_psy_noise','stimcat_M','runspeed', 'SS_signal_noise_RSP'],
            'v23': ['PC{}_{}'.format(i,'all') for i in np.arange(nneuraldims)],
            'v24': ['PC{}_{}'.format(i,'V1') for i in np.arange(nneuraldims)],
            'v25': ['PC{}_{}'.format(i,'PM') for i in np.arange(nneuraldims)],
            'v26': ['PC{}_{}'.format(i,'AL') for i in np.arange(nneuraldims)],
            'v27': ['PC{}_{}'.format(i,'RSP') for i in np.arange(nneuraldims)],
            'v28': ['SS_signal_V1','SS_signal_PM','SS_signal_AL','SS_signal_RSP'],
            'v29': ['SS_signal_noise_V1','SS_signal_noise_PM','SS_signal_noise_AL','SS_signal_noise_RSP'],
            'v30': ['SS_signal_max_V1','SS_signal_max_PM','SS_signal_max_AL','SS_signal_max_RSP'],
            'v31': ['SS_signal_noise_V1','SS_signal_noise_PM','SS_signal_noise_AL','SS_signal_noise_RSP','SS_signal_max_V1','SS_signal_max_PM','SS_signal_max_AL','SS_signal_max_RSP'],
            'v32': ['SS_signal_noise_all','SS_signal_max_all'],
            'v33': ['allcells'],
            'v34': ['V1'],
            'v35': ['PM'],
            'v36': ['AL'],
            'v37': ['RSP'],
            'v38': ['trialnumber','signal_psy_noise','stimcat_M','runspeed','allcells'],
            }
    
    return modelvars_dict[version]

def get_dec_modelname(version='v1'):
    abbr_modelnames = {
            'v1': 'Trialnumber',
            'v2': 'Sig',
            'v3': 'Sig_psy',
            'v4': 'Sig_M',
            'v5': 'Sig_N',
            'v6': 'Sig2',
            'v7': 'Sig2_run',
            'v8': 'Sig2_run2',
            'v9': 'Sig2_only',
            'v10': 'Taskvars_PCall',
            'v11': 'Taskvars_PC_V1',
            'v12': 'Taskvars_PC_PM',
            'v13': 'Taskvars_PC_AL',
            'v14': 'Taskvars_PC_RSP',
            'v15': 'Taskvars_Sig_Dim_V1',
            'v16': 'Taskvars_Sig_Dim_PM',
            'v17': 'Taskvars_Sig_Dim_AL',
            'v18': 'Taskvars_Sig_Dim_RSP',
            'v19': 'Taskvars_Noise_Dim_V1',
            'v20': 'Taskvars_Noise_Dim_PM',
            'v21': 'Taskvars_Noise_Dim_AL',
            'v22': 'Taskvars_Noise_Dim_RSP',
            'v23': 'PCall',
            'v24': 'PC_V1',
            'v25': 'PC_PM',
            'v26': 'PC_AL',
            'v27': 'PC_RSP',
            'v28': 'Sig_Dim_Areas',
            'v29': 'Noise_Dim_Areas',
            'v30': 'Max_Dim_Areas',
            'v31': 'Sig_Noise_Max_Dim_Areas',
            'v32': 'Sig_Noise_Max_Dim',
            'v33': 'Allcells',
            'v34': 'V1cells',
            'v35': 'PMcells',
            'v36': 'ALcells',
            'v37': 'RSPcells',
            'v38': 'Taskvars_Allcells',
            }
    
    return abbr_modelnames[version]
