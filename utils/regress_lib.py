
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
    lambdas = np.logspace(-4, 4, 10)
    cv_scores = np.zeros((len(lambdas),))
    for ilambda, lambda_ in enumerate(lambdas):
        
        if model_name == 'LOGR':
            model = LOGR(penalty='l1', solver='liblinear', C=lambda_)
            score_fun = 'accuracy'
        elif model_name == 'SVM':
            model = SVM.SVC(kernel='linear', C=lambda_)
            score_fun = 'accuracy'
        elif model_name == 'LDA':
            model = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lambda_,0,1))
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
                          scoring_type=None,norm_out=False):
    if model_name == 'LogisticRegression':
        model_name = 'LOGR'
    assert len(Xfull.shape)==2, 'Xfull must be a matrix of samples by features'
    assert len(Yfull.shape)==1, 'Yfull must be a vector'
    assert Xfull.shape[0]==Yfull.shape[0], 'Xfull and Yfull must have the same number of samples'
    assert model_name in ['LOGR','SVM','LDA','Ridge','Lasso','LinearRegression'], 'regularization not supported for model %s' % model_name
    assert lam is None or lam > 0
    
    if lam is None and model_name in ['LOGR','SVM','LDA']:
        lam = find_optimal_lambda(Xfull,Yfull,model_name=model_name,kfold=kfold)

    if model_name == 'LOGR':
        model = LOGR(penalty='l1', solver='liblinear', C=lam)
    elif model_name == 'SVM':
        model = SVM.SVC(kernel='linear', C=lam)
    elif model_name == 'LDA':
        model = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))
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
    weights             = np.full((kfold,np.shape(Xfull)[1]), np.nan)
    projs               = np.full((np.shape(Xfull)[0]), np.nan)

    # Loop through each fold
    for ifold, (train_index, test_index) in enumerate(kf.split(Xfull)):
        # Split the data into training and testing sets
        X_train, X_test = Xfull[train_index], Xfull[test_index]
        y_train, y_test = Yfull[train_index], Yfull[test_index]

        # Train a classification model on the training data with regularization
        model.fit(X_train, y_train)

        weights[ifold,:] = model.coef_

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate the decoding performance for this fold
        performance[ifold] = score_fun(y_test, y_pred)

        # Shuffle the labels and calculate the decoding performance for this fold
        np.random.shuffle(y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        projs[test_index] = y_pred

        performance_shuffle[ifold] = score_fun(y_test, y_pred)

    if subtract_shuffle: # subtract the shuffling performance from the average perf
        performance_avg = np.mean(performance - performance_shuffle)
    else: # Calculate the average decoding performance across folds
        performance_avg = np.mean(performance)
    if norm_out: # normalize to maximal range of performance (between shuffle and 1)
        performance_avg = performance_avg / (1-np.mean(performance_shuffle))
    weights = np.nanmean(weights, axis=0) #average across folds

    return performance_avg,weights,projs

def prep_Xpredictor(X,y):
    X           = zscore(X, axis=0,nan_policy='omit')
    idx_nan     = ~np.all(np.isnan(X),axis=1)
    X           = X[idx_nan,:]
    y           = y[idx_nan]
    X[:,np.all(np.isnan(X),axis=0)] = 0
    X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
    y           = np.nan_to_num(y,nan=np.nanmean(y,axis=0,keepdims=True))
    return X,y

# def prep_Xpredictor(X,y):
#     X           = X[:,~np.all(np.isnan(X),axis=0)] #
#     idx_nan     = ~np.all(np.isnan(X),axis=1)
#     X           = X[idx_nan,:]
#     y           = y[idx_nan]
#     X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
#     X           = zscore(X, axis=0)
#     X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
#     return X,y

def get_predictors(ses,ibin=0):
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

def get_predictors_from_modelversion(version='v1'):
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

    modelvars   = get_predictors_from_modelversion(version)

    V           = len(modelvars)
    K           = len(ses.trialdata)
    N           = len(ses.celldata)
    N_idx       = np.sum(idx_N)
    weights     = np.full((N,V),np.nan)
    error       = np.full((N),np.nan)
    y_hat       = np.full((N,K),np.nan)

    y           = ses.respmat[np.ix_(idx_N,idx_T)].T

    X,allvars   = get_predictors(ses)               # get all predictors
    X           = X[:,np.isin(allvars,modelvars)] #get only predictors of interest
    X           = X[idx_T,:]                     #get only trials of interest
   
    X,y         = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

    if optimal_lambda is None:
        # Find the optimal regularization strength (lambda)
        lambdas = np.logspace(-6, 0, 10)
        cv_scores = np.zeros((len(lambdas),))
        for ilambda, lambda_ in enumerate(lambdas):
            model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)
            # model = ElasticNet(alpha=lambda_,l1_ratio=0.9)
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring_type)
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
            
            if subtr_shuffle:
                print('Shuffling labels not yet implemented')
                # # Shuffle the labels and calculate the decoding performance for this fold
                # np.random.shuffle(y_train)
                # model.fit(X_train, y_train)
                # y_pred = model.predict(X_test)
                # fold_r2_shuffle[ifold] = accuracy_score(y_test, y_pred)
    
        # Calculate the average decoding performance across folds
        error[idx_N] = np.nanmean(fold_error, axis=0)
        weights[idx_N,:] = np.nanmean(fold_weights, axis=0)

    else:   
        # Without cross-validation
        model.fit(X, y)
        y_pred = model.predict(X)
        error[idx_N] = r2_score(y, y_pred, multioutput='raw_values')
        y_hat[np.ix_(idx_N,idx_T)] = y_pred.T
        weights[idx_N,:] = model.coef_
    
    return error, weights, y_hat, modelvars




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

    modelvars   = get_predictors_from_modelversion(version)
    
    if optimal_lambda is None:
        y           = ses.respmat[np.ix_(idx_N,idx_T)].T

        X,allvars   = get_predictors(ses)               # get all predictors
        X           = X[:,np.isin(allvars,modelvars)] #get only predictors of interest
        X           = X[idx_T,:]                     #get only trials of interest

        X,y         = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        # Find the optimal regularization strength (lambda)
        lambdas = np.logspace(-6, 0, 20)
        cv_scores = np.zeros((len(lambdas),))
        for ilambda, lambda_ in enumerate(lambdas):
            model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)
            # model = ElasticNet(alpha=lambda_,l1_ratio=0.9)
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring_type)
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
    y_hat       = np.full((N,S,K),np.nan)

    for ibin, bincenter in enumerate(sbins):    # Loop over each spatial bin
        y = ses.stensor[np.ix_(idx_N,idx_T,[ibin])].squeeze().T # Get the neural response data for this bin

        X,allvars   = get_predictors(ses,ibin)               # get all predictors
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
                
                if subtr_shuffle:
                    print('Shuffling labels not yet implemented')
                    # # Shuffle the labels and calculate the decoding performance for this fold
                    # np.random.shuffle(y_train)
                    # model.fit(X_train, y_train)
                    # y_pred = model.predict(X_test)
                    # fold_r2_shuffle[ifold] = accuracy_score(y_test, y_pred)
        
            # Calculate the average decoding performance across folds
            error[idx_N,ibin] = np.nanmean(fold_error, axis=0)
            weights[idx_N,ibin,:] = np.nanmean(fold_weights, axis=0)

        else:   
            # Without cross-validation
            model.fit(X, y)
            y_pred = model.predict(X)
            error[idx_N,ibin] = r2_score(y, y_pred, multioutput='raw_values')
            y_hat[np.ix_(idx_N,[ibin],idx_T)] = y_pred.T[:,np.newaxis,:]
            weights[idx_N,ibin,:] = model.coef_
    
    return error, weights, y_hat