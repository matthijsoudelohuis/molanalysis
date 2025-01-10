
import numpy as np
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
    assert model_name in ['LOGR','SVM','LDA'], 'regularization not supported for model %s' % model_name

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
        elif model_name == 'SVM':
            model = SVM.SVC(kernel='linear', C=lambda_)
        elif model_name == 'LDA':
            model = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lambda_,0,1))

        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        cv_scores[ilambda] = np.mean(scores)
    optimal_lambda = lambdas[np.argmax(cv_scores)]
    # print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))
    if clip:
        optimal_lambda = np.clip(optimal_lambda, 0.03, 166)
    # optimal_lambda = 1
    return optimal_lambda

def my_classifier_wrapper(Xfull,Yfull,model_name='LOGR',kfold=5,lam=None,subtract_shuffle=True,norm_out=False): 
    if model_name == 'LogisticRegression':
        model_name = 'LOGR'
    assert len(Xfull.shape)==2, 'Xfull must be a matrix of samples by features'
    assert len(Yfull.shape)==1, 'Yfull must be a vector'
    assert Xfull.shape[0]==Yfull.shape[0], 'Xfull and Yfull must have the same number of samples'
    assert model_name in ['LOGR','SVM','LDA','GBC']
    assert lam is None or lam > 0
    
    if lam is None and model_name in ['LOGR','SVM','LDA']:
        lam = find_optimal_lambda(Xfull,Yfull,model_name=model_name,kfold=kfold)

    if model_name == 'LOGR':
        model = LOGR(penalty='l1', solver='liblinear', C=lam)
    elif model_name == 'LDA':
        # model = LDA(n_components=1,solver='svd')
        model = LDA(n_components=1,solver='eigen', shrinkage=lam)
    elif model_name == 'GBC': #Gradient Boosting Classifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=10, random_state=0,max_features='sqrt')
    elif model_name == 'SVM':
        model = SVM.SVC(kernel='linear', C=lam)

    # Define the number of folds for cross-validation
    kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

    # Initialize an array to store the decoding performance
    performance = np.full((kfold,), np.nan)
    performance_shuffle = np.full((kfold,), np.nan)

    # Loop through each fold
    for ifold, (train_index, test_index) in enumerate(kf.split(Xfull)):
        # Split the data into training and testing sets
        X_train, X_test = Xfull[train_index], Xfull[test_index]
        y_train, y_test = Yfull[train_index], Yfull[test_index]

        # Train a classification model on the training data with regularization
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate the decoding performance for this fold
        performance[ifold] = accuracy_score(y_test, y_pred)

        # Shuffle the labels and calculate the decoding performance for this fold
        np.random.shuffle(y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        performance_shuffle[ifold] = accuracy_score(y_test, y_pred)

    if subtract_shuffle: # subtract the shuffling performance from the average perf
        performance_avg = np.mean(performance - performance_shuffle)
    else: # Calculate the average decoding performance across folds
        performance_avg = np.mean(performance)
    if norm_out: # normalize to maximal range of performance (between shuffle and 1)
        performance_avg = performance_avg / (1-np.mean(performance_shuffle))
    
    return performance_avg

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



def enc_model_stimwin_wrapper(ses,idx_N,idx_T,modelname='Lasso',optimal_lambda=None,kfold=5,scoring_type = 'r2',
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
        Scoring type to use for cross-validation. Options are 'r2', 'neg_mean_squared_error'. Default is 'r2'.
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

    if optimal_lambda is None:
        y = ses.respmat[np.ix_(idx_N,idx_T)].T

        X = np.stack((
                    ses.trialdata['signal'][idx_T].to_numpy(),
                    ses.trialdata['lickResponse'][idx_T].to_numpy(),
                    ses.respmat_runspeed[0,idx_T],
                    ses.trialdata['trialNumber'][idx_T]
                    ), axis=1)

        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        # Find the optimal regularization strength (lambda)
        lambdas = np.logspace(-6, 0, 10)
        cv_scores = np.zeros((len(lambdas),))
        for ilambda, lambda_ in enumerate(lambdas):
            model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)
            # model = ElasticNet(alpha=lambda_,l1_ratio=0.9)
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring_type)
            cv_scores[ilambda] = np.median(scores)
        optimal_lambda = lambdas[np.argmax(cv_scores)]

    # Train a regression model on the training data with regularization
    # model = ElasticNet(alpha=optimal_lambda,l1_ratio=0.9)
    model = getattr(sklearn.linear_model,modelname)(alpha=optimal_lambda)

    variables   = ['signal','lickresponse','runspeed','trialnumber']
    V           = len(variables)
    K           = len(ses.trialdata)
    N           = len(ses.celldata)
    N_idx       = np.sum(idx_N)
    weights     = np.full((N,V),np.nan)
    error       = np.full((N),np.nan)
    y_hat       = np.full((N,K),np.nan)

    y = ses.respmat[np.ix_(idx_N,idx_T)].squeeze().T # Get the neural response data for this bin

    # Define the X predictors
    X = np.stack((ses.trialdata['signal'][idx_T].to_numpy(),
        ses.trialdata['lickResponse'][idx_T].to_numpy(),
        ses.respmat_runspeed[0,idx_T],
        ses.trialdata['trialNumber'][idx_T]), axis=1)

    X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

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
            y_pred              = model.predict(X_test)
            fold_error[ifold,:]    = r2_score(y_test, y_pred, multioutput='raw_values')
            
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
    
    return error, weights, y_hat




def enc_model_spatial_wrapper(ses,sbins,idx_N,idx_T,modelname='Lasso',optimal_lambda=None,kfold=5,scoring_type = 'r2',
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

    if optimal_lambda is None:
        y = ses.respmat[np.ix_(idx_N,idx_T)].T

        X = np.stack((
                    ses.trialdata['signal'][idx_T].to_numpy(),
                    ses.trialdata['lickResponse'][idx_T].to_numpy(),
                    ses.respmat_runspeed[0,idx_T],
                    ses.trialdata['trialNumber'][idx_T]
                    ), axis=1)

        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

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
    model = getattr(sklearn.linear_model,modelname)(alpha=optimal_lambda)

    variables   = ['signal','lickresponse','runspeed','trialnumber']
    S           = len(sbins)
    V           = len(variables)
    K           = len(ses.trialdata)
    N           = len(ses.celldata)
    N_idx       = np.sum(idx_N)
    weights     = np.full((N,S,V),np.nan)
    error       = np.full((N,S),np.nan)
    y_hat       = np.full((N,S,K),np.nan)


    for ibin, bincenter in enumerate(sbins):    # Loop over each spatial bin
        y = ses.stensor[np.ix_(idx_N,idx_T,[ibin])].squeeze().T # Get the neural response data for this bin

        # Define the X predictors
        X = np.stack((ses.trialdata['signal'][idx_T].to_numpy(),
            ses.trialdata['lickResponse'][idx_T].to_numpy(),
            ses.runPSTH[idx_T,ibin],
            ses.trialdata['trialNumber'][idx_T]), axis=1)

        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

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
                y_pred              = model.predict(X_test)
                fold_error[ifold,:]    = r2_score(y_test, y_pred, multioutput='raw_values')
                
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