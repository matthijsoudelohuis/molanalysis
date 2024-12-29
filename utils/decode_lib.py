
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LOGR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm as SVM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# import sklearn


def find_optimal_lambda(X,y,model_name='LOGR',kfold=5,clip=False):
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

    # Calculate the average decoding performance across folds
    performance_avg = np.mean(performance)
    if subtract_shuffle: # subtract the shuffling performance from the average perf
        performance_avg = np.mean(performance_avg - performance_shuffle)
    if norm_out: # normalize to maximal range of performance (between shuffle and 1)
        performance_avg = performance_avg / (1-np.mean(performance_shuffle))
    
    return performance_avg