# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:51:53 2018

@author: nitesh.yadav
"""

import pandas as pd
import visuals as vs
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import clone

def Data_Load():
    "Loading data from csv file"
    try:
        full_data_frame = pd.read_csv(r"C:\Users\nitesh.yadav\Documents\machine-learning-master\projects\Done\finding_donors\census.csv")
    except FileNotFoundError:
        print("File 'census.csv' does not exist, please check the provided path.")
    return full_data_frame

def Data_Exploration(data_frame):
    """Explore basic stistics of data-set"""
    n_records =  data_frame.shape[0]
    n_greater_50k = data_frame.loc[data_frame['income'] == '>50K'].shape[0]
    n_at_most_50k = data_frame.loc[data_frame['income'] == '<=50K'].shape[0]
    greater_percent = (n_greater_50k / n_records) * 100
    print("Total number of records: {}".format(n_records))
    print("Individuals making more than $50,000: {}".format(n_greater_50k))
    print("Individuals making at most $50,000: {}".format(n_at_most_50k))
    print("Percentage of individuals making more than $50,000: {} %".format(greater_percent))
    
def Preprocess_Data(data_frame):
    """Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as preprocessing."""
    income_raw = data_frame['income']
    features_raw = data_frame.drop('income', axis = 1)
    vs.Distribution(data_frame) 
    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data = features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x : np.log(x + 1))
    # Visualize the new log distributions
    vs.Distribution(features_log_transformed, transformed = True)
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed[numerical])
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    # Show an example of a record with scaling applied
    display(features_log_minmax_transform.head(n = 5))
    # categorical features to numerical feature transformation
    features_final = pd.get_dummies(features_log_minmax_transform)
    income_raw.loc[income_raw == '>50K'] = 1
    income_raw.loc[income_raw == '<=50K'] = 0
    income = income_raw
    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))
    print(encoded)
    return features_final, income
    
def Test_Train_Split(features_final, income):
    """We will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing."""
    features_train, features_test, labels_train, labels_test = train_test_split(features_final, income, test_size = 0.2, random_state = 0) 
    # Show the results of the split
    print("Training set has {} samples.".format(features_train.shape[0]))
    print("Testing set has {} samples.".format(features_test.shape[0]))
    return features_train, features_test, labels_train, labels_test

def Model_Performance(income):
    """
    TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data encoded to numerical values done in the data preprocessing step.
    FP = income.count() - TP # Specific to the naive case
    TN = 0 # No predicted negatives in the naive case
    FN = 0 # No predicted negatives in the naive case
    """
    tp = np.sum(income)
    fp = income.count()
    tn = 0
    fn = 0
    accuracy = tp / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 1.25 * precision * recall / (.25 * precision + recall)
    print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
    return accuracy, fscore

def Train_Predict(learner, sample_size, features_train, labels_train, features_test, labels_test):
    """
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - features_train: features training set
       - labels_train: income training set
       - features_test: features testing set
       - labels_test: income testing set
    """
    results = {}
    start = time()
    learner.fit(features_train[:sample_size], labels_train[:sample_size])
    end = time()
    results['train_time'] = end - start
    
    start = time()
    predictions_test = learner.predict(features_test)
    predictions_train = learner.predict(features_train[:300])
    end = time()
    results['pred_time'] = end - start
    results['acc_train'] = accuracy_score(labels_train[:300], predictions_train)
    results['acc_test'] = accuracy_score(labels_test, predictions_test)
    results['f_train'] = fbeta_score(labels_train[:300], predictions_train, .5)
    results['f_test'] = fbeta_score(labels_test, predictions_test, .5)
    print("{} trained on {} samples".format(learner.__class__.__name__, sample_size))
       
    return results

def Initial_Model(features_train, labels_train, features_test, labels_test, accuracy, fscore):
    """
    Import the three supervised learning models you've discussed in the previous section.
    Initialize the three models and store them in 'clf_A', 'clf_B', and 'clf_C'.
    """
    clf_A = KNeighborsClassifier()
    clf_B = DecisionTreeClassifier()
    clf_C = SVC()
    
    samples_100 = len(labels_train)
    samples_10 = int(.1 * samples_100)
    samples_1 = int(.1 * samples_10)
    # collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = Train_Predict(clf, samples, features_train, labels_train, features_test, labels_test)
    # Run metrics visualization for the three supervised learning models chosen
    vs.Evaluate(results, accuracy, fscore)
    
def Model_Tuning(features_train, labels_train, features_test, labels_test):
    """
    perform a grid search optimization for the model over the entire training set (features_train and labels_train) by tuning at least one parameter to improve upon the untuned model's F-score.
    """
    clf = DecisionTreeClassifier()
    parameters = {'min_samples_split': [2, 4, 6, 8], 'min_samples_leaf': [1, 2, 3, 4]}
    scorer = make_scorer(fbeta_score, beta = 0.5)
    grid_obj = GridSearchCV(clf, parameters, scoring = scorer)
    grid_fit = grid_obj.fit(features_train, labels_train)
    best_clf = grid_fit.best_estimator_
    predictions = (clf.fit(features_train, labels_train)).predict(features_test)
    best_predictions = best_clf.predict(features_test)
    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(labels_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(labels_test, predictions, beta = 0.5)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(labels_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(labels_test, best_predictions, beta = 0.5)))
    # Feature Relevance Observation
    model = ExtraTreesClassifier()
    model.fit(features_train, labels_train)
    importances = model.feature_importances_
    vs.feature_plot(importances, features_train, labels_train)
    # Feature selection
    X_train_reduced = features_train[features_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
    X_test_reduced = features_test[features_test.columns.values[(np.argsort(importances)[::-1])[:5]]]
    # Train on the "best" model found from grid search earlier
    clf = (clone(best_clf)).fit(X_train_reduced, labels_train)
    # Make new predictions
    reduced_predictions = clf.predict(X_test_reduced)
    # Report scores from the final model using both versions of data
    print("Final Model trained on full data\n------")
    print("Accuracy on testing data: {:.4f}".format(accuracy_score(labels_test, best_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(labels_test, best_predictions, beta = 0.5)))
    print("\nFinal Model trained on reduced data\n------")
    print("Accuracy on testing data: {:.4f}".format(accuracy_score(labels_test, reduced_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(labels_test, reduced_predictions, beta = 0.5)))





