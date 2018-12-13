# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:48:32 2018

@author: nitesh.yadav
"""
import Donors_Search as ds
def main():
    # data load from csv file using pandas
    data_frame = ds.Data_Load()
    # Data exploration
    ds.Data_Exploration(data_frame) 
    # Preprocess data
    features_final, income = ds.Preprocess_Data(data_frame)
    # Split data into training and testing sets
    features_train, features_test, labels_train, labels_test = ds.Test_Train_Split(features_final, income)
    # Naive predictor scores
    accuracy, fscore = ds.Model_Performance(income)
    # Initial Model Evaluation
    ds.Initial_Model(features_train, labels_train, features_test, labels_test, accuracy, fscore)
    # Model tuning
    ds.Model_Tuning(features_train, labels_train, features_test, labels_test)
           
if __name__ == "__main__":
    main()