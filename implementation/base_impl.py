## Casey Astiz
## CS Thesis
## Base implementation file
from __future__ import division
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import numpy as np
#import matplotlib.pyplot as plt
import csv
import pandas as pd



def logisticRegression(train_data, test_data, train_lbl, test_lbl):
    # default solver is incredibly slow thats why we change it
    logisticRegr = LogisticRegression(solver = 'lbfgs')
    logisticRegr.fit(train_data, train_lbl.values.ravel())
    # Returns a NumPy Array
    # Predict for One Observation (image)
    predictions = logisticRegr.predict(test_data)
    accuracy = logisticRegr.score(test_data, test_lbl)
    print("Logistic Regression Accuracy: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1))

    return accuracy

def neural_network(train_data, test_data, train_lbl, test_lbl):
    """implementation of neural network with scikit learn"""
    clf = MLPClassifier(solver='lbfgs', alpha=.01, hidden_layer_sizes=(5, 2), batch_size=100, random_state=1)
    clf.fit(train_data, train_lbl.values.ravel())
    predictions = clf.predict(test_data)
    accuracy = clf.score(test_data, test_lbl)
    print("Neural Network Accuracy: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1))

    return accuracy

def decision_tree(train_data, test_data, train_lbl, test_lbl):
    """implementation of decision tree with scikit learn"""
    decTreeClass = DecisionTreeClassifier(random_state=0)
    decTreeClass.fit(train_data, train_lbl)
    predictions = decTreeClass.predict(test_data)
    accuracy = decTreeClass.score(test_data, test_lbl)
    print("Decision Tree Accuracy: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1))

    return accuracy

    
def support_vector_machine(train_data, test_data, train_lbl, test_lbl):
    """implementation of support vector machine with scikit learn"""
    clf = svm.SVC()
    clf.fit(train_data, train_lbl.values.ravel())
    predictions = clf.predict(test_data)
    accuracy = clf.score(test_data, test_lbl)
    print("Decision Tree Accuracy: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1))

    return accuracy   


def precision_and_recall(test_lbl, predictions):
    """calculates the precision and recall of the results"""

    # Precision: A measure of a classifiers exactness.
    # Recall: A measure of a classifiers completeness
    # F1 Score (or F-score): A weighted average of precision and recall.
    
    tp = 0
    fp = 0
    fn = 0
    precision = 0
    recall = 0
    F1 = 0

    for i in range(len(test_lbl)):
        if(test_lbl[i] == predictions[i] and predictions[i] == 1):
            tp += 1
        else:
            if(test_lbl[i] != predictions[i] and test_lbl[i] == 1):
                fp += 1
            elif(test_lbl[i] != predictions[i] and test_lbl[i] == 0):
                fn += 1
            else:
                continue
    
    if (tp != 0):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * (precision * recall) / (precision + recall)

    # calculate the F1 score

    return precision, recall, F1


def descriptive_stat(df):
    """This function prints out information about the dataset; df = data frame"""    
    print(df)
    ##proportion of fraud: 0.001290820448
    print(df.groupby('isFraud').count())
    
    print(df.describe())
    print(df.groupby("type").describe())
    print(df.groupby("isFraud").describe())

def main():
    """main controlling function"""

    df = pd.read_csv('credit_data.csv')
    df['type'] = df['type'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    
    data = df[['type','amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    labels = df[['isFraud']]

    # descriptive_stat(df)


    train_data, test_data, train_lbl, test_lbl = train_test_split(data, labels, test_size=1/7.0, random_state=0)

    # runs experiments
    
    #logisticRegression(train_data, test_data, train_lbl, test_lbl)
    #neural_network(train_data, test_data, train_lbl, test_lbl)
    #decision_tree(train_data, test_data, train_lbl, test_lbl)
    support_vector_machine(train_data, test_data, train_lbl, test_lbl)
    
main()


# ToDo: look up methods for unbalanced classes
# get simulator working or get the full dataset
# possibly do feature engineering
# recursive neural networks
# read paysim paper and try to create more data

