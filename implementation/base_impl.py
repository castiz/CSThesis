## Casey Astiz
## CS Thesis
## Base implementation file
from __future__ import division
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
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

    precision, recall = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall))

    return accuracy

def neural_network(train_data, test_data, train_lbl, test_lbl):
    """implementation of neural network with scikit learn"""
    clf = MLPClassifier(solver='lbfgs', alpha=.01, hidden_layer_sizes=(5, 2), batch_size=100, random_state=1)
    clf.fit(train_data, train_lbl.values.ravel())
    predictions = clf.predict(test_data)
    accuracy = clf.score(test_data, test_lbl)
    print("Neural Network Accuracy: \n" + str(accuracy))

    precision, recall = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall))

    return accuracy

def decision_tree(train_data, test_data, train_lbl, test_lbl):
    decTreeClass = DecisionTreeClassifier(random_state=0)
    decTreeClass.fit(train_data, train_lbl)
    predictions = decTreeClass.predict(test_data)
    accuracy = decTreeClass.score(test_data, test_lbl)
    print("Decision Tree Accuracy: \n" + str(accuracy))

    precision, recall = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall))

    return accuracy

    

def precision_and_recall(test_lbl, predictions):
    """calculates the precision and recall of the results"""
    tp = 0
    fp = 0
    fn = 0
    precision = 0
    recall = 0

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

    return precision, recall

def main():
    """main controlling function"""
##    data = []
##    labels = []
##    column_labels = []
##    with open('credit_data.csv') as csv_file:
##        csv_reader = csv.reader(csv_file, delimiter=',')
##        line_count = 0
##        for row in csv_reader:
##            if (line_count == 0):
##                column_labels.append(row)
##                line_count += 1
##            elif (line_count < 10):
##                data.append(row)
##                labels.append(row[9])
##                line_count += 1
##            else:
##                break

    df = pd.read_csv('credit_data.csv')
    df['type'] = df['type'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    #print(df)
    ##proportion of fraud: 0.001290820448
    #print(df.groupby('isFraud').count())
    

    
    data = df[['type','amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    labels = df[['isFraud']]

    train_data, test_data, train_lbl, test_lbl = train_test_split(data, labels, test_size=1/7.0, random_state=0)

    logisticRegression(train_data, test_data, train_lbl, test_lbl)
    neural_network(train_data, test_data, train_lbl, test_lbl)
    decision_tree(train_data, test_data, train_lbl, test_lbl)
    
main()
