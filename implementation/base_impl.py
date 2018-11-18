## Casey Astiz
## CS Thesis
## Base implementation file
from __future__ import division
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import pandas as pd
import os
import glob
from time import time
import math


def add_features(df):
    """
    Takes in a dataframe and creates some new features from current information
    """
    users = list(set(df['nameOrig'].tolist())) # gets unique users
    # add user's average transaction size
    df['avg_transaction'] = 0
    df['time_btwn_trans'] = 0
    df['interacted_before'] = 0

    means = df.groupby(['nameOrig'])['amount'].mean()

    for i in users:
        df.loc[df.nameOrig == i, 'avg_transaction'] = means[i]


    # add time between user's last transaction and now

    # for i in users:
    #     last_trans = -1
    #     for index, row in df.iterrows():
    #         if row['nameOrig'] == i:
    #             if last_trans != -1:
    #                 row['time_btwn_trans'] = row['step'] - last_trans
    #             else:
    #                 row['time_btwn_trans'] = 0
    #             last_trans = row['step']
    #
    #
    # # add dummy for if user has interacted with nameDest before
    #
    # for i in users:
    #     past_interactions = []
    #     for index, row in df.iterrows():
    #         if row['nameOrig'] == i:
    #             if row['nameDest'] in past_interactions:
    #                 row['interacted_before'] = 1
    #             else:
    #                 past_interactions.append(row['nameDest'])

    return df

def logisticRegression(train_data, test_data, train_lbl, test_lbl):
    """
    Logistic Regression implementation from scikit learn
    """
    # default solver is incredibly slow thats why we change it
    logisticRegr = LogisticRegression(solver = 'lbfgs')
    #logisticRegr = LogisticRegression(solver = 'sag') ## much worse than lbfgs
    logisticRegr.fit(train_data, train_lbl.values.ravel())
    # Returns a NumPy Array
    # Predict for One Observation (image)
    predictions = logisticRegr.predict(test_data)
    accuracy = logisticRegr.score(test_data, test_lbl)
    print("Logistic Regression Accuracy: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1) + "\n")

    return accuracy, precision, recall, F1


# reference for Gaussian work: https://www.kaggle.com/peterkim95/multivariate-gaussian-anomaly-detection
def selectThreshold(train_lbl, pval):

    bestEpsilon = 0.0
    bestF1 = 0.0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        predictions = (pval < epsilon).astype(int)

        _, _, F1 = precision_and_recall(train_lbl['isFraud'].tolist(), predictions.tolist())

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1

def multivariateGaussian(X, mu, sigma):
    m, n = X.shape
    X = X.values - mu.values.reshape(1,n)
    p = (1.0 / (math.pow((2 * math.pi), n / 2.0) * math.pow(np.linalg.det(sigma),0.5))) * np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(sigma)) * X, axis=1))

    return p

def estimateGaussian(X):
    mu = X.mean()
    m, n = X.shape
    sigma = np.zeros((n,n))

    for i in range(0,m):
        sigma = sigma + (X.iloc[i] - mu).values.reshape(n,1).dot((X.iloc[i] - mu).values.reshape(1, n))

    sigma = sigma * (1.0/m) # Use 1.0 instead of 1 to force float conversion

    return mu, sigma

def gaussian(train_data, test_data, train_lbl, test_lbl):
    """
    implementation of a gaussian distribution, which is what Andrew Ng suggests for anomaly detection
    """
    mu, sigma = estimateGaussian(train_data)

    ptrain = multivariateGaussian(train_data, mu, sigma)

    epsilon, F1 = selectThreshold(train_lbl, ptrain)

    print ("Found best epsilon = " + str(epsilon) + ", best F1 = " + str(F1))

    ptest = multivariateGaussian(test_data, mu, sigma)

    predictions = (ptest < epsilon).astype(int)
    test_lbl = np.squeeze(test_lbl.values).astype(int)

    accuracy = accuracy_score(test_lbl, predictions)

    print("Gaussian Accuracy: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl.tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1) + "\n")

    return accuracy, precision, recall, F1

def neural_network(train_data, test_data, train_lbl, test_lbl, layers, epochs=1):
    """
    implementation of neural network with scikit learn
    """
    clf = MLPClassifier(solver='sgd', activation='relu', alpha=.001, hidden_layer_sizes=layers, batch_size=100, random_state=42, max_iter=epochs)
    clf.fit(train_data, train_lbl.values.ravel())
    predictions = clf.predict(test_data)
    accuracy = clf.score(test_data, test_lbl)
    print("Neural Network Accuracy: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1) + "\n")

    return accuracy, precision, recall, F1

def decision_tree(train_data, test_data, train_lbl, test_lbl):
    """
    implementation of decision tree with scikit learn
    """
    decTreeClass = DecisionTreeClassifier(random_state=0, criterion='entropy') # better with entropy criterion
    #Decision Tree Accuracy:
    # 0.9923306348530039
    # precision = 0.9949494949494949 recall = 0.9899497487437185 F1 = 0.9924433249370276

    decTreeClass.fit(train_data, train_lbl)
    predictions = decTreeClass.predict(test_data)
    accuracy = decTreeClass.score(test_data, test_lbl)
    print("Decision Tree Accuracy: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1) + "\n")

    return accuracy, precision, recall, F1


def support_vector_machine(train_data, test_data, train_lbl, test_lbl):
    """
    implementation of support vector machine with scikit learn
    """
    clf = svm.SVC(max_iter=100)
    clf.fit(train_data, train_lbl.values.ravel())
    predictions = clf.predict(test_data)
    accuracy = clf.score(test_data, test_lbl)
    print("SVM: \n" + str(accuracy))

    precision, recall, F1 = precision_and_recall(test_lbl['isFraud'].tolist(), predictions.tolist())

    print("precision = " + str(precision)+ " recall = " + str(recall) + " F1 = " + str(F1) + "\n")

    return accuracy, precision, recall, F1


def precision_and_recall(test_lbl, predictions):
    """
    calculates the precision and recall of the results
    """

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
    """
    This function prints out information about the dataset; df = data frame
    """
    print(df)
    ##proportion of fraud: 0.001290820448
    print(df.groupby('isFraud').count())

    print(df.describe())
    print(df.groupby("type").describe())
    print(df.groupby("isFraud").describe())

def data_split(df, fraud_prop):
    """
    takes data, splits based on fraud_prop and splits into train/test
    """
    mask = df['isFraud'] > 0
    data1 = df[mask] #fraud data
    data0 = df[~mask] #non-fraud data

    n_fraud = data1.shape[0]

    split = None

    # if fraud_prop == .5:
    #     split = data0.sample(n=n_fraud)
    #
    # if fraud_prop == .25:
    #     split = data0.sample(n=n_fraud*3)

    if fraud_prop == 0:
        split = data0
    else:
        split = data0.sample(n=n_fraud*fraud_prop)

    frames = [data1, split]

    data = pd.concat(frames)

    cleaned_data = data[['type','amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']]
    labels = data[['isFraud']]


    x_train, x_test, y_train, y_test = train_test_split(cleaned_data, labels, test_size=1/7.0, random_state=42) # had this set to 0 before

    return x_train, x_test, y_train, y_test

def experiments_nn(df):
    """
    Read data, train neural network models and test models, save accuracy
    """

    epoch_limit = 20

    models1 = [
        (256,64,2),
        (256,256,2),
        (512,64,2)]
    models2 = [
        (256,128,64,2),
        (256,128,128,2),
        (256,256,64,2),
        (512,128,128,2)]
    models3 = [
        (256,256,256,256,2),
        (512,128,128,128,2),
        (512,256,256,128,2),
        (1024,256,128,128,2)]

    file = open('results_nn.txt', 'w')
    file.write('fraud_prop\tnormalized\tx\ty\tz\ta\ttime\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\t15\t16\t17\t18\t19\t20\n')


    for i in [.5, .25]:#range(0, .5, .1):
        x_train, x_test, y_train, y_test = data_split(df, i)

        #non normalized

        for param in models1:
            x,y,z = param
            acc = []
            start_time = time()
            for i in range (20,epoch_limit, 5):
                score = neural_network(x_train, x_test, y_train, y_test, param, i)
                acc.append(str(round(score[0], 3)))
            end_time = time()
            line = '\t'.join([str(i), '0', str(x), str(y), str(z), '\t', str(round(end_time - start_time, 1))]+acc)
            print(line)
            file.write(line + '\n')

        for param in models2:
            x,y,z,a = param
            acc = []
            start_time = time()
            for i in range (1,epoch_limit):
                score = neural_network(x_train, x_test, y_train, y_test, param, i)
                acc.append(str(round(score[0], 3)))
            end_time = time()
            line = '\t'.join([str(i), '0', str(x), str(y), str(z), str(a), '', str(round(end_time - start_time, 1))]+acc)
            print(line)
            file.write(line + '\n')

        for param in models3:
            x,y,z,a,b = param
            acc = []
            start_time = time()
            for i in range(1,epoch_limit):
                score = neural_network(x_train, x_test, y_train, y_test, param, i)
                acc.append(str(round(score[0], 3)))
            end_time = time()
            line = '\t'.join([str(i), '0', str(x), str(y), str(z), str(a), str(b), str(round(end_time - start_time, 1))]+acc)
            print(line)
            file.write(line + '\n')

        # normalized

        x_train = x_train / np.max(x_train) # Normalise data
        x_test = x_train / np.max(x_test) # Normalise data
        y_train = y_train / np.max(y_train) # Normalise data
        y_test = y_train / np.max(y_test) # Normalise data

        for param in models1:
            x,y,z = param
            acc = []
            start_time = time()
            for i in range (20,epoch_limit, 5):
                score = neural_network(x_train, x_test, y_train, y_test, param, i)
                acc.append(str(round(score[0], 3)))
            end_time = time()
            line = '\t'.join([str(i), '1', str(x), str(y), str(z), '\t', str(round(end_time - start_time, 1))]+acc)
            print(line)
            file.write(line + '\n')

        for param in models2:
            x,y,z,a = param
            acc = []
            start_time = time()
            for i in range (1,epoch_limit):
                score = neural_network(x_train, x_test, y_train, y_test, param, i)
                acc.append(str(round(score[0], 3)))
            end_time = time()
            line = '\t'.join([str(i), '1', str(x), str(y), str(z), str(a), '', str(round(end_time - start_time, 1))]+acc)
            print(line)
            file.write(line + '\n')

        for param in models3:
            x,y,z,a,b = param
            acc = []
            start_time = time()
            for i in range(1,epoch_limit):
                score = neural_network(x_train, x_test, y_train, y_test, param, i)
                acc.append(str(round(score[0], 3)))
            end_time = time()
            line = '\t'.join([str(i), '1', str(x), str(y), str(z), str(a), str(b), str(round(end_time - start_time, 1))]+acc)
            print(line)
            file.write(line + '\n')

def experiments(df):
    """
    Read data, train models and test models, save accuracy
    """
    file = open('results.txt', 'w')
    file.write('model\tfraud_prop\tnormalized\ttime\taccuracy\tprecision\trecall\tF1\n')

    for i in range(0, 5, 1):
        x_train, x_test, y_train, y_test = data_split(df, i)

        # test with non-normalized data
        start_time = time()
        accuracy, precision, recall, F1 = logisticRegression(x_train, x_test, y_train, y_test)
        end_time = time()
        line = '\t'.join(['logReg', str(i), '0', str(round(end_time - start_time, 1)), str(accuracy), str(precision), str(recall), str(F1) ])
        file.write(line + '\n')

        start_time = time()
        accuracy, precision, recall, f1 = gaussian(x_train, x_test, y_train, y_test)
        end_time = time()
        line = '\t'.join('gaussian', str(i), '0', str(round(end_time - start_time, 1)), str(accuracy), str(precision), str(recall), str(F1) )
        file.write(line + '\n')

        start_time = time()
        accuracy, precision, recall, F1 = decision_tree(x_train, x_test, y_train, y_test)
        end_time = time()
        line = '\t'.join(['decision_tree', str(i), '0', str(round(end_time - start_time, 1)), str(accuracy), str(precision), str(recall), str(F1)] )
        file.write(line + '\n')

        start_time = time()
        accuracy, precision, recall, F1 = support_vector_machine(x_train, x_test, y_train, y_test)
        end_time = time()
        line = '\t'.join(['svm', str(i), '0', str(round(end_time - start_time, 1)), str(accuracy), str(precision), str(recall), str(F1)] )
        file.write(line + '\n')


        # test with normalized data0

        x_train = x_train / np.max(x_train) # Normalise data
        x_test = x_train / np.max(x_test) # Normalise data
        y_train = y_train / np.max(y_train) # Normalise data
        y_test = y_train / np.max(y_test) # Normalise data

        start_time = time()
        accuracy, precision, recall, F1 = logisticRegression(x_train, x_test, y_train, y_test)
        end_time = time()
        line = '\t'.join(['logReg', str(i), '1', str(round(end_time - start_time, 1)), str(accuracy), str(precision), str(recall), str(F1)] )
        file.write(line + '\n')

        start_time = time()
        accuracy, precision, recall, f1 = gaussian(train_data, test_data, train_lbl, test_lbl)
        end_time = time()
        line = '\t'.join('gaussian', str(i), '1', str(round(end_time - start_time, 1)), str(accuracy), str(precision), str(recall), str(F1) )
        file.write(line + '\n')

        start_time = time()
        accuracy, precision, recall, F1 = decision_tree(x_train, x_test, y_train, y_test)
        end_time = time()
        line = '\t'.join(['decision_tree', str(i), '1', str(round(end_time - start_time, 1)), str(accuracy), str(precision), str(recall), str(F1)] )
        file.write(line + '\n')

        start_time = time()
        accuracy, precision, recall, F1 = support_vector_machine(x_train, x_test, y_train, y_test)
        end_time = time()
        line = '\t'.join(['svm', str(i), '1', str(round(end_time - start_time, 1)), str(accuracy), str(precision), str(recall), str(F1) ])
        file.write(line + '\n')



def main():
    """
    main controlling function
    """

    df = pd.read_csv('credit_data.csv')
    df['type'] = df['type'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    ## Uncomment to print out summary stats
    #descriptive_stat(df)


    df2 = add_features(df)

    df2.to_csv('altered_credit_data.csv')


    ## runs experiments
    #experiments_nn(df)
    #experiments(df)

main()
