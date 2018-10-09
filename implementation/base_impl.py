## Casey Astiz
## CS Thesis
## Base implementation file

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import csv


def logisticRegression(train_data, test_data, train_lbl, test_lbl):
    # default solver is incredibly slow thats why we change it
    logisticRegr = LogisticRegression(solver = 'lbfgs')
    logisticRegr.fit(train_data, train_lbl)
    # Returns a NumPy Array
    # Predict for One Observation (image)
    predictions = logisticRegr.predict(test_data)
    accuracy = logisticRegr.score(test_data, test_lbl)
    print(accuracy)

    return accuracy

def main():
    """main controlling function"""
    with open('credit_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        data = []
        labels = []
        column_labels = []
        for row in csv_reader:
            if (line_count == 0):
                column_labels.append(row)
            else:
                data.append(row)
                labels.append(row[9])
            line_count += 1

    train_data, test_data, train_lbl, test_lbl = train_test_split(data, labels, test_size=1/7.0, random_state=0)

    return logisticRegression(train_data, test_data, train_lbl, test_lbl)
    
main()
