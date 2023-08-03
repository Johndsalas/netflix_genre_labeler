'''                                                                 File for holding model functions                                                           '''

import pandas as pd
import regex as re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def get_counts(train_X, validate_X, test_X):
    ''' Take in X values for train, validate and test
        Return values vectorized by count'''

    cv = CountVectorizer()

    train_counts = cv.fit_transform(train_X['description'])
    validate_counts = cv.transform(validate_X['description'])
    test_counts = cv.transform(test_X['description'])

    # Retrieve the feature names (words) from the CountVectorizer
    feature_names = cv.get_feature_names()

    # Create DataFrames for train_counts and validate_counts
    train_counts = pd.DataFrame(train_counts.todense(), columns=feature_names)
    validate_counts = pd.DataFrame(validate_counts.todense(), columns=feature_names)
    test_counts = pd.DataFrame(test_counts.todense(), columns=feature_names)

    return train_counts, validate_counts, test_counts

def get_acc_table(train_X, train_y, validate_X, validate_y):

    ''' take in train data split into X and y, Validate data split into X and y
        print table of accuracy scores for train and validate data when run on each classifier in the list'''

    clf_lst = [DecisionTreeClassifier(random_state = 411),
               RandomForestClassifier(random_state = 411),
               KNeighborsClassifier(),
               LogisticRegression(random_state = 411)]
    
    label_lst = ["Decision Tree ",
                 "Random Forest ",
                 "KNeighbors ",
                 "Logistic "]
    
    index = 0

    print('Accuracy Scores')
    print('---------------')

    for clf in clf_lst:

        label = label_lst[index]
        
        obj = clf.fit(train_X, train_y)
        train_score = obj.score(train_X, train_y)
        validate_score = obj.score(validate_X, validate_y)

        print(f'{label} Train: {train_score} Validate: {validate_score}')

        index += 1