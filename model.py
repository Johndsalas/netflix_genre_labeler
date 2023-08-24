'''                                                                 File for holding prediction and model functions                                                           '''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

########################################################################## Pre-Modeling Prediction Functions #######################################################

def contains_matching_item(value_set, unique_set):
    '''takes in two sets returns if the value_set contains an item that is also in the unique_set'''
    
    return len(value_set.difference(unique_set)) != len(value_set)


def is_present_value(value, gen_set, non_gen_set):
    '''takes in a pandas value representing a film description and two sets 
       returns prediction value based on weather words in the description are in either of the sets'''
    
    # convert value to a set
    value = set(value.split(' '))
    
    # return prediction value based on input sets that contain words that are in the value set
    if contains_matching_item(value, gen_set) and (contains_matching_item(value, non_gen_set)):
        
        return "contains both"
    
    elif contains_matching_item(value, gen_set):
        
        return True
    
    elif contains_matching_item(value, non_gen_set):
        
        return False
    
    else:
        
        return "contains niether";
    

def by_number_value(value, gen_set, non_gen_set):
    '''takes in a pandas value representing a film description and two sets 
       returns value based on number of words in the description appearing in each set'''
    
    # split value into a list of words
    value = value.split(' ')
    
    gen_unique_count = 0
    
    non_gen_unique_count = 0
    
    # get count of number of words in value list that are in gen_set and non_gen set
    for item in value:
        
        if item in gen_set:
            
            gen_unique_count += 1
            
        if item in non_gen_set:
            
            non_gen_unique_count += 1
            
    # return value based on which set contains the most words from the description
    
    if gen_unique_count == non_gen_unique_count:
        
        return "equal count"
    
    elif gen_unique_count > non_gen_unique_count:
        
        return True
    
    elif gen_unique_count < non_gen_unique_count:
        
        return False
    
    else:
        
        return "did not catch";
    
    
def get_predictions_presence(df, gen_uniques, non_gen_uniques):
    '''Takes in df containing film description set of in-genre words and set of out of genre words'''
    
    # create prediction column for dataframe predicting comedy based on presence of unique words 
    df['prediction'] = df.description.apply(lambda value : is_present_value(value, gen_uniques, non_gen_uniques)) 
    df['evaluation'] = df['comedy'] == df['prediction']
    
    print("Prediction Value Counts")
    print("-----------------------")
    print(df.prediction.value_counts())
    print()
    print("Evaluation Results")
    print("-----------------------")
    print(df.evaluation.value_counts())
    print()
    print("Overall Accuracy")
    print("-----------------------")
    print(round(df.evaluation.mean(), 2) * 100)
    print()
    print("Override Accuracy")
    print("-----------------------")
    print(round(df.evaluation[(df.prediction == True) | (df.prediction == False)].mean(), 2) * 100)
    

def get_predictions_number(df, gen_uniques, non_gen_uniques):
    
    df['prediction'] = df.description.apply(lambda value : by_number_value(value, gen_uniques, non_gen_uniques)) 
    df['evaluation'] = df['comedy'] == df['prediction']
    
    print("Prediction Value Counts")
    print("-----------------------")
    print(df.prediction.value_counts())
    print()
    print("Evaluation Results")
    print("-----------------------")
    print(df.evaluation.value_counts())
    print()
    print("Overall Accuracy")
    print("-----------------------")
    print(round(df.evaluation.mean(), 2) * 100)
    print()
    print("Override Accuracy")
    print("-----------------------")
    print(round(df.evaluation[(df.prediction == True) | (df.prediction == False)].mean(), 2) * 100)


######################################################################### Functions for Modeling ####################################################################################


def get_vector_counts(train_X, validate_X, test_X):
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
                 "K Neighbors ",
                 "Logistic Regression"]
    
    index = 0

    print('Accuracy Scores')
    print('---------------')

    for clf in clf_lst:

        label = label_lst[index]
        
        obj = clf.fit(train_X, train_y)
        train_score = obj.score(train_X, train_y)
        validate_score = obj.score(validate_X, validate_y)

        print(f'{label} Train: {round(train_score, 4) * 100}% Validate: {round(validate_score,4) *100}%')

        index += 1


def remove_low_freq(df, freq_dict, threshold):
    ''' removes columns from df if abs val of relative freq is equal or less than input number '''
    
    for col in df.columns.to_list():
        
        if abs(freq_dict[col]) <= threshold:
            
            df = df.drop(columns=[col])
    
    return df


def get_acc_after_freq_drop(train_X, train_y, validate_X, validate_y, freq_lst, threshold):
    ''' get accuracy table after dropping words from training data 
        that have a relative freq less than the input number'''
    
    train_X = remove_low_freq(train_X, dict(freq_lst), threshold)
    validate_X = remove_low_freq(validate_X, dict(freq_lst), threshold)

    print(f"Drop threshold is {threshold}")
    get_acc_table(train_X, train_y, validate_X, validate_y)