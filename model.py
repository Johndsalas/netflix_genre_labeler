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
        
        return "contains niether"
    

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
        
        return "did not catch"
    
    
def get_predictions_presence(df, gen_uniques, non_gen_uniques):
    '''Takes in df containing film description set of in-genre words and set of out of genre words
       Predicts if the film is comedy or not in genre based on presents of unique words
       Prints accuracy chart evaluating those predictions'''
    
    # create prediction column for dataframe predicting comedy based on presence of unique words 
    df['prediction'] = df.description.apply(lambda value : is_present_value(value, gen_uniques, non_gen_uniques)) 
    df['evaluation'] = df['comedy'] == df['prediction']
    
    # print chart evaluating predictions
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
    '''Takes in df containing film description set of in-genre words and set of out of genre words
       Predicts if the film is in-genre or not in genre based on number of unique words
       Prints accuracy chart evaluating those predictions'''

    # create prediction column for dataframe predicting comedy based on number of unique words 
    df['prediction'] = df.description.apply(lambda value : by_number_value(value, gen_uniques, non_gen_uniques)) 
    df['evaluation'] = df['comedy'] == df['prediction']
    
    # print chart evaluating predictions
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


############################################################################# Modeling Functions ####################################################################################


def get_vectorized_data(train_X, validate_X, test_X, vector):
    ''' Take in X values for train, validate and test
        Return values vectorized by specified vectorization method'''

    cv = vector

    train_X = cv.fit_transform(train_X['description'])
    validate_X = cv.transform(validate_X['description'])
    test_X = cv.transform(test_X['description'])

    # Retrieve the feature names (words) from the Vectorizer
    feature_names = cv.get_feature_names()

    # Create DataFrames for train_counts and validate_counts
    train_X = pd.DataFrame(train_X.todense(), columns=feature_names)
    validate_X = pd.DataFrame(validate_X.todense(), columns=feature_names)
    test_X = pd.DataFrame(test_X.todense(), columns=feature_names)

    return train_X, validate_X, test_X


def evaluate_model(train_X, train_y, validate_X, validate_y, clf, label, threshold):
    ''' Evaluates Model on train and validate and prints line of evaluation chart'''
    
    obj = clf.fit(train_X, train_y)

    train_score = obj.score(train_X, train_y)
    train_score = str(round(train_score * 100, 2)) + "%"

    validate_score = obj.score(validate_X, validate_y)
    validate_score = str(round(validate_score * 100, 2)) + "%"

    print("| {0:^9} | {1:^21}|{2:^19}|{3:^22}|".format(threshold, label, train_score, validate_score))
    
    
def remove_low_freq(df_X, freq_dict, threshold):
    ''' Restricts columns in vectorized df to those that have an relative frequency grater than threshold numbers from zero'''
    
    df_cols = list(df_X.columns)

    new_cols = [col for col in df_cols if abs(freq_dict[col]) > threshold]
 
    return df_X[new_cols]


def get_acc_tables(train_X, train_y, validate_X, validate_y, freq_dict):
    ''' take in train data split into X and y, Validate data split into X and y
        print table of accuracy scores for train and validate data when run on each classifier in the list'''

    # hold original input for train and valadate X data frames
    original_train = train_X
    original_validate = validate_X
    
    # List of classifiers and labels
    clf_lst = [DecisionTreeClassifier(random_state = 411),
               RandomForestClassifier(random_state = 411),
               KNeighborsClassifier(),
               LogisticRegression(random_state = 411)]
    
    label_lst = ["Decision Tree ",
                 "Random Forest ",
                 "K Neighbors ",
                 "Logistic Regression"]
    
    index = 0

    # Itterate through classifiers and evaluate each on train and validate 
    # Then print results in a table 
    for clf in clf_lst:
        
        # set train and validate df's to original input 
        train_X = original_train
        validate_X = original_validate

        # Evaluate model with full vectorized data
        label = label_lst[index]
        
        threshold = "N/A"

        # print beginning of first table
        print()
        print(" _____________________________________________________________________________ ")
        print("| Threshold |        Model         | Accuracy On Trian | Accuracy on Validate |")
        print(" ----------------------------------------------------------------------------- ")
        
        # Build evaluate and print results of model in table
        evaluate_model(train_X, train_y, validate_X, validate_y, clf, label, threshold)

        # Itterate through thresholds 
        for thresh in [0, 1, 2, 3, 4, 5]:
             
            train_X = remove_low_freq(train_X, freq_dict, thresh)
            validate_X = remove_low_freq(validate_X, freq_dict, thresh)
            
            # Build evaluate and print results of model in table
            evaluate_model(train_X, train_y, validate_X, validate_y, clf, label, thresh)
            
        # print bottom of table and beginning of the next table
        print(" ----------------------------------------------------------------------------- ")
            
        index += 1  


def evaluate_model_test(train_X, train_y, test_X, test_y, freq_dict, threshold):
    ''' Evaluates Model on test'''
    
    # remove low frequency values
    train_X = remove_low_freq(train_X, freq_dict, threshold)
    test_X = remove_low_freq(train_X, freq_dict, threshold)
    
    # Train model on train data
    obj = LogisticRegression(random_state = 411).fit(train_X, train_y)

    # Evaluate model on test data and print result
    test_score = obj.score(test_X, test_y)
    test_score = str(round(test_score * 100, 2))
    
    print(f"The top model predicts with {test_score}% accuracy on test data")