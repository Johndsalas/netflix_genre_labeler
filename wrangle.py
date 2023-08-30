''' Code for acquiering and preparing data '''


import unicodedata
import pandas as pd
import regex as re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

################################################################ main acquire and prep function ##########################################################

def get_show_data():

    '''Acquiers and preps Netflix data'''
    # get dataframe of descriptions and genres

    df = pd.read_csv('titles.csv')

    df = df[['description', 'genres']]

    # drop rows where genre is empty
    df = df[(df.genres != '[]')]

    # convert strings in descriptions to lists
    df = str_to_list(df)

    # for each unique genre add a column genre_name displaying if the show is in that genre
    df = get_genre_columns(df)

    # clean description column
    df = prep_description(df)

    return df

############################################################## Helper prep functions for each column ###################################################

def get_genre_columns(df):
    '''for each unique genre add a column genre_name displaying if the show is in that genre'''

    # get set of unique genres
    gens = df[['genres']].explode('genres')

    gen_set = set(gens.genres.to_list())

    # for each unique genre add a column genre_name displaying if the show is in that genre
    for gen in gen_set:

        df[f'{gen}'] = df['genres'].apply(lambda gen_list: gen in gen_list)

    return df 


def prep_description(df):
    ''' Prepare film description text for exploration'''

    # convert strings to lowercase
    df['description'] = df['description'].apply(lambda value: str(value).lower())

    # remove special characters from description text
    df['description'] = df['description'].apply(lambda value: re.sub(r'[^\w\s]|[\d]', '', value))

    # remove non-ascii characters from description text 
    df['description'] = df['description'].apply(lambda value: unicodedata.normalize('NFKD', value)
                                                                         .encode('ascii', 'ignore')
                                                                         .decode('utf-8', 'ignore'))
    # tokenizes text in description
    df = get_disc_tokens(df)

    # lemmatize the text in description
    df['description'] = df['description'].apply(lemmatizer)

    # remove stopwords and words with less than three letters from text in description
    # return a list of words in the text

    df['description'] = df['description'].apply(remove_stopwords)

    return df

############################################################## Helper functions ############################################################################

def get_disc_tokens(df):
    '''Tokenize text in descriptions column of a pandas data frame'''

    tokenizer = ToktokTokenizer()

    # tokenize text in description
    df['description'] = df['description'].apply(lambda value: tokenizer.tokenize(value, return_str=True))

    return df


def lemmatizer(value):
    '''Takes in a value from a pandas column and returns the value lemmatized'''

    # create lemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()

    # get list of lemmatized words in value
    value_lemmas = [wnl.lemmatize(word) for word in value.split()]

    # turn list or words back into a string and return value
    return ' '.join(value_lemmas)


def remove_stopwords(value):
    ''' remove stopwords from text'''

    # get list of english language stopwords list from nlt
    stpwrd = nltk.corpus.stopwords.words('english')
    stpwrd.extend(['ing'])


    # split words in pandas value into a list and remove words from the list that are in stopwords or less than 3 letters
    value_words = value.split()
    filtered_list = [word for word in value_words if (word not in stpwrd) and (len(word) >= 3)]

    # convert list back into string and return value
    return ' '.join(filtered_list)


def str_to_list(df):
    ''' Remove punctuation from genres in genres column
        convert strings to lists'''

    puncs = ['[',
             ']',
             "'",
             ' ']

    for punc in puncs:

        df['genres'] = df['genres'].str.replace(punc,'')

    df['genres'] = df['genres'].str.split(',')
    
    return df

######################################################################### Data Splitting Function #####################################################################

def split_my_data(df):
    '''Splits full dataframe into train, validate, and test dataframes'''

    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    train, validate =  train_test_split(train_validate, test_size=.3, random_state=123)

    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, validate, test