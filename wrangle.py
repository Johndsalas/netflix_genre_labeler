''' Code for acquiering and preparing data '''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

################# main acquire and prep function##################################

def get_show_data():

    '''Acquiers and preps Netflix data'''
    # get dataframe of descriptions and genres 
    df = pd.read_csv('titles.csv')

    df = df[['description', 'genres']]

    # drop rows with empty genres
    df = df[df.genres != '[]']

    # convert strings in genres to actual lists
    df = str_to_list(df)
    
    # for each unique genre add a column genre_name displaying if the show is in that genre
    df = get_genre_columns(df)
    
    # clean description column
    df = prep_description(df)
    
    return df

####### Helper prep functions for each column###################################################

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

    # remove special characters from description text
    df['description'] = df['description'].apply(lambda value: str(value).lower())
    
    df['description'] = df['description'].apply(lambda value: re.sub(r'[^\w\s]|[\d]', '', value))
    
    # remove non-ascii characters from description text 
    df['description'] = df['description'].apply(lambda value: unicodedata.normalize('NFKD', value)
                                                                         .encode('ascii', 'ignore')
                                                                         .decode('utf-8', 'ignore'))
    # tokenizes text in description
    df = get_disc_tokens(df)

    # lemmatize the text in description
    df['description'] = df['description'].apply(lambda value: lemmatizer(value))

    # remove stopwords from text in description and return a list of words in the text
    df['description'] = df['description'].apply(lambda value: remove_stopwords(value))

    return df

######Minor Helper functions############################################

def get_disc_tokens(df):
    
    tokenizer = nltk.tokenize.ToktokTokenizer()

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

    # get list english language stopwords list from nlt
    stopword_list = stopwords.words('english')
    
    # split words in pandas value into a list and remove words from the list that are in stopwords
    value_words = value.split()
    filtered_list = [word for word in value_words if word not in stopword_list]
    
    # convert list back into string and return value
    return ' '.join(filtered_list)


def str_to_list(df):
    
    puncs = ['[',']',"'",' ']

    for punc in puncs:

        df['genres'] = df['genres'].str.replace(punc,'')

    df['genres'] = df['genres'].str.split(',')
    
    return df