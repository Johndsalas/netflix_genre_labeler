''' Contains functions for exploring Netflix shows data'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def get_gens(df):

    # get set of unique genres
    gens = df[['genres']].explode('genres')

    gen_set = set(gens.genres.to_list())

    gen_set.remove('western')

    return gen_set

def get_majoriety_counts(train):
    
    freq_doc = {}
    freq_count = {}

    list_freq_doc = []
    list_freq_count = []
    
    train_set_of_words = set(train['description'].apply(lambda value : value.split(' '))
                                                 .explode('description')
                                                 .to_list())
    
    comedy_train = train[train.comedy == True]

    non_comedy_train = train[train.comedy == False]
    
    for word in train_set_of_words:

        # get count of comedy films that have word in them
        comedy_train['com_count'] = comedy_train['description'].apply(lambda words: words.count(word))
        
        comedy_train['com_doc'] = comedy_train['com_count'] > 0
        
        c_w_count = comedy_train['com_count'].sum()
        
        c_d_count = comedy_train['com_doc'].sum()
        
        # get count of non-comedy films that have word in them
        non_comedy_train['non_count'] = non_comedy_train['description'].apply(lambda words: words.count(word))
        
        non_comedy_train['non_doc'] = non_comedy_train['non_count'] > 0
        
        n_w_count = non_comedy_train['non_count'].sum()
        
        n_d_count = non_comedy_train['non_doc'].sum()
        
        # subtract number of non-comedy films from number of comedy films
        
        w_total = c_w_count - n_w_count
        
        d_total = c_d_count - n_d_count
        
        # append difference
        
        freq_doc[f'{word}'] = d_total
        
        list_freq_doc.append(d_total)
        
        freq_count[f'{word}'] = w_total
        
        list_freq_count.append(w_total)
        
    return freq_doc, freq_count, list_freq_doc, list_freq_count












####################################### Visualizations ###########################################################

def shows_per_gen(df, gen_set):

    # get pairs of gens and number of rows
    gens = []
    nums = []

    for gen in gen_set:
        
        gens.append(gen)
        
        nums.append(len(df[df[f'{gen}'] == True]))
        
    sort = pd.DataFrame(dict(
                            gens = gens,
                            nums = nums))

    sort = sort.sort_values('nums')

    plt.figure(figsize=(18,10))


    plt.bar('gens', 'nums', data=sort, color='lightblue')


    plt.title("Number of Shows Representing Each Genre")
    plt.tight_layout()


    plt.show()


def unique_words_per_gen(df, gen_set):

    gens = []
    nums = []

    for gen in gen_set:

        t_bow = ''
        f_bow = ''

        for value in df.description[df[f'{gen}'] == True]:

            t_bow += value

            t_bow += ' '

        for value in df.description[df[f'{gen}'] == False]:

            f_bow += value

            f_bow += ' '

        t_sow = set(t_bow.split(' '))
        f_sow = set(f_bow.split(' '))

        d_sow = t_sow - f_sow
        
        gens.append(gen)
        
        nums.append(len(d_sow))
        
    sort = pd.DataFrame(dict(
                            gens = gens,
                            nums = nums))

    sort = sort.sort_values('nums')

    plt.figure(figsize=(20,10))


    plt.bar('gens', 'nums', data=sort, color='lightblue')

    plt.title("Number of Unique Words in Each Genre")
    plt.tight_layout()
    plt.show()

def unique_words_frequency(df, gen_set):

    gens = []
    nums = []

    for gen in gen_set:

        t_bow = ''
        f_bow = ''

        for value in df.description[df[f'{gen}'] == True]:

            t_bow += value

            t_bow += ' '

        for value in df.description[df[f'{gen}'] == False]:

            f_bow += value

            f_bow += ' '

        t_sow = set(t_bow.split(' '))
        f_sow = set(f_bow.split(' '))

        d_sow = t_sow - f_sow
        
        gens.append(gen)
        
        nums.append(len(d_sow)/len(df[df[f'{gen}'] == True]))
        
        
    sort = pd.DataFrame(dict(
                            gens = gens,
                            nums = nums))

    sort = sort.sort_values('nums')

    plt.figure(figsize=(20,10))


    plt.bar('gens', 'nums', data=sort, color='lightblue')


    plt.title("On Average Each Show Description Contains Between 2-4 Unique Words")
    plt.tight_layout()
    plt.show()


def omni_pie(panda_series,title = "Super Awsome Title I'll Think of Latter"):

    labels = set(value for value in panda_series)

    values = [len(panda_series[panda_series == label]) for label in labels]

    # generate and show chart
    plt.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4', '#cf98eb', '#77d198'])
    plt.title(f'{title}')
    plt.show()


def get_hist(li):
    
    plt.figure(figsize=(20, 5))

    plt.xlim(-50,50)

    plt.hist(li, bins = 500)

    plt.show()