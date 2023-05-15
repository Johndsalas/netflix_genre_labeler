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