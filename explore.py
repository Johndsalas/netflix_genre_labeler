''' Contains functions for exploring Netflix shows data'''

import pandas as pd
import matplotlib.pyplot as plt


def get_gens(df):
    '''get set of unique genres'''

    return set(df['genres'].explode('genres').to_list())

def get_description_set_of_words(df):
    '''get set of unique words in descriptions'''

    return set(df['description'].apply(lambda value : value.split(' '))
                                .explode('description')
                                .to_list())

def get_count_of_words(df, word):
    ''' take in a dataframe and a word 
        return total number of times that the word appears 
        in values in the description column of that dataframe'''

    df['word_count'] = df['description'].apply(lambda value: value.count(word))

    word_count = df.word_count.sum()

    df = df.drop(columns=['word_count'])

    return word_count

def get_count_of_docs(df, word):
    ''' take in a dataframe and a word 
        return total number of in values 
        in the description column of that dataframe
        containing that value'''
    
    df['word_count'] = df['description'].apply(lambda value: value.count(word))

    df['doc_count'] = df['word_count'] > 0

    doc_count = df.doc_count.sum()

    df = df.drop(columns=['word_count', 'doc_count'])

    return doc_count


def get_word_freq(train):
    ''' takes in training data
        returns a dictionary of each word in the training set
        and its relative frequency of apearances in comedy and non-comedy descriptions'''

    word_freq = {}

    # get set of unique words in training data
    train_set_of_words = get_description_set_of_words(train)

    com_train = train[train.comedy == True]

    non_train = train[train.comedy == False]

    for word in train_set_of_words:

        # get number of times a word appears in description column values of comedies
        com_words = get_count_of_words(com_train, word)

        # get number of times a word appears in description column values
        non_words = get_count_of_words(non_train, word)

        # subtract number of comedy words from non comedy words
        word_total = com_words - non_words

        # add word and total to dictionary
        word_freq[f'{word}'] = word_total

    # convert to list and order by number
    word_freq = sorted(word_freq.items(), key = lambda x : x[1])

    return word_freq


def get_doc_freq(train):
    ''' takes in training data
        returns a dictionary of each word in the training set
        and its relative frequency of apearances in comedy and non-comedy descriptions'''

    doc_freq = {}

    # get set of unique words in training data
    train_set_of_words = get_description_set_of_words(train)

    com_train = train[train.comedy == True]

    non_train = train[train.comedy == False]

    for word in train_set_of_words:

        # get number of comedy documents in description containing the word
        com_docs = get_count_of_docs(com_train, word)

        # get number of non documents in description containing the word
        non_docs = get_count_of_docs(non_train, word)

        # get difference in documents
        doc_total = com_docs - non_docs

        # add word and total to dictionary
        doc_freq[f'{word}'] = doc_total

    # convert to list and order by number
    doc_freq = sorted(doc_freq.items(), key = lambda x : x[1])
    
    return doc_freq


def get_counts(lst):
    ''' makes a list from the second item of paired items from a list '''

    counts =[]

    for item in lst:

        counts.append(item[1])

    return counts

############################################################## Visualizations #############################################################################

def get_shows_per_gen(df, gen_set):
    '''Takes in a dataframe and a set of genres
       Returns a dataframe containing each genre and number of rows in input dataframe containing that genre'''
    
    gens = []
    nums = []

    # for each genre add its name and the number of rows in the dataframe containing that genre to seperate lists
    for gen in gen_set:
        
        gens.append(gen)
        
        nums.append(len(df[df[f'{gen}'] == True]))
        
    # combine the two rows into a dictionary and create a dataframe
    df = pd.DataFrame(dict(
                            gens = gens,
                            nums = nums))

    # sort by number values
    df = df.sort_values('nums')

    return df


def get_unique_words_per_gen(df, gen_set):
    '''Takes in a data frame and a list of genres 
       Returns a dataframs containing the number of differint unique words in each genre'''
    gens = []
    nums = []

    # for each genre add its name and the number of unique words belonging to that genre to seperate lists    
    for gen in gen_set:

        true_set = get_description_set_of_words(df[df[f'{gen}'] == True])
        false_set = get_description_set_of_words(df[df[f'{gen}'] == False])

        diff_set = true_set - false_set
        
        gens.append(gen)
        
        nums.append(len(diff_set))
    
    # create a dataframe from the lists and sort the dataframe by the number column
    df = pd.DataFrame(dict(gens = gens,
                           nums = nums))

    df = df.sort_values('nums')

    return df


def get_unique_word_appearance_per_genre(df, gen_set):
     
    gens = []
    nums = []

    for gen in gen_set:

        true_set = get_description_set_of_words(df[df[f'{gen}'] == True])
        false_set = get_description_set_of_words(df[df[f'{gen}'] == False])

        unique_to_genre_set = true_set - false_set
        
        df['num_uniques'] = df['description'].apply(lambda value: get_num_uniques(value, unique_to_genre_set))

        unique_word_appearances = df.num_uniques.sum()

        rows_in_genre = len(df[df[f'{gen}'] == True])

        gens.append(gen)

        nums.append(unique_word_appearances / rows_in_genre)

    # create a dataframe from the lists and sort the dataframe by the number column
    df = pd.DataFrame(dict(gens = gens,
                           nums = nums))

    df = df.sort_values('nums')

    return df


def get_num_uniques(value, u_set):

    count = 0

    value_list = value.split(' ')

    for value in value_list:

        if value in u_set:

            count += 1

    return count









def get_bar(df, title ):
    '''Display barplot for df containing number of shows in each genre'''

    plt.figure(figsize=(20,10))
    plt.bar('gens', 'nums', data=df, color='lightblue')
    plt.title(f"{title}")

    plt.tight_layout()
    plt.show()







def omni_pie(panda_series,title = "Super Awsome Title I'll Think of Latter"):

    labels = set(value for value in panda_series)

    values = [len(panda_series[panda_series == label]) for label in labels]

    # generate and show chart
    plt.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4', '#cf98eb', '#77d198'])
    plt.title(f'{title}')
    plt.show()


def get_hist_word(li):
    
    plt.figure(figsize=(20, 5))
    plt.xlim(-30,30)
    
    plt.title("Relative Word Count Normalizes around Zero Suggesting there is a Lot of Noise to be Removed")
    plt.xlabel("Relative Word frequecy")
    plt.ylabel("Number of Unique Words")
    plt.hist(li, bins = 100)

    plt.show()


def get_hist_doc(li):
    
    plt.figure(figsize=(20, 5))
    plt.xlim(-30,30)
    
    plt.title("Relative Document Frequency Normalizes around Zero Suggesting there is a Lot of Noise to be Removed")
    plt.xlabel("Relative Document frequecy")
    plt.ylabel("Number of Unique Words")
    plt.hist(li, bins = 100)

    plt.show()


def get_doc_ext(data, title):

    # Extract labels and values from the dictionaries
    labels = list(data.keys())
    values = list(data.values())

    # Plot the chart
    plt.barh(labels, values)
    plt.xlabel('Value')
    plt.ylabel('Label')
    plt.title(f'{title}')

    plt.show()





    # def unique_words_frequency(df, gen_set):


#     # get df of each genre and number of differint unique words
#     num_unique_df = unique_words_per_gen(df, gen_set)


#     for genre in gen_set:

#         num_unique = num_unique_df[f'{genre}']

#         total_docs = df[f'{genre}' == True]

#         num_





#     gens = []
#     nums = []

#     for gen in gen_set:

#         true_set = get_all_unique_words( df.description[df[f'{gen}'] == True])
#         false_set = get_all_unique_words( df.description[df[f'{gen}'] == False])

#         diff_set = true_set - false_set
        
#         gens.append(gen)
        
#         nums.append(len(diff_set))

#         t_sow = set(t_bow.split(' '))
#         f_sow = set(f_bow.split(' '))

#         d_sow = t_sow - f_sow
        
#         gens.append(gen)
        
#         nums.append(len(d_sow)/len(df[df[f'{gen}'] == True]))
        
        
#     sort = pd.DataFrame(dict(
#                             gens = gens,
#                             nums = nums))

#     sort = sort.sort_values('nums')

#     plt.figure(figsize=(20,10))


#     plt.bar('gens', 'nums', data=sort, color='lightblue')


#     plt.title("On Average Each Show Description Contains Between 2-4 Unique Words")
#     plt.tight_layout()
#     plt.show()