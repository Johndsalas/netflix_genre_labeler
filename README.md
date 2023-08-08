# netflix_genre_labeler

# Description:

Streaming services such as Netflix and Hulu are regularly in need of updating the shows they offer on their platform. A major way those shows are recommended to customers is by their genre. Customers will often search for “something to watch” by scrolling through films of a particular genre until they find something that catches their eye. This means that properly labeling a movie’s genre is an important part of meeting customer’s expectations. This combined with a constant need to update which shows appear on a given platform made me curious as to whether I could develop a machine learning model that was capable of predicting a film's genre based on the description of the film. If successful this algorithm would provide genre labels for films that could be fed into more complex recommendation algorithms without the need to rely on time consuming human labor to provide those labels.

# Initial Thoughts
Upon viewing the data I encountered a problem. Films were not classified as one genre or another exclusively. In fact most of the films were classified as a list of genres. While this makes sense, a film can easily fall under more than one genre. It means that I will not be able to use genre as my target variable.

The solution I came up with was to build a model that could predict whether a film was or was not a given genre. If built generally enough, the model could be used to predict each genre in turn adding a tag for that genre to films with a positive prediction. Perhaps models would need to be tuned for each genre. The goal for this project is the first step in this process.

# Goal:

Choose a genre as a test case for model development
Build a Classification model to predict whether a film is or is not the test case genre
Evaluate final model to determine if accuracy warrants continuing this process with other genres 

# Data Dictionary
| Feature | Definition |
| :---------- | :------------- |
| genre   | List of genres film is included in |
| comedy | Identifies whether the film is in the comedy genre, engineered from genre feature |
| description | Description of film |

# Plan:

## Wrangle

* Acquire data from Kaggle pairing Netflix show description and genres
* Prepare description data for analysis
* Create ‘genre name’ column labeling films as being in that genre or not being in that genre (True/False)
* Split data into train, validate and test

## Explore

* Choose a genre to use as a test case

* Test Genre should:
    * Have a large representation in the data, for significance
    * Have a large number of genre unique words, for machine identifiability
    * Words that are only used in descriptions relating to that genre
    * Be Intuitively distinct from other genres, for greatest chance of success

* Examine test genre

* How much of the data belongs to the test genre?
* What is the relative frequency of words in the training data by count?
    * Number of times a given word appears in test genre films minus the number of times that word appears in non-test genre films
* What is the relative frequency of words in the training data by document?
    * Number of test genre films documents a given word appears in minus the number of non-test genre films the word appears in
* Which words appear at the most extreme positive and negative frequency?
* Does there appear to be a significant difference in document frequency and Count frequency?

## Model

* Develop best possible model, as determined by overall accuracy
    * Vectorize data using 
        * Count
        * TFIDF
* Model data using
    * Decision Tree
    * Random Forest
    * K Neighbors
    * Logistic Regression
* Attempt to remove noise by dropping word columns with low relative frequency
    * By word count
    * By document count
* Attempt to improve accuracy of model using unique override method
    * Evaluate 4 conditions
        * Override only if description contains a genre unique word
        * Override only if description contains a non-genre unique word
        * Override if description contains a either a genre unique word or non-genre unique word
        * Override only if description contains a unique word from one category and no unique words from the other


# Steps to Reproduce

Clone this repo
Download the data from Kaggle as a .csv (add link latter)
Put the data in the file containing the cloned repo
Run notebook


# Takeaways and Conclusions

# Recommendations
