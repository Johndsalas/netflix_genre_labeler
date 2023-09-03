# Netflix Genre Labeler

# Description:

Streaming services such as Netflix and Hulu are regularly in need of updating the shows they offer on their platform. A major way those shows are recommended to customers is by their genre. Customers will often search for something to watch by scrolling through films of a particular genre until they find something that catches their eye. This means that properly labeling a movie’s genre is an important part of meeting customer’s expectations. This combined with a constant need to update which shows appear on a given platform made me curious as to whether I could develop a machine learning model that was capable of predicting a film's genre based on the description of the film. If successful this algorithm would provide genre labels for films that could be fed into more complex recommendation algorithms without the need to rely on time consuming human labor to provide those labels.

# Initial Thoughts
Upon viewing the data I encountered a problem. Films were not classified as one genre or another exclusively. In fact most of the films were classified as a list of genres. While this makes sense, a film can easily fall under more than one genre. It means that I will not be able to use genre as my target variable.

The solution I came up with was to build a model that could predict whether a film is or is not in a given genre. If an accurate model could be developed for one genre it is likely that the model building method could be generalized to develop a predictive model for each genre. Each model could then be run on the data to create a comprehinsive list of all of the genres that a given film is a part of

# Goal:

* Choose a genre as a test case for model development
* Build a Classification model to predict whether a film is or is not the test case genre
* Evaluate the model to determine if accuracy warrants continuing this process with other genres 

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

* Explore genres and choose a genre to use as a test case

* Test Genre should:
    * Have a large representation in the data
       * for significance
    * Have a large number of genre unique words, words that are only used in descriptions relating to that genre
       * for machine identifiability
    * Be intuitively distinct from other genres
       * for greatest chance of success

* Examine test genre
    * How much of the data belongs to the test genre?
    * What does the relative occurrence of words in the test genre or non-test genre films tell us?

## Pre-Modeling

* Develop two methods of predicting the test genre using unique words, words that only appear in genre or non-genre film descriptions in the training data
    * By presnece of in-genre uniques and absence of non-genre uniques or vice versa
    * By majority count of in-genre uniques vs non-genre uniques
    * Evaluate using accuracy and override accuracy, accuracy of True/False predictions only
        * A method with high enough accuracy could be used to in place of a traditional machine learning model
        * A method with high enough override accuracy could be used in conjunction with a traditional machine learning model to improve its accuracy
    * Predictions will be evaluated using validate data
      
## Model

* Develop best possible model, as determined by overall accuracy
    * Vectorize data using 
        * Count
        * TF-IDF
    * Attempt to remove noise by dropping word columns with low relative frequency
        * By word count
        * By document count
    * Model data using
        * Decision Tree
        * Random Forest
        * K Neighbors
        * Logistic Regression
* Evaluate best model on test and determine if accuracy is high enough to move to the next step 

# Steps to Reproduce

* Clone this repo
* Download the data from Kaggle as a .csv (add link latter)
* Put the data in the file containing the cloned repo
* Run notebook

# Process Summary

* Comedy was chosen as the test case to build a predictive model using the following criteria 
    * Representation in the data
    * Number of unique words
    * High number of unique words appearing in each description
    * human intuitability
* The comedy genre was explored 
    * Data Split
        * Training data has 60% comedy and 40% non-comedy data 
            * This likely caused the relative frequency numbers to skew toward the negative
    * Relative Frequency Distribution
        * Overall trends in the data were the same using relative word frequency and relative document frequency
            * Majority of the data is between -5 and 5
                * Concluded using removal thresholds that are between 0-5 to remove noise would likely produce the most accurate models
            * There is a negative skew to the data likely due to the 60/40 imbalance in the data
    * Extreme Value Words
        * Extreme positive values are much closer to zero than their negative counterparts 
            * More evidence the data is skewed
        * Extreme positive values represent words that are intuitively indicative of comedy films
        * Extreme negative words represent words that are not intuitively indicative of non-comedy films 
            * This gives evidence that the skew is likely having an impact on the relative frequency data
        * High positive value words are still likely to be strong indicators of comedy because the skew works against high positive value words
* Methods were developed to predict comedy genre
    * Pre-Modeling Predictions
        * Predicting comedy using presence or number of unique words yielded disappointing results
            * Presence of Unique Words
                * Overall Accuracy 32%
                * Override Accuracy 68%
            * Number of Unique Words
                * Overall Accuracy 49%
                * Override Accuracy 67%
            * Overall accuracy for both attempts were less than baseline
            * Override accuracy for both attempts were above baseline but worse then the final model 
            * Methods are not viable for predicting comedy at this time
* Machine learning models were developed to predict comedy genre
    * Top Performing Model 
        * Uses count vectorized data
        * Removes words using relative word frequency
        * Uses a threshold of 0
        * Is a Logistic Regression model
        * Has an accuracy of 72.99% on test data beating baseline by ~13%
    * Count Vectorized data seems to outperform TF-IDF data by a small margin
    * Removing word features due to low word frequency seems to outperform removing word features do to low document frequency by a small margin

# Next Steps and Recommendations

* At this time the accuracy of the best performing model stands at 72.99% beating baseline by ~13%
* Enough promise has been shown for me to recommend moving forward with constructing a pipeline to create and evaluate models for predicting each genre individually 
* I also recommend putting more time into developing a more accurate model using the following guidelines 
    * Focus on Logistic Regression models
        * This model was consistently the highest performer on its default settings
        * It may be possible to increase its accuracy by adjusting its hyperparameters
    * Focus on count vectorized Data
        * This type of vectorization seems to result in slightly higher accuracy than TF-IDF
    * Focus on relative frequency by word count
        * Removing low relative frequency words seems to outperform removing words with low document frequency 
        * Normalize relative frequency values to avoid skewing the data  
    * Rethink non-modeling approach to predictions
        * Try making predictions by adding the relative frequency numbers of all of the words in the description
            * positive results would be predicted as comedy 
            * negative results would be predicted as non-comedy
              
