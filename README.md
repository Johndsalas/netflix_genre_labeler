# netflix_genre_labeler

This project is very new. This is a general road map as to what the plan is. A formal writeup for this readme will come once the project is complete. For now I am attempting to use the descriptions of netflix shows to predict the genre of the show.

Acquire data from genre and text description data from Kaggle 

Prepare data by making text description machine readable using bag of words

Drop columns genres that are in less than 100 observations 

Split data into train, validate, and test

Explore train looking for words that are unique to each genre

Create a machine learning model using tf-idf and evaluate on train and validate

Model will predict whether a movie is in a given genre because movies can be in more than one genre it is necessary to predict each genre one at a time

Updata models predictions by overriding movies that have genre unique words to the appropriate genre

Compare accuracy of updated results on train and validate

draw conclusions

Final writeup for notebook

Write README 
