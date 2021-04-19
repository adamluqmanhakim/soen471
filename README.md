# Soen 471 Project

# Abstract
Movie curating has been challenged by algorithms that are able to predict a viewer's next suggested movie. In the age of digital movie streaming, algorithms have been studied and built in order to enhance the experience of viewers and provide them with a list of movies that they might enjoy viewing next. While some best streaming platforms such as the Criterion Channel and MUBI are based on curating - the act of selecting films manually - other platforms such as Netflix and Amazon prime make use of algorithms in order to automatically curate and suggest content. The goal of our project is to build a recommender system using content filtering and collaborative filtering techniques in order to compare them and understand their advantages and disadvantages.

# Introduction

### Context
Streaming platforms are looking to improve their algorithms that suggest content based on a user's metadata. Movies and TV shows are recommended based on what content a user has rated and viewed. 

### Objective

The goal of this project is to study and compare two different movie recommendation system techniques based on the data analysis algorithms studied in the course. We want to analyze and compare results in order to understand why some recommendation systems are preferred than others, and in which context they might be more useful. In applying these techniques, we will predict movie suggestions for users and analyze their accuracy and how they could be improved. We want to understand and document the limitations of these algorithms, and how they could be improved.  

### Problem
The challenge in streaming services is to suggest movies to a user based on their interests, ratings, and metadata in the most accurate possible way. Some algorithms are limited when used seperately. Streaming services most likely use a hybrid system or a customized system with different algorithms to predict movie suggestions. The difficulty in predicting suggested movies lies within the fact that algorithms such as content based filtering have their own limitations. The Netflix Prize, as discussed in class, was a competition to come up with the best collaborative filtering algorithm to predict user ratings for films only based on anonymous users' previous ratings. There is a race between companies for best accuracy in algorithms. 

### Related work
Recommender systems for movies have been studied and implemented in the past. We are not implementing anything new. However, as avid cinephiles and students, we want to understand and learn how these recommender systems work and what their limitations are. A popular notebook on Kaggle titled ["Movie Recommender Systems"](https://www.kaggle.com/rounakbanik/movie-recommender-systems) by user _Rounak Banik_ explores different implementations of algorithms. This notebook shows various implemented algorithms, we will cite work from this notebook in case we deem it relevant. 

# Materials and Methods

### Dataset

 The dataset that will be used is the following: [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata.csv). This dataset contains metadata on over 45,000 movies and 26 milion ratings over 270,000 users from the movie recommendation service [**MovieLens**](https://movielens.org/). The original and complete [MovieLens 20M Dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset) contains over 20 million movie ratings since 1995. However, due to computer limitations and for the purpose of this project, we decided to go with [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata.csv). The dataset consists of the following .csv files:
 | .csv file       | Content         |
| -------------|:-------------:| 
| movies_metadata | Features including posters, backdrop, budget, revenue, release date, languages, production countries and companies |
| keywords        | Movie plot keywords in JSON |  
| credits         | cast and crew information in JSON |   
| links           | IMDB and The Movie Database (TMDB) IDs for all movies from the full MovieLens 20M Dataset |  
| links_small     | IMDB and The Movie Database (TMDB) IDs for 9,000 movies (small subset) |  
| ratings_small   | 100,000 ratings from 700 users on 9,000 movies |  

We are mostly interested in the movies_metadata and ratings_small files as we will perform analysis on ratings and content.

### Techniques
We will use two different approaches explored in class to recommender systems. The first is **content-based** filtering which recommends an item based on the comparison of the content of an item and a user's rating. This system makes use of metadata such as genre, cast of actors, and director. The idea is that if a user likes a specific item, then the user will also like a similar item. A director may have multiple movies, for example. The recommendation can be enhanced by combining different features such as movies from a specific director which contain a similar cast or actors from another liked movie. Based on the lecture, we will create an item profile and a user profile. The item profile will be a set of a vector that based on the rating and the genre of the movies. The user profile would be based on the average of rated item profiles by using this Prediction heuristic formula : u(x,i) = cos(x,i) = (x.i) /(||x||.||i||), where x : user profile and i is item profile. 

The second technique is **collaborative** filtering. As opposed to content-based, collaborative filtering _does not require_ information about the content of items. The idea is based on the assumption that similar users share similar interests. We will consider the Pearson Correlation Coefficient formula. Based on this formula, the value returned is between -1 and 1. The closer to 1, the higher the correlation between two users. Using this technique, we will process the data from the similiar set movie users and the highest correlation movie will be recommended. Time permitting, Pyspark's Alternating Least Squares (ALS) algorithm will also be considered to compare results. 


### Algorithms
For content based filtering, we have decided to build a model that computes the cosine similarity between movies based on the plot description of each movie. Similar movies with similar content (plot descriptions) result with high cosine similarity scores which are then recommended to the user. Our procedure is to read the **links_small.csv** file which contains the movie title, movie genre, and plot description. Then we calculate the Term Frequency-Inverse Document frequency (TF-IDF) matrix which is used to compute the cosine similarity using sklearn. The TD-IDF measures the relevance of a word to a document in a collection of documents. In this TF-IDF, we use the description column of the *links_small* dataframe. 

![image](https://user-images.githubusercontent.com/6520150/115175215-97f7ba80-a098-11eb-8880-05358fefc88a.png)
![image](https://user-images.githubusercontent.com/6520150/115175226-9cbc6e80-a098-11eb-9403-329a0c494d87.png)


Getting the movie recommendations for a specific movie title is simply calculating all the cosine similarity scores of the movies against this specific movie title. We then sort 


For collaborative filtering, we use the Alternating Least Squares (ALS) offered by Pyspark. 
Matrix factorisation (ALS) tries to approximate the ratings matrix R as the product of two lower-rank matrices, X and Y, i.e. X*Yt = R. These approximations are also referred to as ‘factor’ matrices. This method is iterative in nature. One of the factor matrices is kept constant throughout each iteration, and the other is solved using least squares. Cold strategy is set to ‘drop’ as it is common to encounter users or items in the evaluation set which are not in the training set, when using random splits with Spark’s CrossValidator. Spark will by default assign NaN predictions. This is avoided with the ‘drop’ cold strategy. We make use of hyperparameter tuning using ParamGridBuilder. According to Spark’s documentation, it is a builder for a param grid used in grid search-based model selection. We chose 4 parameters for each grid. For the Rank grid, we have chosen the following values: 10, 50 ,100 and 150. We randomly chose these values and tested them manually. For the regParam grid, we chose the values 0.01, 0.05, 0.1 and 0.15. Therefore, our rank is 16 as we have 16 features to use (number of latent factors). 

![image](https://user-images.githubusercontent.com/6520150/115175298-c07fb480-a098-11eb-8500-6fc79d833505.png)
As the evaluator, we use the RMSE evaluator.
![image](https://user-images.githubusercontent.com/6520150/115175316-c6759580-a098-11eb-99fa-64246277d03e.png)

We feed the param_grid and the evaluator into the cross validator for the ALS model with a chosen value of 5 for the folds. In the results we talk about the best model parameters out of the 16 parameters that were inputted in the cross validator. 

### Results
For content based filtering, we decided to get the top ten recommendations for two distinct movie titles: “Star Wars” and “Shrek”. The results are the following titles.
Star Wars gives a recommendation of:
- The Empire Strikes Back
- Return of the Jedi
- Star Wars: The Force Awakens
- Shrek the Third
- Star Wars: Episode III - Revenge of the Sith
- Where Eagles Dare
- Shrek Forever After
- On Her Majesty's Secret Service
- The Ice Pirates
- Princess Caraboo

Shrek gives a recommendation of:
- Shrek 2
- Shrek Forever After
- Shrek the Third
- Silk Stockings
- Dragon Hunters
- White Men Can't Jump
- Shanghai Noon
- Cinderella
- Cinderella
- Into the Woods

![image](https://user-images.githubusercontent.com/6520150/115176419-0fc6e480-a09b-11eb-84c0-fa0e998d4370.png)

As for collaborative filtering, we decided to get the movie recommendations for the 3rd user and the 20th user. We also display the results for the top 10 ratings of both users in order to compare them with our recommendations.
For the 3rd user, we found the following recommendations (first figure) and the movie ratings (second figure)

![image](https://user-images.githubusercontent.com/6520150/115176637-7c41e380-a09b-11eb-84c1-d8858531c17b.png)

Notice the "userId = 3" and limit(10). 

For the 20th user, we found the following recommendations and movie ratings:
![image](https://user-images.githubusercontent.com/6520150/115176704-a398b080-a09b-11eb-9dcf-88ef3b751c4d.png)

We color-coded underlined the movie genres that match in both recommendaations and movie ratings.

### Discussion

In our results for content filtering, we have proved that the results do not change after multiple runs. The movie recommendations remain the same as the calculations are based on the content of each movies (which do not change). The similarity scores do not change after each run as the cosine similarity is simply a mathematical formula that calculates the score based on the content. Therefore, we do not have much to interpret for the results of content filtering, other than that it is a simple way of recommend movies based on the content of the movie. The content of course was chosen to be plot movie description. In order to improve this algorithm, you would need to change the content that it is evaluating. For example, instead of doing it based on plot description, you would analyze the cast/crew of a movie and/or the director of a movie. You can also combine content. 

For collaborative filtering, we measured it using the RMSE evaluator. Since the 100k ratings dataset would take several hours to complete, we decided to cut the ratings to 20,000 and train the model with the smaller version of the dataset. This resulted in a poor RMSE score. For further experimentation, we cut down the ratings to 2000 ratings, and we trained the model again and ran the evaluator again. We found an even worse RMSE score. This proves that a better RMSE score is based on whether there are sufficient ratings to train the model. These are the RMSE results we found using 20,000 ratings:
![image](https://user-images.githubusercontent.com/6520150/115177299-ec049e00-a09c-11eb-8d37-bc5ff38313bf.png)

We still found that 1.037 is a bad RMSE score as it is more than 1.

After experimenting with 2000 ratings, we had an RMSE score of 1.543 which is worse than our initial one with 20,0000. 
![image](https://user-images.githubusercontent.com/6520150/115177409-2a9a5880-a09d-11eb-9ea1-0d3fd29e5c54.png)

As shown in this figure, we found that the 20th user rated 5 movies with these specific genres: Comedy|Romance, Animation|Children and Action|Adventure. 
In our recommendation results, we found three movies with these 3 exact genres. 
![image](https://user-images.githubusercontent.com/6520150/115176704-a398b080-a09b-11eb-9dcf-88ef3b751c4d.png)

Despite the bad RMSE score, our collaborative filtering recommendation system is still able to recommend movies that have exact genres that interest the user - these movies are based on the users' previous history of movie ratings.

To compare content filtering with collaborative filtering, these two techniques are different because the latter is based on the users's explicit ratings (history). This can vary differently between users. We also found that both the 3rd and 20th user rated movies highly with a score of 5.0 for all movies - the highest. Rating all movies the same rating affects the score as there is not a lot of variance between ratings. This is true as well for users with no ratings or new users. The recommendations for users with no ratings will not be accurate as there are no explicit ratings to be based on. 

In the future, we would like to train the collaborative filtering model with 100k ratings in order to see if the RMSE rating has improved. We would expect a RMSE rating lower than 1. We would also like to explore how hybrid recommender systems work as they combine different techniques.
