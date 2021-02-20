# Soen 471 Project

# Abstract
Movie curating has been challenged by algorithms that are able to predict a viewer's next suggested movie. In the age of digital movie streaming, algorithms have been studied and built in order to enhance the experience of viewers and provide them with a list of movies that they might enjoy viewing next. While some best streaming platforms such as the Criterion Channel and MUBI are based on curating - the act of selecting films manually - other platforms such as Netflix and Amazon prime make use of algorithms in order to automatically curate and suggest content. 

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


