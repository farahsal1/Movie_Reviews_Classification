#  Sentiment Analysis of IMDb Movie Reviews:  Training a binary classification model to predict positive and negative reviews


In this project, I have performed sentiment analysis on a labelled IMBb movie reviews dataset and trained a binary classification model to identify whether the movie review is positive or negative. 

## Data Source 

 For my model, I have used the IMDb Movie Review dataset, obtained from Mass, Daly et al (2011) paper ‘Learning Word Vectors for Sentiment Analysis’. The dataset was created as a part of the original paper, by scraping reviews data from IMDb website. It contains 50,000 reviews, with no more than 30 reviews per movie. The dataset has an even number of positive and negative reviews i.e. 25,000 each. Only highly polarized reviews were considered and neutral reviews are not part of this dataset. 


 ## Methodology

The model uses a TF-IDF vectorizer to quantify the relative importance of each word occurring in the reviews with respect to all words occurring in all reviews, and then uses these scores to train a binary classification model. However, before applying the TF-IDF vectorizer, I cleaned the data to:
1. Remove punctuations and special characters
2. Remove stop words (common words that don't carry a significant meaning). For this, I used spaCy stop words corpus in Python
3. Lemmatized all words using Lemmatizer from Natural Language ToolKit (NLTK) package

I then created Feature vectors from the final clean version of this dataset. 

## Exploratory Data Analysis

