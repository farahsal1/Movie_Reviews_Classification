#  Sentiment Analysis of IMDb Movie Reviews:  Training a binary classification model to predict positive and negative reviews


In this project, I have performed sentiment analysis on a labelled IMBb movie reviews dataset and trained a binary classification model to identify whether the movie review is positive or negative. 

## Data Source 

 For my model, I have used the IMDb Movie Review dataset, obtained from Kaggle. It was originally developed by Mass, Daly et al (2011) original paper: [Learning Word Vectors for Sentiment Analysis]('https://ai.stanford.edu/~amaas/data/sentiment/'). The dataset was created by scraping reviews data from IMDb website. It contains 50,000 reviews, with no more than 30 reviews per movie. The dataset has an even number of positive and negative reviews i.e. 25,000 each. Only highly polarized reviews were considered and neutral reviews are not part of this dataset. 


 ## Methodology

I used a TF-IDF vectorizer to quantify the relative importance of each word occurring in the reviews with respect to all words occurring in all reviews, and then uses these scores to train a binary classification model. However, before applying the TF-IDF vectorizer, I cleaned the data to:
1. Remove punctuations and special characters
2. Remove stop words (common words that don't carry a significant meaning). For this, I used Natural Language ToolKit (NLTK) stop words corpus in Python
3. Lemmatize all words using spaCy Lemmatizer.

I then created Feature vectors from the final clean version of this dataset. 

For the final classification model, I applied and evaluated three models: 
1. Multinomial Naive-Bayes
2. Logistic Regression
3. Neural Networks (Multilayer Perceptron Classifier)
4. Random Forest
5. K-Nearest Neighbors

Each of the models selected have strengths that make them suitable for training feature vectors for text classification. For example, while naive bayes models are efficient and simple to implement on TF-IDF vectorization, they assume independence between the features, whereas the words in the corpus are likely not independent. Logistic regression was selected as it does not assume independence between different features.


