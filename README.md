# Sarcasm-Sniffer
Sarcasm Sniffer is situated at the intersection of natural language processing and sentiment analysis to address challenge of sarcasm detection within textual data, accurately identifying sarcasm presents a nuanced task that conventional sentiment analysis tools often struggle to accomplish.

**Twitter Sarcasm Detection Using Machine Learning & Deep Learning Models**
This repository is focusing on the application and evaluation of machine learning and deep learning models for sarcasm detection within textual data enriched with emojis and hashtags.

## Goal
- Build a good dataset which has sarcastic documents
- Annotate the data as 0/1 (sarcastic/not sarcastic)
- Identify different patterns in the text that reveal sarcasm
- Build a model that classifies a new unseen text or tweet as Sarcastic or not sarcastic
- Evaluate the model built using f1 score

## Models
- Machine Learning - Logistic Regression
- Deep Nueral Network - Long Short-Term Memory (LSTM)

## Datasets
- Labelled Dataset
The labelled dataset employed in this project was acquired from the GitHub repository crafted by Muhammad Adyl, accessible at Sarcasm Detection Dataset.
**Train.csv** & **Test.csv**
- Realtime Dataset
I employed Scraper API as a robust tool for extracting a pertinent dataset of tweets.
The utilization of Scraper API facilitated responsible and respectful data gathering, ensuring a seamless and uninterrupted process.
**Sarcastic_tweets.csv** & **Non_sarcastic_tweets.csv**

## Evaluation - Accuracy
- Logistic Regression - 0.893
- LSTM - 0.754
