# Twitter Sentiment Analysis (CPE342 Mini Project)

## Overview
A machine learning system that classifies tweets as positive, negative, or neutral using natural language processing techniques. This project implements multiple classification models to analyze the emotional tone of social media text.

![Application Demo](./assets/application_demo_1.png)
![Application Demo](./assets/application_demo_2.png)

## Dataset
The dataset contains labeled tweets from [Kaggle](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) 

The data includes tweet text, sentiment labels, and metadata such as time of posting, user demographics, and geographic information.

## Features
- Text preprocessing pipeline with tokenization, stopword removal, and stemming
- Multiple classification models (Naive Bayes, Logistic Regression, Random Forest)
- Model comparison and evaluation metrics
- Interactive web application for real-time sentiment analysis

## Web Application
The included Streamlit application provides an intuitive interface for sentiment analysis:

- Real-time tweet sentiment prediction
- Optional Xquik JSON, JSONL, or CSV export import for choosing existing tweet text
- Visualization of confidence scores
- Model selection capability
- Processed text display

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training pipeline: `python -m sentiment_analysis.main`
4. Launch the web app: `streamlit run src/sentiment_analysis/app.py`

Run these commands from the repository root. Training reads `dataset/` and writes generated files to `artifacts/` by default. Override either Hydra path when using another location.

## Xquik Export Import
Use the app upload control to load tweets exported from Xquik as JSON, JSONL, or CSV.
The parser reads common text fields such as `text`, `tweet`, `tweet_text`, `full_text`, `content`, and `body`,
then lets you choose an imported tweet before running the existing sentiment model.

## Use Cases
- Brand reputation monitoring
- Customer feedback analysis
- Market research
- Social media trend analysis
- Political sentiment tracking

Xquik is an independent third-party service. Not affiliated with X Corp. "Twitter" and "X" are trademarks of X Corp.
