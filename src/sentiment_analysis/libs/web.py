import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sentiment_analysis.libs.data import normalize_text
import os
import hydra
from omegaconf import DictConfig

def load_model(model_path):
    """
    Load the model from the model path.

    Args:
        model_path (str): The path to the model.

    Returns:
        The model.
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)
    
def load_vectorizer(vectorizer_path):
    """
    Load the vectorizer from the vectorizer path.

    Args:
        vectorizer_path (str): The path to the vectorizer.

    Returns:
        The vectorizer.
    """
    with open(vectorizer_path, 'rb') as f:
        return pickle.load(f)

def predict_sentiment(text, model, vectorizer):
    """
    Predict the sentiment of the text.

    Args:
        text (str): The text to predict the sentiment of.
        model (sklearn.ensemble.RandomForestClassifier): The model to predict the sentiment with.
        vectorizer (sklearn.feature_extraction.text.CountVectorizer): The vectorizer to transform the text with.
    
    Returns:
        The prediction, the probabilities, and the processed text.
    """
    processed_text = normalize_text(text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return prediction, probabilities, processed_text
def create_app():
    st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide", initial_sidebar_state="expanded")
    
    with st.sidebar:
        st.title('Twitter Sentiment Analysis')
        st.markdown("---")
        st.write("This tool analyzes tweet sentiment as positive, negative, or neutral.")
        
    try:
        try:
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        except:
            pass
        
        hydra.initialize(version_base=None, config_path="../../../configs")
        config = hydra.compose(config_name="config")
        
        model_paths = {
            "Best Model": config.models.best_model.file,
            "Naive Bayes": config.models.naive_bayes.file,
            "Logistic Regression": config.models.logistic_regression.file,
            "Random Forest": config.models.random_forest.file
        }
        
        vectorizer_path = config.models.vectorizer.file
        
        if not os.path.exists(vectorizer_path):
            st.error("Vectorizer not found. Please train the model first.")
            return
        
        available_models = {name: path for name, path in model_paths.items() if os.path.exists(path)}
        
        if not available_models:
            st.error("No trained models found. Please train at least one model first.")
            return
        
        vectorizer = load_vectorizer(vectorizer_path)
        
        with st.sidebar:
            selected_model_name = st.selectbox("Select model:", list(available_models.keys()))
            st.markdown("---")
            st.info("💡 Enter your tweet in the main panel and click 'Analyze' to see results")
        
        model = load_model(available_models[selected_model_name])
        
        st.header("Tweet Analysis")
        tweet_text = st.text_area("Enter a tweet to analyze:", height=100)
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)
        
        if analyze_button:
            if not tweet_text:
                st.warning("Please enter some text to analyze.")
                return
            
            with st.spinner("Analyzing sentiment..."):
                prediction, probabilities, processed_text = predict_sentiment(tweet_text, model, vectorizer)
            
            sentiment_color = {
                "positive": "#28a745",
                "negative": "#dc3545",
                "neutral": "#ffc107"
            }
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"## Sentiment: <span style='color:{sentiment_color.get(prediction, 'blue')}'>{prediction.upper()}</span>", unsafe_allow_html=True)
                st.markdown("### Processed Text")
                st.text_area("", processed_text, height=100, disabled=True)
            
            with col2:
                st.markdown("### Confidence Scores")
                
                class_names = model.classes_
                proba_df = pd.DataFrame({
                    'Sentiment': class_names,
                    'Probability': probabilities
                })
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = sns.barplot(x='Sentiment', y='Probability', data=proba_df, ax=ax, palette=["#dc3545", "#ffc107", "#28a745"])
                
                for p in bars.patches:
                    bars.annotate(f'{p.get_height():.2%}', 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='bottom', 
                                 xytext=(0, 5), textcoords='offset points')
                
                plt.ylim(0, 1)
                st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
