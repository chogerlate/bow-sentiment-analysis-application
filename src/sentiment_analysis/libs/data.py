import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

def load_data(train_path, test_path):
    """
    Load the data from the train and test paths.

    Args:
        train_path (str): The path to the training data.
        test_path (str): The path to the test data.

    Raises:
        ValueError: If the data cannot be read with any of the encodings.

    Returns:
        tuple: A tuple containing the training dataframe and the test dataframe.
    """
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    def try_read_csv(file_path):
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Failed to read {file_path} with any of the following encodings: {encodings}")
    
    train_df = try_read_csv(train_path)
    test_df = try_read_csv(test_path)
    return train_df, test_df

def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing URLs, mentions, hashtags, and HTML tags, removing non-alphabetic characters, and stemming.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags, and HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'<.*?>', '', str(text))
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stem the tokens
    lancaster = LancasterStemmer()
    stemmed_tokens = [lancaster.stem(word) for word in filtered_tokens]
    
    return ' '.join(stemmed_tokens)

def prepare_data(df):
    """
    Prepare the data by normalizing the text and creating a processed text column.
    Also drop rows with missing text or sentiment.
    Args:
        df (pd.DataFrame): The dataframe to prepare.

    Returns:
        pd.DataFrame: The prepared dataframe.
    """
    df.dropna(subset=['text', 'sentiment'], inplace=True)
    df['processed_text'] = df['text'].apply(normalize_text)
    return df

def create_bow_features(
    train_data, 
    test_data=None,
    max_features=3000,
    ngram_range=(1, 1)
):
    """
    Create bag-of-words features from the processed text.

    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame, optional): The test data.
        max_features (int, optional): The maximum number of features. Defaults to 6000 which is double the size of Oxford 3000.
        ngram_range (tuple, optional): The ngram range.

    Returns:
        tuple: A tuple containing the training features, the test features, and the vectorizer.
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    X_train = vectorizer.fit_transform(train_data['processed_text'])
    
    if test_data is not None:
        X_test = vectorizer.transform(test_data['processed_text'])
        return X_train, X_test, vectorizer
    
    return X_train, vectorizer

def init_nltk():
    """
    Initialize the NLTK library.

    Args:
        None.

    Returns:
        None.
    """
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)