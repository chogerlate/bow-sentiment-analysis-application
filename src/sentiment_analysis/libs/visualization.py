import matplotlib.pyplot as plt
import seaborn as sns

def visualize_confusion_matrix(conf_matrix, class_names, filename='confusion_matrix.png'):
    """
    Visualize the confusion matrix.

    Args:
        conf_matrix (np.ndarray): The confusion matrix.
        class_names (list): The class names.
        filename (str, optional): The filename to save the plot. Defaults to 'confusion_matrix.png'.

    Returns:
        str: The filename of the saved plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()
    
    return filename

def visualize_sentiment_distribution(df, filename='sentiment_distribution.png'):
    """
    Visualize the sentiment distribution.

    Args:
        df (pd.DataFrame): The dataframe to visualize.
        filename (str, optional): The filename to save the plot. Defaults to 'sentiment_distribution.png'.

    Returns:
        str: The filename of the saved plot.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.savefig(filename)
    plt.close()
    
    return filename

def visualize_model_comparison(models_results, filename='model_comparison.png'):
    """
    Visualize the model comparison.

    Args:
        models_results (dict): The models results.
        filename (str, optional): The filename to save the plot. Defaults to 'model_comparison.png'.
    Returns:
        str: The filename of the saved plot.
    """
    models = list(models_results.keys())
    accuracies = [results['accuracy'] for results in models_results.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    plt.ylim(0, 1.0)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.savefig(filename)
    plt.close()
    
    return filename 