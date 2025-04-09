import os
import pickle
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from sentiment_analysis.libs.data import load_data, prepare_data, create_bow_features, init_nltk
from sentiment_analysis.libs.models import train_model
from sentiment_analysis.libs.evaluation import evaluate_model, compare_models
from sentiment_analysis.libs.visualization import visualize_confusion_matrix, visualize_sentiment_distribution, visualize_model_comparison
from sklearn.model_selection import train_test_split

def create_directories(config: DictConfig):
    os.makedirs(config.paths.models, exist_ok=True)
    os.makedirs(config.paths.visualizations, exist_ok=True)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    try:
        print("Initializing training process...")
        create_directories(config)
        init_nltk()
        
        print("Loading data...")
        try:
            train_df, test_df = load_data(config.paths.train_data, config.paths.test_data)
        except ValueError as e:
            print(f"Error loading data: {str(e)}")
            print("Please ensure your data files are in a supported encoding (UTF-8, Latin1, ISO-8859-1, or CP1252)")
            return
        except Exception as e:
            print(f"Unexpected error loading data: {str(e)}")
            return
        
        print("Preprocessing data...")
        train_df = prepare_data(train_df)
        test_df = prepare_data(test_df)
        
        dist_file = config.visualizations.sentiment_distribution.file
        visualize_sentiment_distribution(train_df, dist_file)
        print(f"Sentiment distribution visualization saved as '{dist_file}'")
        
        print("Creating features...")
        X_train, X_test, vectorizer = create_bow_features(train_df, test_df)
        
        print("Preparing target variable...")
        y_train = train_df['sentiment']
        y_test = test_df['sentiment']
        
        print("Splitting data for validation...")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, 
            test_size=config.data.test_size, 
            random_state=config.data.random_state
        )
        
        models_to_train = {
            'naive_bayes': config.models.naive_bayes.name,
            'logistic_regression': config.models.logistic_regression.name,
            'random_forest': config.models.random_forest.name
        }
        
        models_results = {}
        test_results = {}
        
        for model_type, model_name in models_to_train.items():
            print(f"Training {model_name} model...")
            model = train_model(X_train_split, y_train_split, model_type)
            
            print(f"Evaluating {model_name} model on validation set...")
            accuracy, report, conf_matrix = evaluate_model(model, X_val_split, y_val_split)
            
            models_results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'report': report,
                'conf_matrix': conf_matrix
            }
            
            print(f"{model_name} Validation Accuracy: {accuracy:.4f}")
            print("Validation Classification Report:")
            print(report)
            
            print(f"Evaluating {model_name} model on test set...")
            test_accuracy, test_report, test_conf_matrix = evaluate_model(model, X_test, y_test)
            
            test_results[model_name] = {
                'accuracy': test_accuracy,
                'report': test_report,
                'conf_matrix': test_conf_matrix
            }
            
            print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
            print("Test Classification Report:")
            print(test_report)
            
            model_file = config.models[model_type].file
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        best_model_name, best_model_results = compare_models(models_results)
        print(f"Best model: {best_model_name} with validation accuracy {best_model_results['accuracy']:.4f}")
        print(f"Test accuracy for best model: {test_results[best_model_name]['accuracy']:.4f}")
        
        best_model_file = config.models.best_model.file
        with open(best_model_file, 'wb') as f:
            pickle.dump(best_model_results['model'], f)
        
        vectorizer_file = config.models.vectorizer.file
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        cm_file = config.visualizations.confusion_matrix.file
        visualize_confusion_matrix(
            best_model_results['conf_matrix'], 
            train_df['sentiment'].unique(),
            cm_file
        )
        print(f"Confusion matrix visualization saved as '{cm_file}'")
        
        test_cm_file = os.path.join(os.path.dirname(cm_file), "test_confusion_matrix.png")
        visualize_confusion_matrix(
            test_results[best_model_name]['conf_matrix'],
            test_df['sentiment'].unique(),
            test_cm_file
        )
        print(f"Test confusion matrix visualization saved as '{test_cm_file}'")
        
        comp_file = config.visualizations.model_comparison.file
        visualize_model_comparison(models_results, comp_file)
        print(f"Model comparison visualization saved as '{comp_file}'")
        
        test_comp_file = os.path.join(os.path.dirname(comp_file), "test_model_comparison.png")
        visualize_model_comparison(test_results, test_comp_file)
        print(f"Test model comparison visualization saved as '{test_comp_file}'")
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()