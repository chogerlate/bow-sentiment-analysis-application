from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    class_names = model.classes_
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=class_names)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    
    return accuracy, report, conf_matrix, class_names

def compare_models(models_results):
    best_model_name = max(models_results.items(), key=lambda x: x[1]['accuracy'])[0]
    return best_model_name, models_results[best_model_name]
