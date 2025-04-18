# Train a tree-based model to predict the next item_id for user 0 only
# No scaling or normalization is applied to the features

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # -------------------------------------------------------------------------
    # 1) Load Preprocessed Data
    # -------------------------------------------------------------------------
    data_path = 'data/processed_data.csv'
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Filter data for user 0 only
    df = df[df['user_id'] == 0].copy()
    print(f"Data shape for user 0: {df.shape}")
    
    # Ensure data is sorted by timestamp
    df.sort_values(by=['timestamp'], inplace=True)
    
    # -------------------------------------------------------------------------
    # 2) Define Features (X) and Label (y)
    # -------------------------------------------------------------------------
    # Use raw features without scaling or normalization
    X = df[['user_id', 'view_time', 'click_rate']]
    y = df['item_id']
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of unique items to predict: {y.nunique()}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # -------------------------------------------------------------------------
    # 3) Train Tree-based Models
    # -------------------------------------------------------------------------
    tree_models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0.0
    best_model_name = None
    
    # Loop through each model and log the experiment using MLflow
    for model_name, model in tree_models.items():
        with mlflow.start_run(run_name=f"User0_{model_name}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("user_id", 0)
            mlflow.log_param("test_size", 0.2)
            
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluate performance
            accuracy = accuracy_score(y_test, y_pred)
            # Fix warnings by setting zero_division=0 in classification_report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision_macro", report['macro avg']['precision'])
            mlflow.log_metric("recall_macro", report['macro avg']['recall'])
            mlflow.log_metric("f1_macro", report['macro avg']['f1-score'])
            
            # Log the model artifact with input example to fix the signature warning
            sample_input = X_train.iloc[:1].copy()
            mlflow.sklearn.log_model(model, model_name, input_example=sample_input)
            
            print(f"Model: {model_name}, Accuracy: {accuracy:.4f}")
            
            # Track the best model based on accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
    
    # -------------------------------------------------------------------------
    # 4) Save the Best Model
    # -------------------------------------------------------------------------
    print(f"\nBest model: {best_model_name} with accuracy = {best_accuracy:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    best_model_path = f"models/user0_{best_model_name}_best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to: {best_model_path}")
    
    # Save feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        feature_names = X.columns
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature importance:")
        for i in range(len(feature_names)):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

if __name__ == "__main__":
    main()
