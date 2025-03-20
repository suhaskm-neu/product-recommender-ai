import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # -------------------------------------------------------------------------
    # 1) Load Preprocessed Data
    # -------------------------------------------------------------------------
    data_path = 'data/processed_data.csv'  # Adjust if your final CSV is named differently
    df = pd.read_csv(data_path)
    
    # If 'timestamp' is not numeric, convert it to a numeric type or
    # confirm it’s already seconds from epoch in your pre-processing script.
    # e.g., df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    # -------------------------------------------------------------------------
    # 2) Sort & Create 'next_item_id'
    #    We assume the data is sorted by user & time in the pre-processing step,
    #    but let's ensure it again.
    # -------------------------------------------------------------------------
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    
    # Group by user, shift item_id by -1 to get the "next item"
    df['next_item_id'] = df.groupby('user_id')['item_id'].shift(-1)
    
    # Drop rows where next_item_id is NaN (the last item for each user)
    df.dropna(subset=['next_item_id'], inplace=True)
    
    # -------------------------------------------------------------------------
    # 3) Label Encode user_id, item_id, and next_item_id
    # -------------------------------------------------------------------------
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()  # We can use the same encoder for item_id & next_item_id
                                  # or separate ones if items differ across tasks.
    
    df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
    df['item_id_encoded'] = item_encoder.fit_transform(df['item_id'])
    df['next_item_id_encoded'] = item_encoder.fit_transform(df['next_item_id'])
    
    # -------------------------------------------------------------------------
    # 4) (Optional) Create or use a binary 'target' feature
    #    If you still want to keep the idea of click_rate > median,
    #    treat it as an additional input variable. For next-item prediction,
    #    the real label is next_item_id_encoded.
    # -------------------------------------------------------------------------
    df['binary_target'] = (df['click_rate'] > df['click_rate'].median()).astype(int)
    
    # -------------------------------------------------------------------------
    # 5) Define Features (X) and Label (y)
    #    We'll predict 'next_item_id_encoded' as a multi-class classification.
    # -------------------------------------------------------------------------
    # Example features:
    #   - user_id_encoded, item_id_encoded, timestamp, view_time, click_rate, binary_target
    # Adjust as needed.
    X = df[['user_id_encoded',
            'item_id_encoded',
            'timestamp',
            'view_time',
            'click_rate',
            'binary_target']]
    
    y = df['next_item_id_encoded']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True
    )
    
    # -------------------------------------------------------------------------
    # 6) Train Multiple Models and Compare
    # -------------------------------------------------------------------------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    
    best_model = None
    best_accuracy = 0.0
    best_model_name = None
    
    # Loop through each model, start an MLflow run for each
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", 0.2)
            
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            # For multi-class, log macro-averaged precision, recall, f1
            mlflow.log_metric("precision_macro", report['macro avg']['precision'])
            mlflow.log_metric("recall_macro", report['macro avg']['recall'])
            mlflow.log_metric("f1_macro", report['macro avg']['f1-score'])
            
            # Log the model itself
            mlflow.sklearn.log_model(model, model_name)
            
            # Print a quick summary
            print(f"Model: {model_name}, Accuracy: {accuracy:.4f}")
            
            # Track the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
    
    # -------------------------------------------------------------------------
    # 7) Save the Best Model
    # -------------------------------------------------------------------------
    print(f"\nBest model: {best_model_name} with accuracy = {best_accuracy:.4f}")
    best_model_path = f"models/{best_model_name}_best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main()
