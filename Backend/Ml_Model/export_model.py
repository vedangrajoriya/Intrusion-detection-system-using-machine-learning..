# ml_model/export_model.py

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import os
import warnings
import gc

warnings.filterwarnings('ignore')

def export_trained_model():
    """Export trained model components for Django integration"""
    
    print("Loading and preprocessing data for model export...")
    
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, "cleaned_for_training.csv")
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file not found at {dataset_path}")
            return False
        
        # Load dataset
        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Get target column (assuming it's the last column)
        target_column = df.columns[-1]
        print(f"Target column: {target_column}")
        
        # Split features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target unique values: {y.unique()}")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Select top features (adjust k as needed)
        k = min(20, X.shape[1])  # Select top 20 features or all if less
        print(f"Selecting top {k} features...")
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"\nSelected {len(selected_features)} features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"  {i}. {feature}")
        
        # Encode target labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"\nTarget classes: {le.classes_}")
        print(f"Encoded target unique values: {np.unique(y_encoded)}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train XGBoost model
        print("\n" + "="*50)
        print("Training XGBoost model...")
        print("="*50)
        
        num_classes = len(np.unique(y_encoded))
        
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,  # Reduced for faster training
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
            num_class=num_classes if num_classes > 2 else None,
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        # Evaluate model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print("\n" + "="*50)
        print("Model Performance:")
        print("="*50)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        # Create output directory
        output_dir = script_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model components
        print(f"\n" + "="*50)
        print(f"Saving model components to {output_dir}/")
        print("="*50)
        
        model_path = os.path.join(output_dir, 'xgboost_model.pkl')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        le_path = os.path.join(output_dir, 'label_encoder.pkl')
        features_path = os.path.join(output_dir, 'selected_features.pkl')
        
        joblib.dump(model, model_path)
        print(f"✓ Saved: xgboost_model.pkl")
        
        joblib.dump(scaler, scaler_path)
        print(f"✓ Saved: scaler.pkl")
        
        joblib.dump(le, le_path)
        print(f"✓ Saved: label_encoder.pkl")
        
        joblib.dump(selected_features, features_path)
        print(f"✓ Saved: selected_features.pkl")
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_path, index=False)
        print(f"✓ Saved: feature_importance.csv")
        
        # Test model loading
        print("\n" + "="*50)
        print("Testing model loading...")
        print("="*50)
        
        loaded_model = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)
        loaded_le = joblib.load(le_path)
        loaded_features = joblib.load(features_path)
        
        print("✅ All files loaded successfully!")
        print(f"Model type: {type(loaded_model)}")
        print(f"Selected features count: {len(loaded_features)}")
        print(f"Target classes: {loaded_le.classes_}")
        
        # Test prediction
        if len(X_test) > 0:
            test_pred = loaded_model.predict(X_test[:1])
            test_proba = loaded_model.predict_proba(X_test[:1])
            print(f"\nTest prediction: {loaded_le.inverse_transform(test_pred)[0]}")
            print(f"Test confidence: {np.max(test_proba) * 100:.2f}%")
        
        print("\n" + "="*50)
        print("✅ MODEL EXPORT COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during model export: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("XGBoost Model Export for Django Integration")
    print("="*50)
    print()
    
    success = export_trained_model()
    
    if success:
        print("\n" + "="*50)
        print("NEXT STEPS:")
        print("="*50)
        print("1. ✓ Model files are now in the ml_model/ folder")
        print("2. Create ml_analyzer.py in ids_api/ folder")
        print("3. Update views.py to use XGBoostURLAnalyzer")
        print("4. Start Django server: python manage.py runserver")
        print("5. Test with a URL from your frontend")
        print("="*50)
    else:
        print("\n❌ Export failed. Please check the error messages above.")