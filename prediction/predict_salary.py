import os
import sys
import argparse
import pandas as pd
import joblib
import torch
import numpy as np
from train import NeuralNetwork
from data_preprocessing import DataPreprocessor

class SalaryPredictor:
    def __init__(self, models_dir='./models'):
        """
        Initialize SalaryPredictor with the directory containing trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.loaded_models = False
        self.preprocessor = DataPreprocessor()
        
        # Try to load preprocessor's scaler and label encoders
        self.load_preprocessor_artifacts()
    
    def load_preprocessor_artifacts(self):
        """
        Load preprocessor artifacts (scaler, label encoders) if available
        """
        scaler_path = os.path.join(self.models_dir, 'standard_scaler.joblib')
        if os.path.exists(scaler_path):
            self.preprocessor.standard_scaler = joblib.load(scaler_path)
        
        # Load label encoders for categorical features
        for feature in ['Field', 'Experience', 'Location', 'Company Size']:
            encoder_path = os.path.join(self.models_dir, f'{feature}_encoder.joblib')
            if os.path.exists(encoder_path):
                self.preprocessor.label_encoders[feature] = joblib.load(encoder_path)
    
    def load_models(self, input_size=None):
        """
        Load all trained models from disk
        """
        print("Loading models...")
        
        # Load Decision Tree, Random Forest, and XGBoost models
        for model_name in ['decision_tree', 'random_forest', 'xgboost']:
            model_path = os.path.join(self.models_dir, f'{model_name}_model.joblib')
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model")
            else:
                print(f"Warning: {model_name} model not found at {model_path}")
        
        # Load Neural Network model if input_size is provided
        # Default to loading the *best* checkpoint
        nn_model_path = os.path.join(self.models_dir, 'neural_network_best_model.pth')
        if os.path.exists(nn_model_path) and input_size is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NeuralNetwork(input_size).to(device)
            model.load_state_dict(torch.load(nn_model_path, map_location=device))
            model.eval()
            self.models['neural_network'] = model
            print(f"Loaded neural network model")
        elif os.path.exists(nn_model_path):
            # Try to infer input size from feature names
            feature_names_path = os.path.join(self.models_dir, 'feature_names.csv')
            if os.path.exists(feature_names_path):
                feature_names = pd.read_csv(feature_names_path)
                input_size = len(feature_names)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = NeuralNetwork(input_size).to(device)
                model.load_state_dict(torch.load(nn_model_path, map_location=device))
                model.eval()
                self.models['neural_network'] = model
                print(f"Loaded neural network model with inferred input size: {input_size}")
            else:
                print("Warning: input_size not provided and couldn't be inferred, neural network model not loaded")
        else:
            print(f"Warning: neural network model not found at {nn_model_path}")
        
        self.loaded_models = len(self.models) > 0
        return self.loaded_models
    
    def create_job_dataframe(self, field, experience, location, company_size, requirements):
        """
        Create a DataFrame with job details for a single job
        """
        data = {
            'Field': [field],
            'Experience': [experience],
            'Location': [location],
            'Company Size': [company_size],
            'Job Requirements': [requirements],
            'Salary': ['Thoả thuận']  # Placeholder value
        }
        
        return pd.DataFrame(data)
    
    def preprocess_job_data(self, job_df):
        """
        Preprocess job data for prediction
        """
        # Make a copy of the preprocessor to avoid modifying the original
        # Ensure preprocessor artifacts are loaded first
        self.load_preprocessor_artifacts()
        preprocessor_copy = self.preprocessor
        
        # Set dataframe
        preprocessor_copy.df = job_df.copy()
        
        # Run preprocessing steps
        # Skip clean_salary as we're predicting salary
        preprocessor_copy.df['Salary_Value'] = np.nan  # Placeholder
        
        # Extract features from requirements
        preprocessor_copy.extract_features_from_requirements()
        
        # Encode categorical features using loaded encoders
        for col in ['Field', 'Experience', 'Location', 'Company Size']:
            if col in preprocessor_copy.df.columns:
                # Handle new categories not seen during training
                try:
                    if col in preprocessor_copy.label_encoders:
                        encoder = preprocessor_copy.label_encoders[col]
                        preprocessor_copy.df[f'{col}_encoded'] = encoder.transform(preprocessor_copy.df[col].astype(str))
                    else:
                        # If encoder not available, use a default value
                        print(f"Warning: No encoder found for {col}, using default value 0")
                        preprocessor_copy.df[f'{col}_encoded'] = 0
                except:
                    print(f"Warning: Error encoding {col}, using default value 0")
                    preprocessor_copy.df[f'{col}_encoded'] = 0
        
        # Get feature columns
        feature_cols = [col for col in preprocessor_copy.df.columns if col.endswith('_encoded') or col.startswith('has_')]
        
        if not feature_cols:
            print("Error: No features available for prediction")
            return None
        
        # Extract features
        X = preprocessor_copy.df[feature_cols]
        
        # Scale features
        try:
            X_scaled = preprocessor_copy.standard_scaler.transform(X)
        except:
            print("Warning: Error scaling features, using unscaled features")
            X_scaled = X.values
        
        return X_scaled, feature_cols
    
    def predict(self, field, experience, location, company_size, requirements, model_name=None):
        """
        Predict salary based on job details
        
        Parameters:
        - field: Job field/industry
        - experience: Experience requirement
        - location: Job location
        - company_size: Company size
        - requirements: Job requirements text
        - model_name: Optional, specific model to use for prediction
        
        Returns:
        - Dictionary with predictions from all models or the specified model
        """
        if not self.loaded_models:
            loaded = self.load_models()
            if not loaded:
                print("Error: No models available")
                return None
        
        # Create dataframe with job details
        job_df = self.create_job_dataframe(field, experience, location, company_size, requirements)
        
        # Preprocess job data
        X_scaled, feature_names = self.preprocess_job_data(job_df)
        
        if X_scaled is None:
            return None
        
        # Make predictions
        predictions = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name is not None:
            # Use specific model if requested
            if model_name not in self.models:
                print(f"Error: Model '{model_name}' not found")
                return None
            
            model = self.models[model_name]
            if model_name == 'neural_network':
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                with torch.no_grad():
                    pred = model(X_tensor).cpu().numpy().flatten()[0]
            else:
                pred = model.predict(X_scaled)[0]
            
            predictions[model_name] = pred
        else:
            # Use all models
            for name, model in self.models.items():
                if name == 'neural_network':
                    X_tensor = torch.FloatTensor(X_scaled).to(device)
                    with torch.no_grad():
                        pred = model(X_tensor).cpu().numpy().flatten()[0]
                else:
                    pred = model.predict(X_scaled)[0]
                
                predictions[name] = pred
        
        return predictions
    
    def get_ensemble_prediction(self, predictions):
        """
        Get an ensemble prediction (average of all model predictions)
        """
        if not predictions:
            return None
        
        values = list(predictions.values())
        return sum(values) / len(values)

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Predict salary based on job details')
    parser.add_argument('--field', required=True, help='Job field/industry')
    parser.add_argument('--experience', required=True, help='Experience requirement (e.g., "2 năm")')
    parser.add_argument('--location', required=True, help='Job location')
    parser.add_argument('--company_size', required=True, help='Company size (e.g., "25-99 nhân viên")')
    parser.add_argument('--requirements', required=True, help='Job requirements')
    parser.add_argument('--model', choices=['decision_tree', 'random_forest', 'xgboost', 'neural_network', 'ensemble'], 
                        default='ensemble', help='Model to use for prediction (default: ensemble)')
    parser.add_argument('--models_dir', default='./models', help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SalaryPredictor(models_dir=args.models_dir)
    
    # Make prediction
    if args.model == 'ensemble':
        # Use all models
        predictions = predictor.predict(
            args.field, args.experience, args.location, args.company_size, args.requirements
        )
        
        if predictions:
            # Print individual model predictions
            print("\nModel predictions (million VND):")
            for model_name, pred in predictions.items():
                print(f"  {model_name}: {pred:.2f}")
            
            # Print ensemble prediction
            ensemble_pred = predictor.get_ensemble_prediction(predictions)
            print(f"\nEnsemble prediction: {ensemble_pred:.2f} million VND")
        else:
            print("Error making predictions")
    else:
        # Use specific model
        predictions = predictor.predict(
            args.field, args.experience, args.location, args.company_size, args.requirements, 
            model_name=args.model
        )
        
        if predictions and args.model in predictions:
            print(f"\nPredicted salary ({args.model}): {predictions[args.model]:.2f} million VND")
        else:
            print("Error making prediction")

if __name__ == "__main__":
    main() 