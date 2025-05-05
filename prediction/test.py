import os
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from train import NeuralNetwork
from data_preprocessing import DataPreprocessor
import json  # For saving results

class ModelTester:
    def __init__(self, models_dir='./models', results_dir='./results'):
        """
        Initialize ModelTester with directories for models and results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True) # Ensure results dir exists
        self.models = {}
        self.predictions = {}
        self.loaded_models = False
    
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
        # Load the *best* checkpoint
        nn_model_path = os.path.join(self.models_dir, 'neural_network_best_model.pth')
        if os.path.exists(nn_model_path) and input_size is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NeuralNetwork(input_size).to(device)
            model.load_state_dict(torch.load(nn_model_path, map_location=device))
            model.eval()
            self.models['neural_network'] = model
            print(f"Loaded neural network model")
        elif input_size is None:
            print("Warning: input_size not provided, neural network model not loaded")
        else:
            print(f"Warning: neural network model not found at {nn_model_path}")
        
        self.loaded_models = len(self.models) > 0
        return self.loaded_models
    
    def predict(self, X_test, device=None):
        """
        Make predictions using all loaded models
        """
        if not self.loaded_models:
            print("No models loaded. Call load_models() first.")
            return None
        
        print("Making predictions...")
        self.predictions = {}
        
        for name, model in self.models.items():
            if name == 'neural_network':
                if device is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Convert to tensor for neural network
                X_tensor = torch.FloatTensor(X_test).to(device)
                with torch.no_grad():
                    predictions = model(X_tensor).cpu().numpy().flatten()
            else:
                predictions = model.predict(X_test)
            
            self.predictions[name] = predictions
        
        return self.predictions
    
    def evaluate(self, X_test, y_test, device=None):
        """
        Evaluate model predictions against true values
        """
        # If predictions haven't been made yet, make them
        if not self.predictions:
            self.predict(X_test, device)
        
        if not self.predictions:
            return None
        
        print("Evaluating model predictions...")
        results = {}
        
        for name, predictions in self.predictions.items():
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"{name.upper()} Model Test Performance:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R2: {r2:.4f}")
        
        # Save results to a file
        results_path = os.path.join(self.results_dir, 'evaluation_results.json')
        try:
            with open(results_path, 'w') as f:
                # Convert numpy arrays in predictions to lists for JSON serialization
                serializable_results = {k: {**v, 'Predictions': v['Predictions'].tolist()} for k, v in results.items()}
                json.dump(serializable_results, f, indent=4)
            print(f"Evaluation results saved to {results_path}")
        except Exception as e:
            print(f"Error saving evaluation results: {e}")

        return results
    
    def visualize_predictions(self, y_test, output_dir=None):
        """
        Visualize prediction results
        """
        if output_dir is None:
            output_dir = self.results_dir # Use the results directory by default

        if not self.predictions:
            print("No predictions available. Call predict() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot actual vs predicted for each model
        plt.figure(figsize=(16, 12))
        
        for i, (name, preds) in enumerate(self.predictions.items(), 1):
            plt.subplot(2, 2, i)
            plt.scatter(y_test, preds, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
            plt.title(f'{name.upper()}: Actual vs Predicted')
            plt.xlabel('Actual Salary')
            plt.ylabel('Predicted Salary')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_predictions.png'))
        
        # Create prediction residuals plot
        plt.figure(figsize=(16, 12))
        
        for i, (name, preds) in enumerate(self.predictions.items(), 1):
            residuals = y_test - preds
            plt.subplot(2, 2, i)
            plt.scatter(preds, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'{name.upper()}: Residuals')
            plt.xlabel('Predicted Salary')
            plt.ylabel('Residuals')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_residuals.png'))
        
        print(f"Visualizations saved to {output_dir}")
    
    def predict_on_new_data(self, new_data, preprocessor=None, device=None):
        """
        Make predictions on new, unprocessed data
        """
        if not self.loaded_models:
            print("No models loaded. Call load_models() first.")
            return None
        
        if preprocessor is None:
            print("Warning: No preprocessor provided. Assuming data is already preprocessed.")
            X = new_data
        else:
            # Preprocess the new data
            print("Preprocessing new data...")
            # Apply the same preprocessing steps as with the training data
            preprocessor.df = new_data.copy()  # Set the dataframe to the new data
            preprocessor.clean_salary()
            preprocessor.extract_features_from_requirements()
            preprocessor.encode_categorical_features()
            
            # Extract features
            feature_cols = [col for col in preprocessor.df.columns if col.endswith('_encoded') or col.startswith('has_')]
            X = preprocessor.df[feature_cols]
            
            # Scale the features
            X = preprocessor.standard_scaler.transform(X)
        
        # Make predictions
        return self.predict(X, device)

    def generate_report(self, results_file='evaluation_results.json'):
        """
        Generate a comparison report from saved evaluation results.
        """
        results_path = os.path.join(self.results_dir, results_file)
        if not os.path.exists(results_path):
            print(f"Error: Results file not found at {results_path}")
            return

        try:
            with open(results_path, 'r') as f:
                results_data = json.load(f)
        except Exception as e:
            print(f"Error reading results file: {e}")
            return

        report = "# Model Comparison Report\n\n"
        report += "## Performance Metrics\n\n"
        report += "| Model           | MSE      | RMSE     | MAE      | R2 Score |\n"
        report += "|-----------------|----------|----------|----------|----------|\n"

        for model_name, metrics in results_data.items():
            report += f"| {model_name.replace('_', ' ').title():<15} | {metrics.get('MSE', 0):<8.4f} | {metrics.get('RMSE', 0):<8.4f} | {metrics.get('MAE', 0):<8.4f} | {metrics.get('R2', 0):<8.4f} |\n"

        # Add a simple comparison conclusion
        best_r2_model = max(results_data, key=lambda k: results_data[k].get('R2', -1))
        worst_r2_model = min(results_data, key=lambda k: results_data[k].get('R2', 2))

        report += "\n## Conclusion\n\n"
        report += f"- Based on R² score, the **{best_r2_model.replace('_', ' ').title()}** model performed best ({results_data[best_r2_model].get('R2', 0):.4f}).\n"
        report += f"- The **{worst_r2_model.replace('_', ' ').title()}** model had the lowest R² score ({results_data[worst_r2_model].get('R2', 0):.4f}).\n"
        report += "- Lower MSE, RMSE, and MAE indicate better performance in terms of error.\n"
        
        report_path = os.path.join(self.results_dir, 'comparison_report.md')
        try:
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Comparison report saved to {report_path}")
        except Exception as e:
            print(f"Error saving comparison report: {e}")

def main():
    # Load test data using the preprocessor instance from the main script if available
    # For standalone testing, re-instantiate and run preprocessing.
    # It's better practice to load saved processed data if available.
    data_dir = '.' # Assuming processed data is saved in the prediction folder root
    try:
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test_np = np.load(os.path.join(data_dir, 'y_test.npy'))
        y_test = pd.Series(y_test_np) # Convert back to Series if needed
        feature_names_df = pd.read_csv(os.path.join(data_dir, 'feature_names.csv'))
        input_size = len(feature_names_df)
        print("Loaded preprocessed test data.")
    except FileNotFoundError:
        print("Preprocessed data not found. Running preprocessing again...")
        preprocessor = DataPreprocessor()
        _, X_test, _, y_test = preprocessor.run_preprocessing()
        input_size = X_test.shape[1]
    
    # Initialize model tester and load models
    tester = ModelTester(models_dir='./models', results_dir='./results')
    if not tester.load_models(input_size=input_size):
        print("Failed to load models. Exiting.")
        return
    
    # Evaluate models on test data
    results = tester.evaluate(X_test, y_test)
    
    if results:
        # Visualize predictions
        tester.visualize_predictions(y_test)
        # Generate comparison report
        tester.generate_report()
    else:
        print("Evaluation failed.")
    
    # --- Example of predicting on new, *unprocessed* data --- 
    # This requires loading the preprocessor artifacts correctly
    # predictor = SalaryPredictor(models_dir='./models') # Use SalaryPredictor for this
    # sample_new_data = pd.DataFrame(...) # Create a sample dataframe
    # sample_predictions = predictor.predict(...) 
    # print(sample_predictions)

if __name__ == "__main__":
    main() 