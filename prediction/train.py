import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import time
from data_preprocessing import DataPreprocessor
import wandb  # Import wandb

# Add wandb configuration details (optional)
# wandb.login(key="YOUR_WANDB_API_KEY")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class ModelTrainer:
    def __init__(self, output_dir='./models'):
        """
        Initialize ModelTrainer with output directory for saving models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}
        self.model_performances = {}
        self.best_nn_loss = float('inf') # Track best NN loss for checkpointing
    
    def train_decision_tree(self, X_train, y_train, max_depth=10, random_state=42):
        """
        Train a Decision Tree model
        """
        print("Training Decision Tree model...")
        start_time = time.time()
        
        dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        dt_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Decision Tree model trained in {training_time:.2f} seconds")
        
        self.models['decision_tree'] = dt_model
        # Log parameters and time to wandb
        wandb.log({'decision_tree_training_time': training_time})
        return dt_model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
        """
        Train a Random Forest model
        """
        print("Training Random Forest model...")
        start_time = time.time()
        
        rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                         random_state=random_state, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Random Forest model trained in {training_time:.2f} seconds")
        
        self.models['random_forest'] = rf_model
        # Log parameters and time to wandb
        wandb.log({'random_forest_training_time': training_time})
        return rf_model
    
    def train_xgboost(self, X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Train an XGBoost model
        """
        print("Training XGBoost model...")
        start_time = time.time()
        
        xgb_model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                 learning_rate=learning_rate, random_state=random_state)
        xgb_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"XGBoost model trained in {training_time:.2f} seconds")
        
        self.models['xgboost'] = xgb_model
        # Log parameters and time to wandb
        wandb.log({'xgboost_training_time': training_time})
        return xgb_model
    
    def train_neural_network(self, X_train, y_train, input_size, batch_size=32, num_epochs=100, 
                             learning_rate=0.001, device=None):
        """
        Train a Neural Network model
        """
        print("Training Neural Network model...")
        start_time = time.time()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
        
        # Create TensorDataset and DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize the model
        model = NeuralNetwork(input_size).to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Track best model
        best_loss = float('inf')
        best_model_path = os.path.join(self.output_dir, 'neural_network_best_model.pth')

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch+1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
                # Log loss to wandb
                wandb.log({"epoch": epoch + 1, "neural_network_loss": avg_loss})

                # Checkpoint the best model based on loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Checkpoint saved for epoch {epoch+1} with loss {avg_loss:.4f}")

        training_time = time.time() - start_time
        print(f"Neural Network model trained in {training_time:.2f} seconds")
        print(f"Best NN model saved to {best_model_path} with loss {best_loss:.4f}")
        
        # Load the best model for evaluation
        model.load_state_dict(torch.load(best_model_path))
        self.models['neural_network'] = model

        # Log training time and best loss
        wandb.log({'neural_network_training_time': training_time, 'neural_network_best_loss': best_loss})
        
        return model
    
    def evaluate_models(self, X_test, y_test, device=None):
        """
        Evaluate all trained models on the test set
        """
        print("Evaluating models...")
        results = {}
        
        for name, model in self.models.items():
            if name == 'neural_network':
                if device is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test).to(device)
                    predictions = model(X_tensor).cpu().numpy().flatten()
            else:
                predictions = model.predict(X_test)           
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)    
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Predictions': predictions
            }
            
            print(f"{name.upper()} Model Performance:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R2: {r2:.4f}")

            # Log metrics to wandb
            wandb.log({
                f"{name}_MSE": mse,
                f"{name}_RMSE": rmse,
                f"{name}_MAE": mae,
                f"{name}_R2": r2,
            })
        
        self.model_performances = results
        return results
    
    def visualize_results(self, y_test):
        """
        Create visualizations of model performance
        """
        if not self.model_performances:
            print("No model performance data available. Run evaluate_models first.")
            return
        # Plot actual vs predicted values for each model
        plt.figure(figsize=(16, 12))
        for i, (name, perf) in enumerate(self.model_performances.items(), 1):
            plt.subplot(2, 2, i)
            plt.scatter(y_test, perf['Predictions'], alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
            plt.title(f'{name.upper()}: Actual vs Predicted')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'actual_vs_predicted.png'))    
        # Plot comparison of model performance metrics
        metrics = ['MSE', 'RMSE', 'MAE']
        models = list(self.model_performances.keys())  
        plt.figure(figsize=(12, 6))   
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            values = [perf[metric] for perf in self.model_performances.values()]
            sns.barplot(x=models, y=values)
            plt.title(metric)
            plt.xticks(rotation=45)  
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'))   
        # Plot R2 scores
        plt.figure(figsize=(10, 6))
        r2_scores = [perf['R2'] for perf in self.model_performances.values()]
        sns.barplot(x=models, y=r2_scores)
        plt.title('R² Score by Model')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'r2_comparison.png'))
        # Log plots to wandb
        wandb.log({
            "Actual_vs_Predicted_Plot": wandb.Image(os.path.join(self.output_dir, 'actual_vs_predicted.png')),
            "Model_Comparison_Plot": wandb.Image(os.path.join(self.output_dir, 'model_comparison.png')),
            "R2_Comparison_Plot": wandb.Image(os.path.join(self.output_dir, 'r2_comparison.png'))
        })    
    def save_models(self):
        """
        Save all trained models to disk
        """
        print("Saving models...")       
        for name, model in self.models.items():
            if name == 'neural_network':
                # Save final PyTorch model state dictionary
                final_model_path = os.path.join(self.output_dir, f'{name}_final_model.pth')
                torch.save(model.state_dict(), final_model_path)
                # Log best and final model as artifacts
                best_model_artifact = wandb.Artifact(f'{name}-best-model', type='model')
                best_model_artifact.add_file(os.path.join(self.output_dir, 'neural_network_best_model.pth'))
                wandb.log_artifact(best_model_artifact)

                final_model_artifact = wandb.Artifact(f'{name}-final-model', type='model')
                final_model_artifact.add_file(final_model_path)
                wandb.log_artifact(final_model_artifact)
            else:
                # Save sklearn/xgboost model
                model_path = os.path.join(self.output_dir, f'{name}_model.joblib')
                joblib.dump(model, model_path)
                # Log model as artifact
                model_artifact = wandb.Artifact(f'{name}-model', type='model')
                model_artifact.add_file(model_path)
                wandb.log_artifact(model_artifact)
        # Save the preprocessor objects as well
        joblib.dump(preprocessor.standard_scaler, os.path.join(self.output_dir, 'standard_scaler.joblib'))
        for name, encoder in preprocessor.label_encoders.items():
            joblib.dump(encoder, os.path.join(self.output_dir, f'{name}_encoder.joblib'))
        pd.DataFrame({'feature_name': preprocessor.feature_names}).to_csv(
            os.path.join(self.output_dir, 'feature_names.csv'), index=False
        )
        # Log preprocessor artifacts
        preprocessor_artifact = wandb.Artifact('preprocessor', type='preprocessing')
        preprocessor_artifact.add_file(os.path.join(self.output_dir, 'standard_scaler.joblib'))
        for name in preprocessor.label_encoders.keys():
            preprocessor_artifact.add_file(os.path.join(self.output_dir, f'{name}_encoder.joblib'))
        preprocessor_artifact.add_file(os.path.join(self.output_dir, 'feature_names.csv'))
        wandb.log_artifact(preprocessor_artifact)
        
        print(f"Models and preprocessor artifacts saved to {self.output_dir} and logged to WandB")

# Global preprocessor instance to be accessible in save_models
preprocessor = None 

def main():
    global preprocessor # Allow modification of the global preprocessor

    # Initialize wandb run
    wandb.init(project="salary-prediction-vn", job_type="train")
    print(f"WandB Run URL: {wandb.run.get_url()}")

    # --- Hyperparameters --- 
    dt_max_depth = 10
    rf_n_estimators = 100
    rf_max_depth = 10
    xgb_n_estimators = 100
    xgb_max_depth = 6
    xgb_learning_rate = 0.1
    nn_batch_size = 64
    nn_num_epochs = 50 # Reduced for faster example run
    nn_learning_rate = 0.001
    test_split_size = 0.2
    random_state = 42

    # Log hyperparameters to wandb
    wandb.config.update({
        "decision_tree_max_depth": dt_max_depth,
        "random_forest_n_estimators": rf_n_estimators,
        "random_forest_max_depth": rf_max_depth,
        "xgboost_n_estimators": xgb_n_estimators,
        "xgboost_max_depth": xgb_max_depth,
        "xgboost_learning_rate": xgb_learning_rate,
        "nn_batch_size": nn_batch_size,
        "nn_num_epochs": nn_num_epochs,
        "nn_learning_rate": nn_learning_rate,
        "test_split_size": test_split_size,
        "random_state": random_state
    })

    # Create data preprocessing pipeline
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run_preprocessing()
    
    # Get feature count for neural network input size
    input_size = X_train.shape[1]
    wandb.config.update({"input_feature_size": input_size}) # Log input size
    
    # Create model trainer
    trainer = ModelTrainer(output_dir='./models')
    
    # Train models
    trainer.train_decision_tree(X_train, y_train, max_depth=dt_max_depth, random_state=random_state)
    trainer.train_random_forest(X_train, y_train, n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_state)
    trainer.train_xgboost(X_train, y_train, n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate, random_state=random_state)
    trainer.train_neural_network(X_train, y_train, input_size, batch_size=nn_batch_size, num_epochs=nn_num_epochs, learning_rate=nn_learning_rate)
    # Evaluate models
    trainer.evaluate_models(X_test, y_test)
    # Visualize results (logging is now inside visualize_results)
    trainer.visualize_results(y_test)
    # Save models (logging is now inside save_models)
    trainer.save_models()
    wandb.finish() # Finish the wandb run
if __name__ == "__main__":
    main() 