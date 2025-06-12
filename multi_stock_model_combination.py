import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

class MultiStockModelCombiner:
    def __init__(self, model_paths, stock_symbols):
        """
        Initialize the model combiner
        
        Args:
            model_paths: List of paths to individual stock models
            stock_symbols: List of stock symbols corresponding to models
        """
        self.model_paths = model_paths
        self.stock_symbols = stock_symbols
        self.individual_models = {}
        self.ensemble_model = None
        
    def load_individual_models(self):
        """Load all individual stock models"""
        print("Loading individual models...")
        for symbol, path in zip(self.stock_symbols, self.model_paths):
            try:
                model = keras.models.load_model(path)
                self.individual_models[symbol] = model
                print(f"✓ Loaded {symbol} model from {path}")
            except Exception as e:
                print(f"✗ Failed to load {symbol} model: {e}")
        
        return len(self.individual_models) > 0
    
    def _create_model_wrapper(self, model, symbol, input_tensor):
        """Create a wrapper function that applies the model to input with unique scope"""
        def model_function(x):
            # Use tf.name_scope to create unique naming context
            with tf.name_scope(f"{symbol}_model"):
                return model(x)
        return model_function(input_tensor)
    
    # Method 1: Ensemble Model (Average Predictions) - COMPLETELY FIXED
    def create_ensemble_model(self, input_shape):
        """Create an ensemble model that averages predictions from all individual models"""
        inputs = keras.Input(shape=input_shape, name="ensemble_input")
        
        # Get predictions from all models using Lambda layers to avoid naming conflicts
        predictions = []
        for i, (symbol, model) in enumerate(self.individual_models.items()):
            # Use Lambda layer to wrap each model call with unique name
            pred = keras.layers.Lambda(
                lambda x, model=model: model(x),
                name=f"model_wrapper_{symbol}_{i}"
            )(inputs)
            predictions.append(pred)
        
        # Average all predictions
        if len(predictions) > 1:
            averaged_output = keras.layers.Average(name="ensemble_average")(predictions)
        else:
            averaged_output = predictions[0]
        
        # Create the ensemble model
        ensemble_model = keras.Model(inputs=inputs, outputs=averaged_output, name="ensemble_model")
        
        return ensemble_model
    
    # Method 2: Weighted Ensemble - FIXED
    def create_weighted_ensemble(self, input_shape, weights=None):
        """Create a weighted ensemble model"""
        if weights is None:
            weights = [1.0 / len(self.individual_models)] * len(self.individual_models)
        
        inputs = keras.Input(shape=input_shape, name="weighted_ensemble_input")
        
        # Get weighted predictions using Lambda layers
        weighted_predictions = []
        for i, (symbol, model) in enumerate(self.individual_models.items()):
            # Get prediction using Lambda wrapper
            pred = keras.layers.Lambda(
                lambda x, model=model: model(x),
                name=f"weighted_model_{symbol}_{i}"
            )(inputs)
            
            # Apply weight
            weighted_pred = keras.layers.Lambda(
                lambda x, w=weights[i]: x * w,
                name=f"weight_apply_{symbol}_{i}"
            )(pred)
            weighted_predictions.append(weighted_pred)
        
        # Sum weighted predictions
        final_output = keras.layers.Add(name="weighted_sum")(weighted_predictions)
        
        ensemble_model = keras.Model(inputs=inputs, outputs=final_output, name="weighted_ensemble_model")
        return ensemble_model
    
    # Method 3: Stacked Ensemble (Meta-learner) - FIXED
    def create_stacked_ensemble(self, input_shape, meta_model_units=64):
        """Create a stacked ensemble with a meta-learner"""
        inputs = keras.Input(shape=input_shape, name="stacked_ensemble_input")
        
        # Get predictions from all base models using Lambda layers
        base_predictions = []
        for i, (symbol, model) in enumerate(self.individual_models.items()):
            # Freeze the original model (optional for stacked ensemble)
            for layer in model.layers:
                layer.trainable = False
            
            # Get prediction using Lambda wrapper
            pred = keras.layers.Lambda(
                lambda x, model=model: model(x),
                name=f"stacked_model_{symbol}_{i}"
            )(inputs)
            base_predictions.append(pred)
        
        # Concatenate all base predictions
        concatenated = keras.layers.Concatenate(name="stacked_concat")(base_predictions)
        
        # Meta-learner (learns how to combine predictions)
        meta_hidden = keras.layers.Dense(meta_model_units, activation='relu', name="meta_hidden")(concatenated)
        meta_dropout = keras.layers.Dropout(0.2, name="meta_dropout")(meta_hidden)
        meta_output = keras.layers.Dense(1, activation='linear', name="meta_output")(meta_dropout)
        
        stacked_model = keras.Model(inputs=inputs, outputs=meta_output, name="stacked_ensemble_model")
        return stacked_model
    
    # Method 4: Multi-output Model
    def create_multi_output_model(self, input_shape):
        """Create a model that predicts all stocks simultaneously"""
        inputs = keras.Input(shape=input_shape, name="multi_output_input")
        
        # Shared layers
        shared_lstm1 = keras.layers.LSTM(128, return_sequences=True, name="shared_lstm1")(inputs)
        shared_dropout1 = keras.layers.Dropout(0.2, name="shared_dropout1")(shared_lstm1)
        shared_lstm2 = keras.layers.LSTM(64, return_sequences=False, name="shared_lstm2")(shared_dropout1)
        shared_dropout2 = keras.layers.Dropout(0.2, name="shared_dropout2")(shared_lstm2)
        
        # Individual output heads for each stock
        outputs = {}
        for symbol in self.stock_symbols:
            stock_dense = keras.layers.Dense(32, activation='relu', name=f'{symbol}_dense')(shared_dropout2)
            stock_output = keras.layers.Dense(1, activation='linear', name=f'{symbol}_output')(stock_dense)
            outputs[symbol] = stock_output
        
        multi_output_model = keras.Model(inputs=inputs, outputs=list(outputs.values()), name="multi_output_model")
        return multi_output_model
    
    # Method 5: Simple Prediction Ensemble (No model creation)
    def predict_simple_ensemble(self, X, method='average', weights=None):
        """
        Simple ensemble prediction without creating a new model.
        This is the most reliable method as it avoids naming conflicts entirely.
        """
        print(f"Making predictions with {len(self.individual_models)} models...")
        predictions = {}
        
        # Get predictions from each model
        for symbol, model in self.individual_models.items():
            try:
                pred = model.predict(X, verbose=0)
                predictions[symbol] = pred
                print(f"✓ Got predictions from {symbol} model")
            except Exception as e:
                print(f"✗ Error predicting with {symbol} model: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models were able to make predictions")
        
        # Combine predictions
        all_preds = np.array(list(predictions.values()))
        
        if method == 'average':
            ensemble_pred = np.mean(all_preds, axis=0)
        elif method == 'median':
            ensemble_pred = np.median(all_preds, axis=0)
        elif method == 'weighted':
            if weights is None:
                weights = [1.0 / len(predictions)] * len(predictions)
            ensemble_pred = np.average(all_preds, axis=0, weights=weights[:len(predictions)])
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_pred, predictions
    
    def predict_ensemble(self, X, method='average'):
        """Make predictions using ensemble of individual models (legacy method)"""
        return self.predict_simple_ensemble(X, method)

def main():
    # Example usage
    model_paths = [
        "AAPL_best_enhanced_model.keras",
        "GOOGL_best_enhanced_model.keras", 
        "MSFT_best_enhanced_model.keras",
        "NVDA_best_enhanced_model.keras",
        "INTC_best_enhanced_model.keras",
        "AMD_best_enhanced_model.keras"
    ]
    
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMD", "NVDA", "INTC"]
    
    # Initialize combiner
    combiner = MultiStockModelCombiner(model_paths, stock_symbols)
    
    # Load individual models
    if combiner.load_individual_models():
        print(f"Successfully loaded {len(combiner.individual_models)} models")
        
        # Assume input shape (adjust based on your models)
        input_shape = (120, 27)  # 120 timesteps, 27 features
        
        # Try the Lambda layer approach first
        try:
            print("\n=== Attempting to create ensemble models ===")
            
            # Method 1: Create ensemble model with Lambda layers
            print("\n1. Creating ensemble model...")
            ensemble_model = combiner.create_ensemble_model(input_shape)
            ensemble_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            ensemble_model.save("ensemble_stock_model.keras")
            print("✓ Ensemble model saved successfully!")
            
            # Method 2: Create weighted ensemble
            print("2. Creating weighted ensemble...")
            weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]  # Adjust weights as needed
            weighted_model = combiner.create_weighted_ensemble(input_shape, weights)
            weighted_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            weighted_model.save("weighted_ensemble_model.keras")
            print("✓ Weighted ensemble model saved successfully!")
            
            print("\nEnsemble models created successfully!")
            
        except Exception as e:
            print(f"Model creation failed: {e}")
            print("\n=== Using simple prediction ensemble instead ===")
        
        # Always try the simple ensemble prediction (most reliable)
        try:
            print("\n=== Testing Simple Ensemble Prediction ===")
            # Create some dummy test data
            dummy_data = np.random.random((5, 120, 27))  # 5 samples
            
            # Test different ensemble methods
            for method in ['average', 'median', 'weighted']:
                print(f"\nTesting {method} ensemble...")
                weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1] if method == 'weighted' else None
                
                ensemble_pred, individual_preds = combiner.predict_simple_ensemble(
                    dummy_data, method=method, weights=weights
                )
                
                print(f"✓ {method.capitalize()} ensemble prediction shape: {ensemble_pred.shape}")
                print(f"  Individual predictions from: {list(individual_preds.keys())}")
                
                # Show sample predictions
                print(f"  Sample {method} ensemble predictions: {ensemble_pred[:3].flatten()}")
            
        except Exception as e:
            print(f"Simple ensemble prediction failed: {e}")
        
        # Method 4: Multi-output model (creates new architecture)
        try:
            print("\n4. Creating multi-output model...")
            multi_model = combiner.create_multi_output_model(input_shape)
            multi_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            multi_model.save("multi_output_stock_model.keras")
            print("✓ Multi-output model saved successfully!")
            print("  Note: This model needs to be trained on multi-stock data")
            
        except Exception as e:
            print(f"Multi-output model creation failed: {e}")

if __name__ == "__main__":
    main()