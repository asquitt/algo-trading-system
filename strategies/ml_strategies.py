"""
ML-Based Trading Strategies
Uses machine learning models for price prediction and signal generation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, Optional
from loguru import logger
import joblib
import warnings
warnings.filterwarnings('ignore')


class MLStrategy:
    """
    Machine Learning Strategy Base Class
    Provides common functionality for ML-based trading strategies
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare ML features from market data
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with ML features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_momentum_5'] = df['close'].pct_change(5)
        features['price_momentum_10'] = df['close'].pct_change(10)
        features['price_momentum_20'] = df['close'].pct_change(20)
        
        # Volatility features
        features['volatility_5'] = df['returns'].rolling(5).std()
        features['volatility_10'] = df['returns'].rolling(10).std()
        features['volatility_20'] = df['returns'].rolling(20).std()
        
        # Volume features
        features['volume_change'] = df['volume'].pct_change()
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Technical indicators (if present in df)
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
            features['rsi_change'] = df['rsi'].diff()
            
        if 'macd' in df.columns:
            features['macd'] = df['macd']
            features['macd_signal' ] = df.get('macd_signal', 0)
            features['macd_hist'] = df.get('macd_hist', 0)
            
        if 'bb_upper' in df.columns:
            features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
        if 'adx' in df.columns:
            features['adx'] = df['adx']
            
        # Price position features
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Distance from moving averages
        if 'sma_20' in df.columns:
            features['price_to_sma20'] = df['close'] / df['sma_20'] - 1
            
        if 'sma_50' in df.columns:
            features['price_to_sma50'] = df['close'] / df['sma_50'] - 1
            
        if 'ema_12' in df.columns:
            features['price_to_ema12'] = df['close'] / df['ema_12'] - 1
            
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            
        # Drop NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.001) -> pd.Series:
        """
        Create labels for classification
        
        Args:
            df: DataFrame with price data
            horizon: Prediction horizon (bars ahead)
            threshold: Minimum return to be classified as buy/sell
            
        Returns:
            Series with labels: 1 (buy), -1 (sell), 0 (hold)
        """
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        
        labels = pd.Series(0, index=df.index)
        labels[future_returns > threshold] = 1  # Buy signal
        labels[future_returns < -threshold] = -1  # Sell signal
        
        return labels
    
    def train(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.001,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train the ML model
        
        Args:
            df: DataFrame with OHLCV and indicators
            horizon: Prediction horizon
            threshold: Minimum return threshold
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.__class__.__name__}...")
        
        # Prepare features and labels
        features = self.prepare_features(df)
        labels = self.create_labels(df, horizon, threshold)
        
        # Remove NaN labels (at the end due to forward shift)
        valid_idx = ~labels.isna()
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        # Train/test split (time-series aware)
        split_idx = int(len(features) * (1 - test_size))
        
        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = labels.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Get predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(self.feature_names),
            'buy_signals_train': (train_pred == 1).sum(),
            'sell_signals_train': (train_pred == -1).sum(),
            'buy_signals_test': (test_pred == 1).sum(),
            'sell_signals_test': (test_pred == -1).sum(),
        }
        
        logger.success(f"Training complete! Test accuracy: {test_score:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using trained model
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Series with predictions: 1 (buy), -1 (sell), 0 (hold)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
            
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled)
        
        return pd.Series(predictions, index=df.index)
    
    def save_model(self, filepath: str):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")
            
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scaler"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class RandomForestStrategy(MLStrategy):
    """
    Random Forest Classifier Strategy
    Ensemble of decision trees for robust predictions
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 20,
        random_state: int = 42
    ):
        """
        Initialize Random Forest strategy
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            random_state: Random seed
        """
        super().__init__()
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
            
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class GradientBoostingStrategy(MLStrategy):
    """
    Gradient Boosting Strategy
    Sequential ensemble that builds trees to correct previous errors
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42
    ):
        """
        Initialize Gradient Boosting strategy
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        super().__init__()
        
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )


class EnsembleStrategy:
    """
    Ensemble of multiple ML models
    Combines predictions from multiple models using voting
    """
    
    def __init__(self, models: list):
        """
        Initialize ensemble
        
        Args:
            models: List of trained MLStrategy instances
        """
        self.models = models
        
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate ensemble predictions using majority voting
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Series with ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(df)
            predictions.append(pred)
            
        # Stack predictions
        predictions = np.vstack(predictions)
        
        # Majority vote for each time step
        ensemble_pred = pd.Series(
            np.apply_along_axis(lambda x: np.bincount(x + 1).argmax() - 1, 0, predictions),
            index=df.index
        )
        
        return ensemble_pred
    
    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate predictions with confidence scores
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Tuple of (predictions, confidence scores)
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(df)
            predictions.append(pred)
            
        # Stack predictions
        predictions = np.vstack(predictions)
        
        # Calculate agreement (confidence)
        n_models = len(self.models)
        confidence = pd.Series(
            np.abs(predictions.sum(axis=0)) / n_models,
            index=df.index
        )
        
        # Majority vote
        ensemble_pred = pd.Series(
            np.apply_along_axis(lambda x: np.bincount(x + 1).argmax() - 1, 0, predictions),
            index=df.index
        )
        
        return ensemble_pred, confidence


# Example usage
if __name__ == "__main__":
    # This would normally come from your database
    # Here's a simple example
    
    from data_pipeline.features.technical_indicators import TechnicalIndicators
    from data_pipeline.storage.storage_manager import StorageManager
    
    # Initialize
    storage = StorageManager()
    indicators = TechnicalIndicators()
    
    # Fetch data
    df = storage.fetch_data('AAPL', days=365)
    
    # Add technical indicators
    df = indicators.add_all_indicators(df)
    
    # Train Random Forest
    rf_strategy = RandomForestStrategy(n_estimators=200, max_depth=8)
    metrics = rf_strategy.train(df, horizon=1, threshold=0.002)
    
    print("Training Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Get feature importance
    importance = rf_strategy.get_feature_importance()
    print("\nTop 10 Features:")
    print(importance.head(10))
    
    # Save model
    rf_strategy.save_model('models/rf_strategy.pkl')
    
    # Generate signals
    signals = rf_strategy.predict(df)
    print(f"\nBuy signals: {(signals == 1).sum()}")
    print(f"Sell signals: {(signals == -1).sum()}")
    print(f"Hold signals: {(signals == 0).sum()}")
