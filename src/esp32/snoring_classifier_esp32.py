"""
ESP32 Optimized Snoring Classifier
Python implementation for MicroPython on ESP32
"""

import math
import array
from typing import List, Tuple


class ESP32SnoringClassifier:
    """
    ESP32-optimized snoring classifier
    """
    
    def __init__(self):
        """Initialize classifier"""
        # Feature extraction parameters
        self.window_size = 80  # 8 seconds * 10 measurements per second
        self.audio_channels = ['b100', 'b400', 'b1000', 'env']
        self.accel_axes = ['x', 'y', 'z']
        
        # Model parameters (will be loaded from trained model)
        self.feature_importances = None
        self.tree_structure = None
        
    def extract_audio_features(self, audio_data: array.array) -> List[float]:
        """
        Extract 7 features for one audio channel (ESP32 optimized)
        
        Args:
            audio_data: Array of audio values
            
        Returns:
            List of 7 features
        """
        if len(audio_data) == 0:
            return [0.0] * 7
        
        # 1. Средний уровень сигнала
        mean_level = sum(audio_data) / len(audio_data)
        
        # 2. Максимальный уровень
        max_level = max(audio_data)
        
        # 3. Изменчивость (стандартное отклонение)
        variance = sum((x - mean_level) ** 2 for x in audio_data) / len(audio_data)
        std_dev = math.sqrt(variance)
        
        # 4. Относительная изменчивость
        relative_std = std_dev / (mean_level + 1e-8)
        
        # 5. Доля высоких значений (выше 75-го перцентиля)
        sorted_data = sorted(audio_data)
        threshold_idx = int(0.75 * len(sorted_data))
        threshold = sorted_data[threshold_idx]
        high_threshold_ratio = sum(1 for x in audio_data if x > threshold) / len(audio_data)
        
        # 6. Тенденция изменения (упрощенный тренд)
        if len(audio_data) > 1:
            trend = (audio_data[-1] - audio_data[0]) / len(audio_data)
        else:
            trend = 0.0
        
        # 7. Регулярность (упрощенная автокорреляция)
        if len(audio_data) > 1:
            # Простая мера регулярности
            diffs = [abs(audio_data[i] - audio_data[i-1]) for i in range(1, len(audio_data))]
            regularity = 1.0 / (1.0 + sum(diffs) / len(diffs))
        else:
            regularity = 0.0
        
        return [
            float(mean_level),
            float(max_level),
            float(std_dev),
            float(relative_std),
            float(high_threshold_ratio),
            float(trend),
            float(regularity)
        ]
    
    def extract_accelerometer_features(self, accel_data: array.array) -> List[float]:
        """
        Extract 3 features for one accelerometer axis (ESP32 optimized)
        
        Args:
            accel_data: Array of accelerometer values
            
        Returns:
            List of 3 features
        """
        if len(accel_data) < 2:
            return [0.0, 0.0, 0.0]
        
        # Вычисляем дельты
        deltas = [abs(accel_data[i] - accel_data[i-1]) for i in range(1, len(accel_data))]
        
        # 1. Общая активность
        mean_activity = sum(deltas) / len(deltas)
        
        # 2. Максимальная активность
        max_activity = max(deltas)
        
        # 3. Доля времени с движением
        movement_threshold = sorted(deltas)[len(deltas)//2]  # медиана
        movement_ratio = sum(1 for d in deltas if d > movement_threshold) / len(deltas)
        
        return [
            float(mean_activity),
            float(max_activity),
            float(movement_ratio)
        ]
    
    def extract_mixed_features(self, audio_features: List[List[float]]) -> List[float]:
        """
        Extract 2 mixed features (ESP32 optimized)
        
        Args:
            audio_features: List of audio features for each channel
            
        Returns:
            List of 2 mixed features
        """
        # 1. b400/b100 ratio
        b400_mean = audio_features[1][0]  # mean of b400 channel
        b100_mean = audio_features[0][0]  # mean of b100 channel
        b400_b100_ratio = b400_mean / (b100_mean + 1e-8)
        
        # 2. b400/b1000 ratio
        b1000_mean = audio_features[2][0]  # mean of b1000 channel
        b400_b1000_ratio = b400_mean / (b1000_mean + 1e-8)
        
        return [
            float(b400_b100_ratio),
            float(b400_b1000_ratio)
        ]
    
    def extract_all_features(self, window_data: dict) -> List[float]:
        """
        Extract all 39 features from one window (ESP32 optimized)
        
        Args:
            window_data: Dictionary with sensor data for one window
            
        Returns:
            List of 39 features
        """
        features = []
        
        # Extract audio features (28 features)
        audio_features = []
        for channel in self.audio_channels:
            if channel in window_data:
                channel_data = array.array('f', window_data[channel])
                channel_features = self.extract_audio_features(channel_data)
                features.extend(channel_features)
                audio_features.append(channel_features)
            else:
                features.extend([0.0] * 7)
                audio_features.append([0.0] * 7)
        
        # Extract accelerometer features (9 features)
        for axis in self.accel_axes:
            if axis in window_data:
                axis_data = array.array('f', window_data[axis])
                axis_features = self.extract_accelerometer_features(axis_data)
                features.extend(axis_features)
            else:
                features.extend([0.0] * 3)
        
        # Extract mixed features (2 features)
        mixed_features = self.extract_mixed_features(audio_features)
        features.extend(mixed_features)
        
        return features
    
    def predict(self, features: List[float]) -> Tuple[str, float]:
        """
        Make prediction using loaded model
        
        Args:
            features: List of 39 features
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.tree_structure is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Simple decision tree traversal (placeholder)
        # In real implementation, this would use the actual tree structure
        
        # For now, return a simple heuristic
        # This should be replaced with actual model inference
        
        # Simple heuristic based on audio features
        b100_mean = features[0]
        b400_mean = features[7]
        b1000_mean = features[14]
        
        # If b400 (snore range) is significantly higher than others, likely snoring
        if b400_mean > (b100_mean + b1000_mean) * 0.6:
            prediction = 'W'  # Snoring
            confidence = 0.8
        else:
            prediction = ''   # No snoring
            confidence = 0.7
        
        return prediction, confidence
    
    def load_model(self, model_data: dict):
        """
        Load trained model data
        
        Args:
            model_data: Dictionary with model parameters
        """
        self.feature_importances = model_data.get('feature_importances', [])
        self.tree_structure = model_data.get('tree_structure', {})
        print("Model loaded successfully")
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model info
        """
        return {
            'feature_count': 39,
            'window_size': self.window_size,
            'audio_channels': self.audio_channels,
            'accel_axes': self.accel_axes,
            'model_loaded': self.tree_structure is not None
        }


# Example usage for ESP32
def example_usage():
    """Example of how to use the classifier on ESP32"""
    
    # Initialize classifier
    classifier = ESP32SnoringClassifier()
    
    # Load model (in real usage, load from file)
    model_data = {
        'feature_importances': [0.1] * 39,
        'tree_structure': {}
    }
    classifier.load_model(model_data)
    
    # Example window data
    window_data = {
        'b100': [100, 110, 105, 115, 120] * 16,  # 80 values
        'b400': [200, 210, 205, 215, 220] * 16,
        'b1000': [300, 310, 305, 315, 320] * 16,
        'env': [50, 55, 52, 58, 60] * 16,
        'x': [0.1, 0.2, 0.15, 0.25, 0.3] * 16,
        'y': [0.05, 0.1, 0.08, 0.12, 0.15] * 16,
        'z': [0.2, 0.25, 0.22, 0.28, 0.32] * 16
    }
    
    # Extract features
    features = classifier.extract_all_features(window_data)
    print(f"Extracted {len(features)} features")
    
    # Make prediction
    prediction, confidence = classifier.predict(features)
    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
    
    # Get model info
    info = classifier.get_model_info()
    print(f"Model info: {info}")


if __name__ == "__main__":
    example_usage() 