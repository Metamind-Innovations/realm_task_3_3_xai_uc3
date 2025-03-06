"""
Centralized configuration file for glucose prediction system.
Contains constants, settings, and default parameters used throughout the codebase.
"""

# LSTM model constants
MAX_SEQUENCE_LENGTH = 10
PREDICTION_HORIZONS = [1, 2, 3]  # Hours into the future
FEATURES_TO_INCLUDE = [
    'glucose_level', 'glucose_std', 'glucose_rate', 'glucose_mean',
    'glucose_min', 'glucose_max', 'glucose_range', 'glucose_acceleration',
    'hour_of_day', 'is_daytime', 'day_of_week', 'is_weekend'
]

# File paths and directories
DEFAULT_LSTM_MODEL_DIR = 'lstm_models'
DEFAULT_XGBOOST_MODEL_DIR = 'xgboost_models'
DEFAULT_OUTPUT_DIR = 'predictions'
DEFAULT_XAI_DIR = 'xai_reports'

# Model parameters
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 100

# Evaluation settings
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

# Visualization settings
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 8
DPI = 300