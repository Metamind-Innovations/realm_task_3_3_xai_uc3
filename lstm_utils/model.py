from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam


def create_lstm_model(sequence_length, n_features):
    """
    Create a bidirectional LSTM model for glucose prediction.

    Architecture includes:
    - Two bidirectional LSTM layers with batch normalization and dropout
    - Dense output layers for final prediction

    :param sequence_length: Number of time points in input sequences
    :type sequence_length: int
    :param n_features: Number of features per time point
    :type n_features: int
    :returns: Compiled LSTM model
    :rtype: tensorflow.keras.models.Sequential
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(sequence_length, n_features)),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


def load_lstm_models(model_dir='lstm_models'):
    """
    Load trained LSTM models and scalers from disk.

    Handles multiple file formats (.keras, .h5) and implements fallback
    strategies if the initial loading approach fails.

    :param model_dir: Directory containing saved models and scalers
    :type model_dir: str
    :returns: Loaded models and scalers
    :rtype: tuple(dict, dict)
    """
    import os
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    import pickle

    models = {}

    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Please train models first.")
        return None, None

    scaler_path = os.path.join(model_dir, 'scalers.pkl')
    if not os.path.exists(scaler_path):
        print(f"Scalers file not found at {scaler_path}")
        return None, None

    try:
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
    except Exception as e:
        print(f"Error loading scalers: {str(e)}")
        return None, None

    found_models = False
    PREDICTION_HORIZONS = [1, 2, 3]  # Hours into the future

    for hours in PREDICTION_HORIZONS:
        model_path_keras = os.path.join(model_dir, f'lstm_model_{hours}hr.keras')
        model_path_h5 = os.path.join(model_dir, f'lstm_model_{hours}hr.h5')

        model_path = None
        if os.path.exists(model_path_keras):
            model_path = model_path_keras
        elif os.path.exists(model_path_h5):
            model_path = model_path_h5

        if model_path:
            try:
                custom_objects = {
                    'Adam': Adam
                }

                try:
                    models[hours] = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    print(f"Loaded {hours}-hour prediction model from {model_path}")
                    found_models = True
                except Exception as e1:
                    print(f"First load attempt failed: {str(e1)}")
                    try:
                        models[hours] = tf.keras.models.load_model(model_path, compile=False)
                        models[hours].compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss='mean_squared_error',
                            metrics=['mae']
                        )
                        print(f"Loaded and recompiled {hours}-hour prediction model from {model_path}")
                        found_models = True
                    except Exception as e2:
                        print(f"Second load attempt failed: {str(e2)}")
                        try:
                            from lstm_utils.model import create_lstm_model
                            MAX_SEQUENCE_LENGTH = 10
                            FEATURES_TO_INCLUDE = [
                                'glucose_level', 'glucose_std', 'glucose_rate', 'glucose_mean',
                                'glucose_min', 'glucose_max', 'glucose_range', 'glucose_acceleration',
                                'hour_of_day', 'is_daytime', 'day_of_week', 'is_weekend'
                            ]
                            temp_model = create_lstm_model(MAX_SEQUENCE_LENGTH, len(FEATURES_TO_INCLUDE))
                            temp_model.load_weights(model_path)
                            models[hours] = temp_model
                            print(f"Loaded weights only for {hours}-hour prediction model")
                            found_models = True
                        except Exception as e3:
                            print(f"All loading attempts failed for {hours}-hour model: {str(e3)}")
            except Exception as e:
                print(f"Error loading model for {hours}-hour prediction: {str(e)}")
        else:
            print(f"Model file for {hours}-hour prediction not found")

    if not found_models:
        print("No models could be loaded successfully")
        return None, None

    return models, scalers
