import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm

from common.data_loader import load_and_preprocess_training_data, get_column_name
from config import (MAX_SEQUENCE_LENGTH, PREDICTION_HORIZONS, FEATURES_TO_INCLUDE,
                    LSTM_BATCH_SIZE, LSTM_EPOCHS, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE)
from lstm_utils.data_processor import extract_patient_features
from lstm_utils.model import create_lstm_model


def prepare_lstm_dataset(patient_dfs, sequence_length=MAX_SEQUENCE_LENGTH):
    """
    Prepare sequences for LSTM model from patient dataframes.

    Creates sliding window sequences of features and matching target values
    for each prediction horizon.

    :param patient_dfs: List of patient DataFrames with features
    :type patient_dfs: list[pandas.DataFrame]
    :param sequence_length: Number of time points in each sequence
    :type sequence_length: int
    :returns: X sequences and y target values for each prediction horizon
    :rtype: tuple(numpy.ndarray, dict)
    """
    print(f"Preparing LSTM dataset with sequence length {sequence_length}...")

    feature_cols = FEATURES_TO_INCLUDE

    X_sequences = []
    y_dict = {hours: [] for hours in PREDICTION_HORIZONS}

    for patient_df in tqdm(patient_dfs, desc="Preparing patient sequences"):
        if 'glucose_level' not in patient_df.columns:
            print("Warning: glucose_level column not found in dataframe, skipping patient")
            continue

        features = []
        for col in feature_cols:
            if col in patient_df.columns:
                features.append(patient_df[col].values)
            else:
                print(f"Feature {col} not available, using zeros")
                features.append(np.zeros(len(patient_df)))

        if not features:
            print("No valid features found for patient, skipping")
            continue

        feature_array = np.column_stack(features)

        for i in range(len(feature_array) - sequence_length):
            sequence = feature_array[i:i + sequence_length]
            X_sequences.append(sequence)

            for hours in PREDICTION_HORIZONS:
                target_col = f'future_glucose_{hours}hr'
                if target_col in patient_df.columns:
                    target_value = patient_df[target_col].iloc[i + sequence_length - 1]
                    y_dict[hours].append(target_value)
                else:
                    y_dict[hours].append(patient_df['glucose_level'].iloc[i + sequence_length - 1])

    X = np.array(X_sequences)
    y = {hours: np.array(values) for hours, values in y_dict.items()}

    print(f"Created dataset with {len(X)} sequences")
    for hours, values in y.items():
        print(f"  {hours}-hour targets: {len(values)} values")

    return X, y


def train_lstm_models(X, y, model_dir='lstm_models',
                      test_size=TEST_SIZE, validation_size=VALIDATION_SIZE, random_state=RANDOM_STATE):
    """
    Train LSTM models for different prediction horizons.

    Handles data scaling, model training with callbacks for early stopping
    and learning rate reduction, and saves models and evaluation metrics.

    :param X: Input sequences
    :type X: numpy.ndarray
    :param y: Target values for each prediction horizon
    :type y: dict
    :param model_dir: Directory to save models and results
    :type model_dir: str
    :param test_size: Fraction of data to use for testing
    :type test_size: float
    :param validation_size: Fraction of training data to use for validation
    :type validation_size: float
    :param random_state: Random seed for reproducibility
    :type random_state: int
    :returns: Trained models, scalers, and evaluation results
    :rtype: tuple(dict, dict, dict)
    """
    os.makedirs(model_dir, exist_ok=True)
    models = {}
    results = {}

    scalers = {
        'X': MinMaxScaler(),
        'y': {hours: MinMaxScaler() for hours in PREDICTION_HORIZONS}
    }

    X_reshaped = X.reshape(-1, X.shape[2])
    X_scaled_2d = scalers['X'].fit_transform(X_reshaped)
    X_scaled = X_scaled_2d.reshape(X.shape)

    for hours in PREDICTION_HORIZONS:
        print(f"\nTraining {hours}-hour prediction model")

        y_current = y[hours].reshape(-1, 1)
        y_scaled = scalers['y'][hours].fit_transform(y_current)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state)

        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adjusted, random_state=random_state)

        print(f"Training set: {X_train.shape[0]} sequences")
        print(f"Validation set: {X_val.shape[0]} sequences")
        print(f"Test set: {X_test.shape[0]} sequences")

        model = create_lstm_model(X.shape[1], X.shape[2])
        model.summary()

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, f'lstm_{hours}hr_checkpoint.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        print(f"Training {hours}-hour model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        test_results = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test loss (MSE): {test_results[0]:.4f}")
        print(f"Test MAE: {test_results[1]:.4f}")

        y_pred_scaled = model.predict(X_test)
        y_pred = scalers['y'][hours].inverse_transform(y_pred_scaled)
        y_test_orig = scalers['y'][hours].inverse_transform(y_test)

        mse = mean_squared_error(y_test_orig, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)

        print(f"RMSE (original scale): {rmse:.2f}")
        print(f"MAE (original scale): {mae:.2f}")
        print(f"R² score: {r2:.3f}")

        results[hours] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'history': history.history
        }

        model_path = os.path.join(model_dir, f'lstm_model_{hours}hr.keras')
        model.save(model_path)
        print(f"Model saved to {model_path}")

        models[hours] = model

    with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

    with open(os.path.join(model_dir, 'training_results.json'), 'w') as f:
        serializable_results = {}
        for hours, res in results.items():
            serializable_results[str(hours)] = {
                'mse': float(res['mse']),
                'rmse': float(res['rmse']),
                'mae': float(res['mae']),
                'r2': float(res['r2']),
                'history': {k: [float(x) for x in v] for k, v in res['history'].items()}
            }
        json.dump(serializable_results, f, indent=2)

    return models, scalers, results


def plot_training_history(results, model_dir='lstm_models'):
    """
    Plot training history for each model.

    Creates a two-panel figure showing loss and MAE during training
    for each prediction horizon model.

    :param results: Training results for each model
    :type results: dict
    :param model_dir: Directory to save the plot
    :type model_dir: str
    """
    plt.figure(figsize=(16, 8))

    for i, hours in enumerate(results.keys()):
        history = results[hours]['history']

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label=f'{hours}-hour train')
        plt.plot(history['val_loss'], label=f'{hours}-hour validation')

        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label=f'{hours}-hour train')
        plt.plot(history['val_mae'], label=f'{hours}-hour validation')

    plt.subplot(1, 2, 1)
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'), dpi=300)
    plt.show()


def compare_prediction_horizons(results, model_dir='lstm_models'):
    """
    Compare metrics across prediction horizons.

    Creates a three-panel figure comparing RMSE, MAE, and R² scores
    across different prediction horizons.

    :param results: Evaluation results for each prediction horizon
    :type results: dict
    :param model_dir: Directory to save the plot
    :type model_dir: str
    """
    hours = list(results.keys())
    rmse_values = [results[h]['rmse'] for h in hours]
    mae_values = [results[h]['mae'] for h in hours]
    r2_values = [results[h]['r2'] for h in hours]

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.bar(hours, rmse_values)
    plt.title('RMSE by Prediction Horizon')
    plt.xlabel('Hours')
    plt.ylabel('RMSE (mg/dL)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.bar(hours, mae_values)
    plt.title('MAE by Prediction Horizon')
    plt.xlabel('Hours')
    plt.ylabel('MAE (mg/dL)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.bar(hours, r2_values)
    plt.title('R² Score by Prediction Horizon')
    plt.xlabel('Hours')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'prediction_horizons.png'), dpi=300)
    plt.show()


def train_models_workflow(training_file, model_dir='lstm_models'):
    """
    Workflow for training LSTM models.

    Loads training data, extracts features for each patient,
    prepares sequences, trains models, and evaluates results.

    :param training_file: Path to CSV file with training data
    :type training_file: str
    :param model_dir: Directory to save trained models
    :type model_dir: str
    :returns: Trained models, scalers, and evaluation results
    :rtype: tuple(dict, dict, dict)
    """
    print(f"Loading training data from {training_file}...")
    df = load_and_preprocess_training_data(training_file)

    subject_id_col = get_column_name(df, 'SUBJECT_ID')
    if subject_id_col not in df.columns:
        subject_id_col = 'subject_id'
        if subject_id_col not in df.columns:
            print(f"Error: Could not find subject ID column. Available columns: {df.columns.tolist()}")
            return None, None, None

    print(f"Processing {df[subject_id_col].nunique()} patients...")
    patient_dfs = []

    subject_ids = df[subject_id_col].unique()
    for subject_id in tqdm(subject_ids, desc="Extracting patient features"):
        patient_data = df[df[subject_id_col] == subject_id]
        patient_features = extract_patient_features(patient_data, is_training=True)
        if patient_features is not None and len(patient_features) >= MAX_SEQUENCE_LENGTH:
            patient_dfs.append(patient_features)

    print(f"Prepared data for {len(patient_dfs)} patients")

    X, y = prepare_lstm_dataset(patient_dfs, sequence_length=MAX_SEQUENCE_LENGTH)

    models, scalers, results = train_lstm_models(X, y, model_dir=model_dir)

    plot_training_history(results, model_dir)
    compare_prediction_horizons(results, model_dir)

    print(f"Models saved to {model_dir} directory")
    return models, scalers, results
