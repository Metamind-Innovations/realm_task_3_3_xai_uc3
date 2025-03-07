import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from common.data_loader import load_patient_data
from config import MAX_SEQUENCE_LENGTH
from lstm_utils import train_models_workflow, process_patient_data as process_lstm_patient_data
from xgboost_utils import train_models as train_xgboost_models, process_patient_data as process_xgboost_patient_data


def setup_arg_parser():
    """
    Set up command line argument parser

    :returns: Configured argument parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Glucose Prediction System')

    # Main operation mode
    parser.add_argument('--train', action='store_true', help='Train new models')
    parser.add_argument('--predict', action='store_true', help='Make predictions on patient data')

    # Model type selection
    parser.add_argument('--model-type', type=str, choices=['xgboost', 'lstm', 'ensemble'],
                        default='xgboost', help='Type of model to use (default: xgboost)')

    # File paths
    parser.add_argument('--training-file', type=str, help='CSV file with training data')
    parser.add_argument('--patient-file', type=str, help='CSV file with patient data (time and glucose_level columns)')

    # Model directories
    parser.add_argument('--xgboost-dir', type=str, default='xgboost_models', help='Directory for XGBoost model storage')
    parser.add_argument('--lstm-dir', type=str, default='lstm_models', help='Directory for LSTM model storage')

    # Output options
    parser.add_argument('--output-file', type=str, help='Output file for predictions (default is auto-generated)')
    parser.add_argument('--output-dir', type=str, default='predictions', help='Directory to save prediction outputs')
    parser.add_argument('--save-plots', action='store_true', help='Save prediction plots to files')

    # Advanced options
    parser.add_argument('--sequence-length', type=int, help='Sequence length for LSTM (default: adaptive)',
                        default=MAX_SEQUENCE_LENGTH)
    parser.add_argument('--compare-models', action='store_true', help='Compare predictions from multiple models')

    # XAI options
    parser.add_argument('--xai', action='store_true', help='Generate model explanations (requires XAI module)')
    parser.add_argument('--xai-dir', type=str, default='xai_reports', help='Directory to save XAI reports')
    parser.add_argument('--xai-global', action='store_true', help='Generate global model explanations')
    parser.add_argument('--xai-local', action='store_true', help='Generate local (per-prediction) explanations')

    return parser


def validate_args(args):
    """
    Validate command line arguments

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :raises ValueError: If arguments are invalid
    """
    if args.train and args.predict:
        raise ValueError("Cannot both train and predict at the same time. Please choose one operation.")

    if not args.train and not args.predict:
        raise ValueError("Must specify either --train or --predict")

    if args.train and not args.training_file:
        raise ValueError("Training file must be specified with --training-file when using --train")

    if args.predict and not args.patient_file:
        raise ValueError("Patient file must be specified with --patient-file when using --predict")

    if args.training_file and not os.path.exists(args.training_file):
        raise ValueError(f"Training file not found: {args.training_file}")

    if args.patient_file and not os.path.exists(args.patient_file):
        raise ValueError(f"Patient file not found: {args.patient_file}")

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def generate_output_filename(args):
    """
    Generate output filename based on model type and timestamp

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :returns: Output file path
    :rtype: str
    """
    if args.output_file:
        return args.output_file

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.model_type == 'xgboost':
        filename = f"xgboost_predictions_{timestamp}.csv"
    elif args.model_type == 'lstm':
        filename = f"lstm_predictions_{timestamp}.csv"
    elif args.model_type == 'ensemble':
        filename = f"ensemble_predictions_{timestamp}.csv"
    else:
        filename = f"predictions_{timestamp}.csv"

    return os.path.join(args.output_dir, filename)


def compare_model_predictions(patient_file, xgboost_dir='xgboost_models', lstm_dir='lstm_models',
                              output_dir='predictions', save_plots=False):
    """
    Compare predictions from both XGBoost and LSTM models

    :param patient_file: Path to patient data CSV
    :type patient_file: str
    :param xgboost_dir: Directory containing XGBoost models
    :type xgboost_dir: str
    :param lstm_dir: Directory containing LSTM models
    :type lstm_dir: str
    :param output_dir: Directory to save outputs
    :type output_dir: str
    :param save_plots: Whether to save plots to files
    :type save_plots: bool
    """
    print("Loading patient data...")
    patient_df = load_patient_data(patient_file)

    print("\nGenerating XGBoost predictions...")
    xgboost_output = os.path.join(output_dir, "xgboost_predictions_compare.csv")
    xgboost_predictions = process_xgboost_patient_data(
        patient_file,
        model_dir=xgboost_dir,
        output_file=xgboost_output
    )

    print("\nGenerating LSTM predictions...")
    lstm_output = os.path.join(output_dir, "lstm_predictions_compare.csv")
    lstm_predictions = process_lstm_patient_data(
        patient_file,
        model_dir=lstm_dir,
        output_file=lstm_output
    )

    if xgboost_predictions is None or lstm_predictions is None:
        print("Error: Could not generate predictions from one or both models")
        return

    print("\nPlotting comparison of model predictions...")
    plt.figure(figsize=(15, 10))

    col_mapping = patient_df.attrs.get('column_mapping', {})
    time_col = col_mapping.get('time', 'time')
    glucose_col = col_mapping.get('glucose_level', 'glucose_level')

    plt.plot(patient_df[time_col], patient_df[glucose_col], 'b-', label='Actual Glucose', linewidth=2)
    plt.scatter(patient_df[time_col], patient_df[glucose_col], color='blue', s=30, alpha=0.5)

    colors = {'xgboost': ['red', 'darkred', 'indianred'],
              'lstm': ['green', 'darkgreen', 'lightgreen']}

    for hours in [1, 2, 3]:
        # XGBoost predictions
        if f'predicted_{hours}hr' in xgboost_predictions.columns:
            plt.scatter(
                xgboost_predictions[f'prediction_time_{hours}hr'],
                xgboost_predictions[f'predicted_{hours}hr'],
                color=colors['xgboost'][hours - 1],
                label=f'XGBoost {hours}-hour',
                marker='o',
                s=50,
                alpha=0.7
            )

        # LSTM predictions
        if f'predicted_{hours}hr' in lstm_predictions.columns:
            plt.scatter(
                lstm_predictions[f'prediction_time_{hours}hr'],
                lstm_predictions[f'predicted_{hours}hr'],
                color=colors['lstm'][hours - 1],
                label=f'LSTM {hours}-hour',
                marker='x',
                s=50,
                alpha=0.7
            )

    plt.title('Comparison of XGBoost and LSTM Predictions', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Glucose Level (mg/dL)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plots:
        comparison_plot = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {comparison_plot}")

    plt.show()

    print("\nCreating combined predictions file...")

    # Merge the predictions from both models
    combined_df = pd.DataFrame()
    combined_df['timestamp'] = xgboost_predictions['timestamp']
    combined_df['current_glucose'] = xgboost_predictions['current_glucose']

    for hours in [1, 2, 3]:
        if (f'predicted_{hours}hr' in xgboost_predictions.columns and
                f'predicted_{hours}hr' in lstm_predictions.columns):

            time_col = f'prediction_time_{hours}hr'
            pred_times = xgboost_predictions[time_col]

            combined_df[f'xgboost_predicted_{hours}hr'] = xgboost_predictions[f'predicted_{hours}hr']

            lstm_times = lstm_predictions[time_col]
            lstm_values = lstm_predictions[f'predicted_{hours}hr']

            # Map LSTM predictions to XGBoost timestamps (simple matching by closest time)
            lstm_pred = []
            for t in pred_times:
                # Find closest matching time in LSTM predictions
                time_diffs = abs(lstm_times - t)
                closest_idx = time_diffs.idxmin()
                lstm_pred.append(lstm_values.iloc[closest_idx])

            combined_df[f'lstm_predicted_{hours}hr'] = lstm_pred

            # Calculate ensemble prediction (simple average)
            combined_df[f'ensemble_predicted_{hours}hr'] = (
                                                                   combined_df[f'xgboost_predicted_{hours}hr'] +
                                                                   combined_df[f'lstm_predicted_{hours}hr']
                                                           ) / 2

            combined_df[f'prediction_time_{hours}hr'] = pred_times

    combined_output = os.path.join(output_dir, "ensemble_predictions.csv")
    combined_df.to_csv(combined_output, index=False)
    print(f"Combined predictions saved to {combined_output}")

    try:
        latest_pred_time = None
        for hours in [3, 2, 1]:
            time_col = f'prediction_time_{hours}hr'
            if time_col in combined_df.columns:
                latest_pred_time = combined_df[time_col].max()
                break

        if latest_pred_time and patient_df[time_col].max() >= latest_pred_time:
            print("\nEvaluating prediction accuracy...")

            # Create evaluation summary
            eval_summary = []

            for hours in [1, 2, 3]:
                pred_col_xgb = f'xgboost_predicted_{hours}hr'
                pred_col_lstm = f'lstm_predicted_{hours}hr'
                pred_col_ensemble = f'ensemble_predicted_{hours}hr'

                if pred_col_xgb in combined_df.columns and pred_col_lstm in combined_df.columns:
                    actual_values = []

                    time_col = f'prediction_time_{hours}hr'
                    pred_times = combined_df[time_col]

                    for pred_time in pred_times:
                        # Find closest actual reading
                        time_diffs = abs(patient_df[time_col] - pred_time)
                        closest_idx = time_diffs.idxmin()
                        closest_time = patient_df.loc[closest_idx, time_col]

                        # Only use if within 10 minutes
                        if abs((closest_time - pred_time).total_seconds()) <= 600:
                            actual_values.append(patient_df.loc[closest_idx, glucose_col])
                        else:
                            actual_values.append(None)

                    # Filter valid predictions
                    valid_indices = [i for i, v in enumerate(actual_values) if v is not None]

                    if valid_indices:
                        valid_actual = [actual_values[i] for i in valid_indices]
                        valid_xgb = [combined_df[pred_col_xgb].iloc[i] for i in valid_indices]
                        valid_lstm = [combined_df[pred_col_lstm].iloc[i] for i in valid_indices]
                        valid_ensemble = [combined_df[pred_col_ensemble].iloc[i] for i in valid_indices]

                        import numpy as np
                        from sklearn.metrics import mean_squared_error

                        rmse_xgb = np.sqrt(mean_squared_error(valid_actual, valid_xgb))
                        rmse_lstm = np.sqrt(mean_squared_error(valid_actual, valid_lstm))
                        rmse_ensemble = np.sqrt(mean_squared_error(valid_actual, valid_ensemble))

                        eval_summary.append({
                            'Horizon': f'{hours}-hour',
                            'XGBoost RMSE': round(rmse_xgb, 2),
                            'LSTM RMSE': round(rmse_lstm, 2),
                            'Ensemble RMSE': round(rmse_ensemble, 2),
                            'Sample Size': len(valid_actual)
                        })

            if eval_summary:
                eval_df = pd.DataFrame(eval_summary)
                print("\nPrediction Accuracy Summary:")
                print(eval_df.to_string(index=False))

                eval_output = os.path.join(output_dir, "model_comparison_metrics.csv")
                eval_df.to_csv(eval_output, index=False)
                print(f"Evaluation metrics saved to {eval_output}")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")


def main():
    """Main execution function"""
    parser = setup_arg_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {str(e)}")
        parser.print_help()
        return 1

    if args.train:
        if args.model_type == 'xgboost' or args.model_type == 'ensemble':
            print(f"Starting XGBoost model training using data from {args.training_file}")
            xgboost_models, xgboost_quantiles = train_xgboost_models(
                args.training_file,
                model_dir=args.xgboost_dir
            )
            print("XGBoost training complete!")

        if args.model_type == 'lstm' or args.model_type == 'ensemble':
            print(f"Starting LSTM model training using data from {args.training_file}")
            lstm_models, lstm_scalers, lstm_results = train_models_workflow(
                args.training_file,
                model_dir=args.lstm_dir
            )
            print("LSTM training complete!")

    if args.predict:
        output_file = generate_output_filename(args)

        if args.compare_models:
            print("Comparing predictions from both XGBoost and LSTM models...")
            compare_model_predictions(
                args.patient_file,
                xgboost_dir=args.xgboost_dir,
                lstm_dir=args.lstm_dir,
                output_dir=args.output_dir,
                save_plots=args.save_plots
            )

        elif args.model_type == 'xgboost':
            print(f"Making XGBoost predictions for patient data in {args.patient_file}")
            predictions = process_xgboost_patient_data(
                args.patient_file,
                args.xgboost_dir,
                output_file,
                skip_plotting=True
            )

            if predictions is not None:
                print(f"XGBoost predictions saved to {output_file}")

                if args.save_plots:
                    patient_df = load_patient_data(args.patient_file)
                    plot_path = output_file.replace('.csv', '.png')

                    try:
                        from xgboost_utils.visualization import plot_patient_predictions as plot_xgboost_predictions
                        plot_xgboost_predictions(patient_df, predictions, args.output_dir, save_png=True)
                        print(f"Plot saved to {plot_path}")
                    except Exception as e:
                        print(f"Error plotting predictions: {str(e)}")

                print("XGBoost prediction complete!")
            else:
                print("XGBoost prediction failed. Please check the error messages above.")

        elif args.model_type == 'lstm':
            print(f"Making LSTM predictions for patient data in {args.patient_file}")
            predictions = process_lstm_patient_data(
                args.patient_file,
                args.lstm_dir,
                output_file,
                sequence_length=args.sequence_length,
                skip_plotting=True
            )

            if predictions is not None:
                print(f"LSTM predictions saved to {output_file}")

                if args.save_plots:
                    patient_df = load_patient_data(args.patient_file)
                    plot_path = output_file.replace('.csv', '.png')

                    try:
                        from lstm_utils.visualization import plot_predictions
                        plot_predictions(patient_df, predictions, args.output_dir, save_png=True)
                        print(f"Plot saved to {plot_path}")
                    except Exception as e:
                        print(f"Error plotting predictions: {str(e)}")

            else:
                print("LSTM prediction failed. Please check the error messages above.")

        elif args.model_type == 'ensemble':
            print(f"Making ensemble predictions for patient data in {args.patient_file}")
            compare_model_predictions(
                args.patient_file,
                xgboost_dir=args.xgboost_dir,
                lstm_dir=args.lstm_dir,
                output_dir=args.output_dir,
                save_plots=args.save_plots
            )
            print("Ensemble prediction complete!")

    if args.xai:
        try:
            # Try to import the XAI module
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from xai_utils import generate_explanation_report
            print("\nGenerating model explanations...")

            xai_output_dir = args.xai_dir if args.xai_dir else os.path.join(args.output_dir, 'xai_reports')

            # Determine global/local explanation flags
            include_global = args.xai_global if (args.xai_global or args.xai_local) else True
            include_local = args.xai_local if (args.xai_global or args.xai_local) else True

            if args.predict:
                if args.model_type == 'xgboost':
                    print("Generating XGBoost model explanations...")

                    # Load data and models
                    patient_df = load_patient_data(args.patient_file)

                    from xgboost_utils.model import load_models
                    models, quantile_models = load_models(args.xgboost_dir)

                    # Process feature data for explanation
                    from xgboost_utils.feature_engineering import create_prediction_features
                    features_df = create_prediction_features(patient_df)

                    # Generate explanation report
                    generate_explanation_report(
                        'xgboost',
                        models,
                        patient_data=patient_df,
                        features_data=features_df,
                        prediction_results=predictions,  # Using the predictions made earlier
                        output_dir=xai_output_dir,
                        include_global_explanation=include_global,
                        include_local_explanations=include_local
                    )

                    print("XGBoost explanations generated successfully!")

                elif args.model_type == 'lstm':
                    print("Generating LSTM model explanations...")
                    patient_df = load_patient_data(args.patient_file)

                    from lstm_utils.model import load_lstm_models
                    models, scalers = load_lstm_models(args.lstm_dir)

                    # Process sequences for LSTM explanation
                    from lstm_utils.data_processor import prepare_prediction_sequences
                    sequence_length = args.sequence_length
                    sequences, timestamps = prepare_prediction_sequences(patient_df, sequence_length)

                    # Generate explanation report
                    generate_explanation_report(
                        'lstm',
                        models,
                        patient_data=patient_df,
                        features_data=sequences,
                        prediction_results=predictions,  # Using the predictions made earlier
                        output_dir=xai_output_dir,
                        include_global_explanation=include_global,
                        include_local_explanations=include_local
                    )

                    print("LSTM explanations generated successfully!")

                elif args.model_type == 'ensemble':
                    print("Generating explanations for ensemble models...")

                    # Handle ensemble case by explaining both models
                    patient_df = load_patient_data(args.patient_file)

                    # XGBoost explanation
                    print("\n1. Generating XGBoost explanations...")
                    from xgboost_utils.model import load_models
                    from xgboost_utils.feature_engineering import create_prediction_features

                    xgboost_models, xgboost_quantiles = load_models(args.xgboost_dir)
                    features_df = create_prediction_features(patient_df)

                    generate_explanation_report(
                        'xgboost',
                        xgboost_models,
                        patient_data=patient_df,
                        features_data=features_df,
                        prediction_results=predictions,
                        output_dir=os.path.join(xai_output_dir, 'xgboost'),
                        include_global_explanation=include_global,
                        include_local_explanations=include_local
                    )

                    # LSTM explanation
                    print("\n2. Generating LSTM explanations...")
                    from lstm_utils.model import load_lstm_models
                    from lstm_utils.data_processor import prepare_prediction_sequences

                    lstm_models, lstm_scalers = load_lstm_models(args.lstm_dir)
                    sequence_length = args.sequence_length
                    sequences, timestamps = prepare_prediction_sequences(patient_df, sequence_length)

                    generate_explanation_report(
                        'lstm',
                        lstm_models,
                        patient_data=patient_df,
                        features_data=sequences,
                        prediction_results=predictions,
                        output_dir=os.path.join(xai_output_dir, 'lstm'),
                        include_global_explanation=include_global,
                        include_local_explanations=include_local
                    )

                    print("\nEnsemble model explanations generated successfully!")
            else:
                print("XAI functionality requires model predictions. Please run with --predict flag")

        except ImportError as e:
            print(f"Error importing XAI utilities: {str(e)}")
            print("Please make sure you have the required packages installed:")
            print("  - shap")
            print("  - matplotlib")
            print("  - pandas")
            print("  - tensorflow (for LSTM models)")
            print("\nInstall with: pip install shap matplotlib pandas tensorflow")

    return 0


if __name__ == "__main__":
    exit(main())
