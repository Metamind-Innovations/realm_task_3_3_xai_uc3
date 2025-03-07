import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc

from xgboost_utils.model import load_models
from common.data_loader import load_patient_data
from xgboost_utils.feature_engineering import create_prediction_features


def calculate_demographic_parity(predictions, demographics, threshold=None):
    if threshold is not None:
        binary_preds = predictions >= threshold
    else:
        binary_preds = predictions

    group_predictions = {}
    group_counts = {}
    for group in demographics.unique():
        mask = demographics == group
        group_predictions[group] = np.mean(binary_preds[mask])
        group_counts[group] = sum(mask)

    disparities = {}
    groups = list(group_predictions.keys())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            disparity = abs(group_predictions[g1] - group_predictions[g2])
            disparities[(g1, g2)] = disparity

    max_disparity = max(disparities.values()) if disparities else 0
    avg_rate = np.mean(list(group_predictions.values()))

    return {
        'group_predictions': group_predictions,
        'group_counts': group_counts,
        'disparities': disparities,
        'max_disparity': max_disparity,
        'average_rate': avg_rate
    }


def calculate_equalized_odds(predictions, true_values, demographics, threshold=None):
    if threshold is not None:
        binary_preds = predictions >= threshold
        binary_true = true_values >= threshold
    else:
        binary_preds = predictions
        binary_true = true_values

    group_metrics = {}
    group_counts = {}
    for group in demographics.unique():
        mask = demographics == group
        group_preds = binary_preds[mask]
        group_true = binary_true[mask]

        # Skip if not enough data or lack of class diversity
        if len(group_true) < 5 or len(np.unique(group_true)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(group_true, group_preds, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate / Sensitivity
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate / Specificity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value / Precision

        group_metrics[group] = {
            'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr, 'ppv': ppv,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        group_counts[group] = sum(mask)

    if len(group_metrics) < 2:
        return {
            'group_metrics': group_metrics,
            'group_counts': group_counts,
            'fpr_disparities': {},
            'fnr_disparities': {},
            'max_fpr_disparity': 0,
            'max_fnr_disparity': 0,
            'error': "Not enough groups with valid metrics"
        }

    fpr_disparities = {}
    fnr_disparities = {}
    tpr_disparities = {}
    groups = list(group_metrics.keys())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            fpr_disparity = abs(group_metrics[g1]['fpr'] - group_metrics[g2]['fpr'])
            fnr_disparity = abs(group_metrics[g1]['fnr'] - group_metrics[g2]['fnr'])
            tpr_disparity = abs(group_metrics[g1]['tpr'] - group_metrics[g2]['tpr'])
            fpr_disparities[(g1, g2)] = fpr_disparity
            fnr_disparities[(g1, g2)] = fnr_disparity
            tpr_disparities[(g1, g2)] = tpr_disparity

    max_fpr_disparity = max(fpr_disparities.values()) if fpr_disparities else 0
    max_fnr_disparity = max(fnr_disparities.values()) if fnr_disparities else 0
    max_tpr_disparity = max(tpr_disparities.values()) if tpr_disparities else 0

    return {
        'group_metrics': group_metrics,
        'group_counts': group_counts,
        'fpr_disparities': fpr_disparities,
        'fnr_disparities': fnr_disparities,
        'tpr_disparities': tpr_disparities,
        'max_fpr_disparity': max_fpr_disparity,
        'max_fnr_disparity': max_fnr_disparity,
        'max_tpr_disparity': max_tpr_disparity
    }


def calculate_roc_auc_by_group(predictions, true_values, demographics):
    group_auc = {}
    group_counts = {}

    for group in demographics.unique():
        mask = demographics == group
        group_pred = predictions[mask]
        group_true = true_values[mask]

        if len(np.unique(group_true)) < 2 or sum(mask) < 5:
            group_auc[group] = None
            group_counts[group] = sum(mask)
            continue

        try:
            fpr, tpr, _ = roc_curve(group_true, group_pred)
            roc_auc = auc(fpr, tpr)
            group_auc[group] = roc_auc
            group_counts[group] = sum(mask)
        except Exception as e:
            print(f"Error calculating ROC AUC for group {group}: {str(e)}")
            group_auc[group] = None
            group_counts[group] = sum(mask)

    # Calculate disparities
    auc_disparities = {}
    valid_groups = [g for g in group_auc.keys() if group_auc[g] is not None]
    for i in range(len(valid_groups)):
        for j in range(i + 1, len(valid_groups)):
            g1, g2 = valid_groups[i], valid_groups[j]
            disparity = abs(group_auc[g1] - group_auc[g2])
            auc_disparities[(g1, g2)] = disparity

    max_disparity = max(auc_disparities.values()) if auc_disparities else 0

    return {
        'group_auc': group_auc,
        'group_counts': group_counts,
        'auc_disparities': auc_disparities,
        'max_auc_disparity': max_disparity
    }


def calculate_threshold_metrics(predictions, true_values, demographics, thresholds):
    threshold_results = {}

    for threshold in thresholds:
        dp_results = calculate_demographic_parity(
            predictions, demographics, threshold)

        eo_results = calculate_equalized_odds(
            predictions, true_values, demographics, threshold)

        threshold_results[threshold] = {
            'demographic_parity': dp_results,
            'equalized_odds': eo_results
        }

    return threshold_results


def analyze_feature_importance_bias(models, feature_names, demographic_features):
    bias_results = {}

    for hours, model in models.items():
        if not hasattr(model, 'feature_importances_'):
            continue

        importance_values = model.feature_importances_
        importances = dict(zip(feature_names, importance_values))

        bias_results[hours] = {
            'all_importances': importances,
            'demographic_importances': {f: importances.get(f, 0) for f in demographic_features}
        }

    return bias_results


def get_intersectional_data(demographic_data, feature1, feature2):
    intersectional_groups = {}
    predictions_by_group = {}
    true_values_by_group = {}

    for data in demographic_data:
        if data[feature1] is None or data[feature2] is None:
            continue

        combined_group = f"{data[feature1]}_{data[feature2]}"
        if combined_group not in intersectional_groups:
            intersectional_groups[combined_group] = []
            predictions_by_group[combined_group] = []
            true_values_by_group[combined_group] = []

        intersectional_groups[combined_group].append(data)
        predictions_by_group[combined_group].extend(data['predictions'])

        valid_indices = [i for i, v in enumerate(data['true_values']) if v is not None]
        if valid_indices:
            true_vals = [data['true_values'][i] for i in valid_indices]
            true_values_by_group[combined_group].extend(true_vals)

    return intersectional_groups, predictions_by_group, true_values_by_group


def suggest_mitigation_strategies(results, output_dir=None):
    mitigation_suggestions = {}

    for key, metrics in results.items():
        parts = key.split('_')
        feature = parts[0]

        dp_disparity = metrics['demographic_parity']['max_disparity']
        eo_results = metrics['equalized_odds']

        suggestions = []

        # Demographic parity strategies
        if dp_disparity > 0.2:
            suggestions.append({
                'issue': f"High demographic parity disparity ({dp_disparity:.2%}) for {feature} groups",
                'strategies': [
                    "Apply post-processing techniques to equalize prediction rates across groups",
                    "Reweight the dataset to balance representation of different groups",
                    "Consider adjusting decision thresholds separately for each group"
                ]
            })
        elif dp_disparity > 0.1:
            suggestions.append({
                'issue': f"Moderate demographic parity disparity ({dp_disparity:.2%}) for {feature} groups",
                'strategies': [
                    "Monitor model performance across groups during development",
                    "Consider collecting more diverse training data",
                    "Validate results with domain experts in healthcare"
                ]
            })

        # Equalized odds strategies
        if 'error' not in eo_results and 'max_fpr_disparity' in eo_results:
            max_error_disparity = max(eo_results['max_fpr_disparity'], eo_results['max_fnr_disparity'])

            if max_error_disparity > 0.2:
                suggestions.append({
                    'issue': f"High error rate disparity ({max_error_disparity:.2%}) for {feature} groups",
                    'strategies': [
                        "Implement adversarial debiasing techniques to minimize disparate error rates",
                        "Collect additional data for underrepresented or higher-error groups",
                        "Consider ensemble methods that combine multiple models optimized for different groups"
                    ]
                })
            elif max_error_disparity > 0.1:
                suggestions.append({
                    'issue': f"Moderate error rate disparity ({max_error_disparity:.2%}) for {feature} groups",
                    'strategies': [
                        "Apply regularization techniques that penalize disparate error rates",
                        "Review feature engineering process for potential sources of bias",
                        "Evaluate performance on specific subgroups within the dataset"
                    ]
                })

        mitigation_suggestions[key] = suggestions

    if output_dir:
        with open(os.path.join(output_dir, 'mitigation_strategies.json'), 'w') as f:
            json.dump(mitigation_suggestions, f, indent=2)

    return mitigation_suggestions


def analyze_model_fairness_bias(model_dir, patient_files, output_dir=None, threshold=150):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models, _ = load_models(model_dir)

    if not models:
        print(f"Error: No models found in {model_dir}")
        return None

    demographic_data = []
    feature_names = None

    for patient_file in patient_files:
        try:
            patient_df = load_patient_data(patient_file)

            filename_demographics = patient_df.attrs.get('demographics_from_filename', {})
            gender = filename_demographics.get('gender')
            age = filename_demographics.get('age')

            if not gender and not age:
                print(f"Warning: No demographic information found in {patient_file}")
                continue

            features_df = create_prediction_features(patient_df)

            if len(features_df) == 0:
                print(f"Warning: No features extracted from {patient_file}")
                continue

            # Store feature names for importance analysis
            if feature_names is None and 'timestamp' in features_df.columns:
                feature_names = [col for col in features_df.columns if col != 'timestamp']

            col_mapping = patient_df.attrs.get('column_mapping', {})
            time_col = col_mapping.get('time', 'time')
            glucose_col = col_mapping.get('glucose_level', 'glucose_level')

            if 'timestamp' in features_df.columns:
                feature_times = features_df['timestamp'].values
            else:
                feature_times = patient_df[time_col].values[-len(features_df):]

            for hours, model in models.items():
                try:
                    # Get the expected feature order from the model
                    expected_features = model.feature_names_in_

                    # Reorder columns to match the model's expected order
                    ordered_features = pd.DataFrame(index=features_df.index)
                    for feature in expected_features:
                        if feature in features_df.columns:
                            ordered_features[feature] = features_df[feature]
                        else:
                            print(f"Warning: Missing feature {feature}, filling with 0")
                            ordered_features[feature] = 0

                    # Make predictions with properly ordered features
                    predictions = model.predict(ordered_features)

                    true_values = []
                    for i, t in enumerate(feature_times):
                        future_time = t + pd.Timedelta(hours=hours)
                        time_diffs = abs(patient_df[time_col] - future_time)
                        if len(time_diffs) > 0:
                            closest_idx = time_diffs.idxmin()
                            closest_time = patient_df.loc[closest_idx, time_col]

                            if abs((closest_time - future_time).total_seconds()) <= 900:
                                true_values.append(patient_df.loc[closest_idx, glucose_col])
                            else:
                                true_values.append(None)
                        else:
                            true_values.append(None)

                    patient_data = {
                        'file': patient_file,
                        'gender': gender,
                        'age': age,
                        'age_group': 'young' if age and age < 30 else 'middle' if age and age < 60 else 'elder' if age else None,
                        'predictions': predictions,
                        'true_values': true_values,
                        'horizon': hours
                    }

                    demographic_data.append(patient_data)

                except Exception as e:
                    print(f"Error making predictions for {patient_file}, {hours}-hour model: {str(e)}")

        except Exception as e:
            print(f"Error processing {patient_file}: {str(e)}")

    if not demographic_data:
        print("Error: No valid demographic data found")
        return None

    results = {}
    threshold_results = {}
    intersectional_results = {}
    roc_auc_results = {}

    demographic_features = ['gender', 'age_group']

    # Calculate fairness & bias metrics for each demographic feature and horizon
    for feature in demographic_features:
        for horizon in set(d['horizon'] for d in demographic_data):
            horizon_data = [d for d in demographic_data if d['horizon'] == horizon]

            valid_data = [d for d in horizon_data if d[feature] is not None]

            if len(valid_data) < 2:
                continue

            dp_predictions = []
            dp_demographics = []

            eo_predictions = []
            eo_true_values = []
            eo_demographics = []

            for d in valid_data:
                dp_predictions.extend(d['predictions'])
                dp_demographics.extend([d[feature]] * len(d['predictions']))

                valid_indices = [i for i, v in enumerate(d['true_values']) if v is not None]
                if valid_indices:
                    eo_predictions.extend([d['predictions'][i] for i in valid_indices])
                    eo_true_values.extend([d['true_values'][i] for i in valid_indices])
                    eo_demographics.extend([d[feature]] * len(valid_indices))

            if not dp_predictions or not dp_demographics:
                continue

            # Calculate metrics with the main threshold
            dp_results = calculate_demographic_parity(
                np.array(dp_predictions),
                pd.Series(dp_demographics),
                threshold)

            if len(eo_predictions) > 0 and len(np.unique(eo_true_values)) > 1:
                eo_results = calculate_equalized_odds(
                    np.array(eo_predictions),
                    np.array(eo_true_values),
                    pd.Series(eo_demographics),
                    threshold)

                # Calculate ROC AUC for each group
                roc_auc_results[f"{feature}_{horizon}hr"] = calculate_roc_auc_by_group(
                    np.array(eo_predictions),
                    np.array(eo_true_values) >= threshold,
                    pd.Series(eo_demographics))
            else:
                eo_results = {"error": "Insufficient data for equalized odds calculation"}

            results[f"{feature}_{horizon}hr"] = {
                'demographic_parity': dp_results,
                'equalized_odds': eo_results
            }

            # Calculate metrics across multiple thresholds
            if len(eo_predictions) > 0 and len(np.unique(eo_true_values)) > 1:
                multiple_thresholds = [70, 100, 130, 150, 180, 200, 250]
                threshold_results[f"{feature}_{horizon}hr"] = calculate_threshold_metrics(
                    np.array(eo_predictions),
                    np.array(eo_true_values),
                    pd.Series(eo_demographics),
                    multiple_thresholds)

    # Calculate intersectional results (combinations of demographic features)
    if len(demographic_features) > 1:
        for horizon in set(d['horizon'] for d in demographic_data):
            horizon_data = [d for d in demographic_data if d['horizon'] == horizon]

            feature1, feature2 = demographic_features[0], demographic_features[1]

            groups, predictions, true_values = get_intersectional_data(
                horizon_data, feature1, feature2)

            if len(groups) < 2:
                continue

            intersectional_dp = {}
            intersectional_eo = {}

            for group_name, group_predictions in predictions.items():
                if len(group_predictions) < 5:
                    continue

                intersectional_dp[group_name] = np.mean(np.array(group_predictions) >= threshold)

                if group_name in true_values and len(true_values[group_name]) > 0:
                    group_true = true_values[group_name]

                    if len(np.unique(group_true)) > 1:
                        cm = confusion_matrix(
                            np.array(group_true) >= threshold,
                            np.array(group_predictions) >= threshold,
                            labels=[0, 1])

                        tn, fp, fn, tp = cm.ravel()
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                        intersectional_eo[group_name] = {
                            'fpr': fpr,
                            'fnr': fnr,
                            'count': len(group_true)
                        }

            intersectional_results[f"{feature1}_{feature2}_{horizon}hr"] = {
                'demographic_parity': intersectional_dp,
                'equalized_odds': intersectional_eo
            }

    # Feature importance bias analysis
    importance_bias = {}
    if feature_names:
        demographic_feature_cols = [
            col for col in feature_names
            if any(demo in col.lower() for demo in ['gender', 'age', 'sex'])
        ]
        importance_bias = analyze_feature_importance_bias(models, feature_names, demographic_feature_cols)

    # Generate mitigation strategies
    mitigation_strategies = suggest_mitigation_strategies(results, output_dir)

    # Generate visualizations
    if output_dir:
        generate_fairness_visualizations(
            results,
            threshold_results,
            intersectional_results,
            roc_auc_results,
            importance_bias,
            output_dir,
            threshold,
            mitigation_strategies
        )

    explanations = {}

    # Generate explanations
    for key, metric_results in results.items():
        parts = key.split('_')
        feature = parts[0]
        horizon = parts[1]

        dp_results = metric_results['demographic_parity']
        eo_results = metric_results['equalized_odds']

        dp_disparity = dp_results['max_disparity']
        dp_explanation = (
            f"Demographic Parity Analysis for {feature.capitalize()} ({horizon}):\n"
            f"This metric measures whether the model's predictions are distributed "
            f"equally across different {feature} groups.\n\n"
            f"A high glucose prediction is defined as >= {threshold} mg/dL.\n\n"
        )

        for group, pred in dp_results['group_predictions'].items():
            count = dp_results['group_counts'][group]
            dp_explanation += f"- {group.capitalize()} group ({count} samples): {pred:.2%} rate of high predictions\n"

        dp_explanation += f"\nMaximum disparity between groups: {dp_disparity:.2%}\n"

        if dp_disparity > 0.2:
            dp_explanation += (
                "The disparity is HIGH, suggesting the model is making significantly "
                f"different predictions for different {feature} groups.\n"
            )
        elif dp_disparity > 0.1:
            dp_explanation += (
                "The disparity is MODERATE, suggesting some differences in how "
                f"the model treats different {feature} groups.\n"
            )
        else:
            dp_explanation += (
                "The disparity is LOW, suggesting the model is making similar "
                f"predictions across different {feature} groups.\n"
            )

        eo_explanation = (
            f"Equalized Odds Analysis for {feature.capitalize()} ({horizon}):\n"
            f"This metric measures whether the model's error rates are equal "
            f"across different {feature} groups.\n\n"
            f"A high glucose prediction is defined as >= {threshold} mg/dL.\n\n"
        )

        if not eo_results or 'error' in eo_results or len(eo_results.get('group_metrics', {})) < 2:
            eo_explanation += (
                "Insufficient data to calculate equalized odds. "
                "This requires ground truth values for future glucose levels.\n"
            )
        else:
            fpr_disparity = eo_results['max_fpr_disparity']
            fnr_disparity = eo_results['max_fnr_disparity']

            for group, metrics in eo_results['group_metrics'].items():
                count = eo_results['group_counts'][group]
                eo_explanation += (
                    f"- {group.capitalize()} group ({count} samples):\n"
                    f"  * False Positive Rate: {metrics['fpr']:.2%} (incorrect high predictions)\n"
                    f"  * False Negative Rate: {metrics['fnr']:.2%} (missed high predictions)\n"
                )

            eo_explanation += (
                f"\nMaximum FPR disparity between groups: {fpr_disparity:.2%}\n"
                f"Maximum FNR disparity between groups: {fnr_disparity:.2%}\n"
            )

            max_disparity = max(fpr_disparity, fnr_disparity)
            if max_disparity > 0.2:
                eo_explanation += (
                    "The error rate disparity is HIGH, suggesting the model has significantly "
                    f"different error patterns for different {feature} groups.\n"
                )
            elif max_disparity > 0.1:
                eo_explanation += (
                    "The error rate disparity is MODERATE, suggesting some differences in error "
                    f"patterns across {feature} groups.\n"
                )
            else:
                eo_explanation += (
                    "The error rate disparity is LOW, suggesting the model has similar "
                    f"error patterns across different {feature} groups.\n"
                )

        explanations[key] = {
            'demographic_parity': dp_explanation,
            'equalized_odds': eo_explanation
        }

    comprehensive_results = {
        'metrics': results,
        'threshold_metrics': threshold_results,
        'intersectional_results': intersectional_results,
        'roc_auc_results': roc_auc_results,
        'importance_bias': importance_bias,
        'explanations': explanations,
        'mitigation_strategies': mitigation_strategies
    }

    # Save all results to JSON
    if output_dir:
        results_file = os.path.join(output_dir, 'fairness_bias_results.json')
        with open(results_file, 'w') as f:
            # Convert NumPy and pandas types to JSON-serializable types
            json.dump(convert_to_serializable(comprehensive_results), f, indent=2)

    return comprehensive_results


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        # Convert tuple keys to strings for JSON serialization
        return {str(k) if isinstance(k, tuple) else k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def generate_fairness_visualizations(
        results,
        threshold_results,
        intersectional_results,
        roc_auc_results,
        importance_bias,
        output_dir,
        threshold,
        mitigation_strategies
):
    # Set style for visualizations
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")

    # Create subdirectories
    demographic_parity_dir = os.path.join(output_dir, 'demographic_parity')
    equalized_odds_dir = os.path.join(output_dir, 'equalized_odds')
    threshold_analysis_dir = os.path.join(output_dir, 'threshold_analysis')
    intersectional_dir = os.path.join(output_dir, 'intersectional')
    roc_dir = os.path.join(output_dir, 'roc_analysis')
    importance_dir = os.path.join(output_dir, 'feature_importance')

    os.makedirs(demographic_parity_dir, exist_ok=True)
    os.makedirs(equalized_odds_dir, exist_ok=True)
    os.makedirs(threshold_analysis_dir, exist_ok=True)
    os.makedirs(intersectional_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(importance_dir, exist_ok=True)

    # 1. Demographic Parity Visualizations
    for key, metrics in results.items():
        parts = key.split('_')
        feature = parts[0]
        horizon = parts[1]

        dp_results = metrics['demographic_parity']

        plt.figure(figsize=(12, 7))
        groups = list(dp_results['group_predictions'].keys())
        values = list(dp_results['group_predictions'].values())
        counts = [dp_results['group_counts'][g] for g in groups]

        group_labels = [f"{g} (n={c})" for g, c in zip(groups, counts)]

        ax = sns.barplot(x=group_labels, y=values, hue=group_labels, legend=False)
        plt.axhline(y=dp_results['average_rate'], color='r', linestyle='--', alpha=0.7, label='Average')
        plt.xlabel(f"{feature.capitalize()} Group")
        plt.ylabel(f'Rate of High Predictions (>{threshold} mg/dL)')
        plt.title(f'Demographic Parity - {feature.capitalize()} ({horizon})')
        plt.ylim(0, max(1.0, max(values) * 1.1))

        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.1%}", ha='center')

        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(demographic_parity_dir, f'demographic_parity_{feature}_{horizon}.png'), dpi=300)
        plt.close()

    # 2. Equalized Odds Visualizations
    for key, metrics in results.items():
        parts = key.split('_')
        feature = parts[0]
        horizon = parts[1]

        eo_results = metrics['equalized_odds']

        if not eo_results or 'error' in eo_results or len(eo_results.get('group_metrics', {})) < 2:
            continue

        plt.figure(figsize=(14, 8))
        groups = list(eo_results['group_metrics'].keys())
        fpr_values = [eo_results['group_metrics'][g]['fpr'] for g in groups]
        fnr_values = [eo_results['group_metrics'][g]['fnr'] for g in groups]
        tpr_values = [eo_results['group_metrics'][g]['tpr'] for g in groups]
        tnr_values = [eo_results['group_metrics'][g]['tnr'] for g in groups]
        counts = [eo_results['group_counts'][g] for g in groups]

        group_labels = [f"{g} (n={c})" for g, c in zip(groups, counts)]

        x = np.arange(len(groups))
        width = 0.2

        fig, ax = plt.subplots(figsize=(14, 8))

        rects1 = ax.bar(x - width * 1.5, tpr_values, width, label='True Positive Rate', color='green')
        rects2 = ax.bar(x - width / 2, tnr_values, width, label='True Negative Rate', color='blue')
        rects3 = ax.bar(x + width / 2, fpr_values, width, label='False Positive Rate', color='red')
        rects4 = ax.bar(x + width * 1.5, fnr_values, width, label='False Negative Rate', color='orange')

        ax.set_xlabel(f"{feature.capitalize()} Group")
        ax.set_ylabel('Rate')
        ax.set_title(f'Equalized Odds - {feature.capitalize()} ({horizon})')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.legend()

        for rect in rects1 + rects2 + rects3 + rects4:
            height = rect.get_height()
            ax.annotate(f"{height:.2%}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=9)

        fig.tight_layout()
        plt.savefig(os.path.join(equalized_odds_dir, f'equalized_odds_{feature}_{horizon}.png'), dpi=300)
        plt.close()

        # Create confusion matrix visualizations for each group
        for group in groups:
            metrics = eo_results['group_metrics'][group]
            count = eo_results['group_counts'][group]

            cm = np.array([
                [metrics['tn'], metrics['fp']],
                [metrics['fn'], metrics['tp']]
            ])

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=['Normal', 'High Glucose'],
                        yticklabels=['Normal', 'High Glucose'])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'Confusion Matrix - {group.capitalize()} (n={count})')
            plt.tight_layout()
            plt.savefig(os.path.join(equalized_odds_dir, f'confusion_matrix_{feature}_{horizon}_{group}.png'), dpi=300)
            plt.close()

    # 3. Threshold Analysis Visualizations
    for key, thresholds_data in threshold_results.items():
        parts = key.split('_')
        feature = parts[0]
        horizon = parts[1]

        # Plot demographic parity across thresholds
        plt.figure(figsize=(12, 8))
        thresholds = sorted(thresholds_data.keys())

        for group in set().union(*[data['demographic_parity']['group_predictions'].keys()
                                   for data in thresholds_data.values()]):
            values = [thresholds_data[t]['demographic_parity']['group_predictions'].get(group, 0)
                      for t in thresholds]
            plt.plot(thresholds, values, 'o-', label=f"{group.capitalize()}")

        plt.xlabel('Glucose Threshold (mg/dL)')
        plt.ylabel('Rate of High Predictions')
        plt.title(f'Demographic Parity Across Thresholds - {feature.capitalize()} ({horizon})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(threshold_analysis_dir, f'dp_thresholds_{feature}_{horizon}.png'), dpi=300)
        plt.close()

        # Plot disparities across thresholds
        plt.figure(figsize=(12, 8))
        max_disparities = [data['demographic_parity']['max_disparity'] for t, data in sorted(thresholds_data.items())]

        plt.plot(thresholds, max_disparities, 'o-', linewidth=2, color='purple')
        plt.xlabel('Glucose Threshold (mg/dL)')
        plt.ylabel('Maximum Disparity')
        plt.title(f'Maximum Demographic Parity Disparity Across Thresholds - {feature.capitalize()} ({horizon})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(threshold_analysis_dir, f'dp_disparity_thresholds_{feature}_{horizon}.png'), dpi=300)
        plt.close()

        # Plot equalized odds metrics across thresholds (if available)
        valid_eo_thresholds = [t for t, data in thresholds_data.items()
                               if 'equalized_odds' in data and 'error' not in data['equalized_odds']]

        if valid_eo_thresholds:
            plt.figure(figsize=(12, 8))

            fpr_disparities = [thresholds_data[t]['equalized_odds']['max_fpr_disparity'] for t in valid_eo_thresholds]
            fnr_disparities = [thresholds_data[t]['equalized_odds']['max_fnr_disparity'] for t in valid_eo_thresholds]

            plt.plot(valid_eo_thresholds, fpr_disparities, 'o-', label='FPR Disparity', color='red')
            plt.plot(valid_eo_thresholds, fnr_disparities, 'o-', label='FNR Disparity', color='orange')

            plt.xlabel('Glucose Threshold (mg/dL)')
            plt.ylabel('Maximum Disparity')
            plt.title(f'Equalized Odds Disparities Across Thresholds - {feature.capitalize()} ({horizon})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(threshold_analysis_dir, f'eo_disparities_thresholds_{feature}_{horizon}.png'),
                        dpi=300)
            plt.close()

    # 4. Intersectional Analysis Visualizations
    for key, data in intersectional_results.items():
        parts = key.split('_')
        if len(parts) < 3:
            continue

        feature1, feature2, horizon = parts[0], parts[1], parts[2].replace('hr', '')

        # Plot demographic parity for intersectional groups
        if data['demographic_parity']:
            plt.figure(figsize=(14, 8))
            groups = list(data['demographic_parity'].keys())
            values = list(data['demographic_parity'].values())

            indices = np.argsort(values)
            sorted_groups = [groups[i] for i in indices]
            sorted_values = [values[i] for i in indices]

            ax = sns.barplot(x=sorted_groups, y=sorted_values, hue=sorted_groups, palette='viridis', legend=False)
            plt.axhline(y=np.mean(values), color='r', linestyle='--', alpha=0.7, label='Average')
            plt.xlabel(f"{feature1.capitalize()} × {feature2.capitalize()} Groups")
            plt.ylabel(f'Rate of High Predictions (>{threshold} mg/dL)')
            plt.title(f'Intersectional Demographic Parity ({horizon}-hour)')
            plt.xticks(rotation=45, ha='right')

            # Add value labels on top of bars
            for i, v in enumerate(sorted_values):
                ax.text(i, v + 0.02, f"{v:.1%}", ha='center')

            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(intersectional_dir, f'intersectional_dp_{feature1}_{feature2}_{horizon}.png'),
                        dpi=300)
            plt.close()

        # Plot equalized odds for intersectional groups if available
        if data['equalized_odds']:
            plt.figure(figsize=(14, 10))
            groups = list(data['equalized_odds'].keys())
            fpr_values = [data['equalized_odds'][g]['fpr'] for g in groups]
            fnr_values = [data['equalized_odds'][g]['fnr'] for g in groups]
            counts = [data['equalized_odds'][g]['count'] for g in groups]

            group_labels = [f"{g}\n(n={c})" for g, c in zip(groups, counts)]

            x = np.arange(len(groups))
            width = 0.35

            fig, ax = plt.subplots(figsize=(14, 8))

            rects1 = ax.bar(x - width / 2, fpr_values, width, label='False Positive Rate', color='red')
            rects2 = ax.bar(x + width / 2, fnr_values, width, label='False Negative Rate', color='orange')

            ax.set_xlabel(f"{feature1.capitalize()} × {feature2.capitalize()} Groups")
            ax.set_ylabel('Error Rate')
            ax.set_title(f'Intersectional Equalized Odds ({horizon}-hour)')
            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, rotation=45, ha='right')
            ax.legend()

            for rect in rects1 + rects2:
                height = rect.get_height()
                ax.annotate(f"{height:.2%}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

            fig.tight_layout()
            plt.savefig(os.path.join(intersectional_dir, f'intersectional_eo_{feature1}_{feature2}_{horizon}.png'),
                        dpi=300)
            plt.close()

    # 5. ROC AUC Analysis
    for key, data in roc_auc_results.items():
        parts = key.split('_')
        feature = parts[0]
        horizon = parts[1]

        if 'group_auc' in data and any(auc is not None for auc in data['group_auc'].values()):
            plt.figure(figsize=(10, 6))
            groups = []
            auc_values = []
            counts = []

            for group, auc_val in data['group_auc'].items():
                if auc_val is not None:
                    groups.append(group)
                    auc_values.append(auc_val)
                    counts.append(data['group_counts'][group])

            group_labels = [f"{g} (n={c})" for g, c in zip(groups, counts)]

            ax = sns.barplot(x=group_labels, y=values, hue=group_labels, legend=False)
            plt.axhline(y=np.mean(auc_values), color='r', linestyle='--', alpha=0.7, label='Average')
            plt.xlabel(f"{feature.capitalize()} Group")
            plt.ylabel('AUC-ROC')
            plt.title(f'ROC AUC by Group - {feature.capitalize()} ({horizon})')
            plt.ylim(0.5, 1.0)  # AUC is between 0.5 and 1.0

            # Add value labels on top of bars
            for i, v in enumerate(auc_values):
                ax.text(i, v + 0.02, f"{v:.3f}", ha='center')

            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(roc_dir, f'roc_auc_{feature}_{horizon}.png'), dpi=300)
            plt.close()

    # 6. Feature Importance Bias
    if importance_bias:
        for hours, data in importance_bias.items():
            if 'demographic_importances' in data and data['demographic_importances']:
                plt.figure(figsize=(10, 6))
                features = list(data['demographic_importances'].keys())
                values = list(data['demographic_importances'].values())

                sorted_indices = np.argsort(values)[::-1]  # Descending order
                sorted_features = [features[i] for i in sorted_indices]
                sorted_values = [values[i] for i in sorted_indices]

                ax = sns.barplot(x=sorted_features, y=sorted_values, hue=sorted_features, palette='viridis', legend=False)
                plt.xlabel('Demographic Features')
                plt.ylabel('Importance')
                plt.title(f'Demographic Feature Importance ({hours}-hour model)')
                plt.xticks(rotation=45, ha='right')

                # Add value labels on top of bars
                for i, v in enumerate(sorted_values):
                    ax.text(i, v + 0.005, f"{v:.4f}", ha='center')

                plt.tight_layout()
                plt.savefig(os.path.join(importance_dir, f'demographic_importance_{hours}hr.png'), dpi=300)
                plt.close()

                # Create a full feature importance plot
                plt.figure(figsize=(12, 10))
                all_features = list(data['all_importances'].keys())
                all_values = list(data['all_importances'].values())

                # Get top 20 features
                top_indices = np.argsort(all_values)[-20:][::-1]  # Top 20, descending
                top_features = [all_features[i] for i in top_indices]
                top_values = [all_values[i] for i in top_indices]

                # Highlight demographic features
                colors = ['#1f77b4' if f not in data['demographic_importances'] else '#d62728' for f in top_features]
                ax = sns.barplot(x=top_values, y=top_features, hue=top_features, palette=colors, legend=False)

                # Add value labels next to bars
                for i, v in enumerate(top_values):
                    ax.text(v + 0.005, i, f"{v:.4f}", va='center')

                plt.xlabel('Importance')
                plt.ylabel('Features')
                plt.title(f'Top 20 Feature Importance ({hours}-hour model)')

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#1f77b4', label='Standard Features'),
                    Patch(facecolor='#d62728', label='Demographic Features')
                ]
                plt.legend(handles=legend_elements)

                plt.tight_layout()
                plt.savefig(os.path.join(importance_dir, f'top_features_importance_{hours}hr.png'), dpi=300)
                plt.close()

    # 7. Create a summary report
    summary_file = os.path.join(output_dir, 'analysis_summary.html')
    with open(summary_file, 'w') as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fairness and Bias Analysis Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; }}
                .image-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
                .image-box {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
                .image-box img {{ max-width: 100%; height: auto; }}
                .image-box p {{ text-align: center; font-weight: bold; margin-top: 10px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                table, th, td {{ border: 1px solid #ddd; }}
                th, td {{ padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Fairness and Bias Analysis Summary</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Glucose Threshold: {threshold} mg/dL</p>

                <div class="section">
                    <h2>Demographic Parity Analysis</h2>
                    <p>
                        Demographic parity measures whether the model's predictions are distributed 
                        equally across different demographic groups. A high disparity indicates potential unfairness.
                    </p>
                    <div class="image-container">
        ''')

        # Add Demographic Parity Images
        for filename in os.listdir(demographic_parity_dir):
            if filename.endswith('.png'):
                parts = filename.replace('.png', '').split('_')
                if len(parts) >= 3:
                    feature = parts[1]
                    horizon = parts[2]
                    f.write(f'''
                    <div class="image-box">
                        <img src="demographic_parity/{filename}" alt="{filename}">
                        <p>{feature.capitalize()} - {horizon} hour</p>
                    </div>
                    ''')

        f.write('''
                    </div>
                </div>

                <div class="section">
                    <h2>Equalized Odds Analysis</h2>
                    <p>
                        Equalized odds measures whether the model's error rates are equal across different 
                        demographic groups. Disparities in false positive or false negative rates indicate bias.
                    </p>
                    <div class="image-container">
        ''')

        # Add Equalized Odds Images
        for filename in os.listdir(equalized_odds_dir):
            if filename.endswith('.png') and filename.startswith('equalized_odds'):
                parts = filename.replace('.png', '').split('_')
                if len(parts) >= 3:
                    feature = parts[2]
                    horizon = parts[3]
                    f.write(f'''
                    <div class="image-box">
                        <img src="equalized_odds/{filename}" alt="{filename}">
                        <p>{feature.capitalize()} - {horizon} hour</p>
                    </div>
                    ''')

        f.write('''
                    </div>
                </div>

                <div class="section">
                    <h2>Threshold Analysis</h2>
                    <p>
                        This analysis shows how fairness metrics change with different glucose thresholds,
                        helping identify optimal fairness-accuracy trade-offs.
                    </p>
                    <div class="image-container">
        ''')

        # Add Threshold Analysis Images
        for filename in os.listdir(threshold_analysis_dir):
            if filename.endswith('.png'):
                f.write(f'''
                <div class="image-box">
                    <img src="threshold_analysis/{filename}" alt="{filename}">
                    <p>{filename.replace('.png', '').replace('_', ' ').title()}</p>
                </div>
                ''')

        f.write('''
                    </div>
                </div>

                <div class="section">
                    <h2>Intersectional Analysis</h2>
                    <p>
                        Intersectional analysis examines fairness metrics for combinations of demographic attributes,
                        revealing potential biases that affect specific subgroups.
                    </p>
                    <div class="image-container">
        ''')

        # Add Intersectional Analysis Images
        for filename in os.listdir(intersectional_dir):
            if filename.endswith('.png'):
                f.write(f'''
                <div class="image-box">
                    <img src="intersectional/{filename}" alt="{filename}">
                    <p>{filename.replace('.png', '').replace('_', ' ').title()}</p>
                </div>
                ''')

        f.write('''
                    </div>
                </div>

                <div class="section">
                    <h2>ROC AUC Analysis</h2>
                    <p>
                        ROC AUC analysis shows the model's discriminative ability across different demographic groups.
                        Similar AUC values indicate consistent performance.
                    </p>
                    <div class="image-container">
        ''')

        # Add ROC Analysis Images
        for filename in os.listdir(roc_dir):
            if filename.endswith('.png'):
                f.write(f'''
                <div class="image-box">
                    <img src="roc_analysis/{filename}" alt="{filename}">
                    <p>{filename.replace('.png', '').replace('_', ' ').title()}</p>
                </div>
                ''')

        f.write('''
                    </div>
                </div>

                <div class="section">
                    <h2>Feature Importance Analysis</h2>
                    <p>
                        This analysis shows the importance of demographic features in the model,
                        highlighting potential direct or proxy influences of sensitive attributes.
                    </p>
                    <div class="image-container">
        ''')

        # Add Feature Importance Images
        for filename in os.listdir(importance_dir):
            if filename.endswith('.png'):
                f.write(f'''
                <div class="image-box">
                    <img src="feature_importance/{filename}" alt="{filename}">
                    <p>{filename.replace('.png', '').replace('_', ' ').title()}</p>
                </div>
                ''')

        f.write('''
                    </div>
                </div>

                <div class="section">
                    <h2>Mitigation Strategies</h2>
                    <p>
                        Based on the analysis, the following strategies are recommended to address
                        any identified fairness or bias issues.
                    </p>
                    <table>
                        <tr>
                            <th>Feature & Horizon</th>
                            <th>Issue</th>
                            <th>Mitigation Strategies</th>
                        </tr>
        ''')

        # Add Mitigation Strategies
        for key, strategies in mitigation_strategies.items():
            if not strategies:
                continue

            parts = key.split('_')
            feature = parts[0]
            horizon = parts[1]

            for strategy in strategies:
                issue = strategy.get('issue', '')
                recommendations = strategy.get('strategies', [])

                f.write(f'''
                <tr>
                    <td>{feature.capitalize()} ({horizon})</td>
                    <td>{issue}</td>
                    <td><ul>''')

                for recommendation in recommendations:
                    f.write(f'<li>{recommendation}</li>')

                f.write('''</ul></td>
                </tr>
                ''')

        f.write('''
                    </table>
                </div>
            </div>
        </body>
        </html>
        ''')

    print(f"Generated comprehensive analysis report: {summary_file}")


def print_results(analysis_results):
    if not analysis_results:
        print("No analysis results to display")
        return

    print("\n" + "=" * 80)
    print("FAIRNESS AND BIAS ANALYSIS RESULTS")
    print("=" * 80)

    for key, explanations in analysis_results['explanations'].items():
        parts = key.split('_')
        feature = parts[0]
        horizon = parts[1]

        print(f"\n\n{'-' * 80}")
        print(f"ANALYSIS FOR {feature.upper()} GROUPS - {horizon} PREDICTION HORIZON")
        print(f"{'-' * 80}\n")

        print(explanations['demographic_parity'])
        print("\n")
        print(explanations['equalized_odds'])

    print("\n" + "=" * 80)

    if 'mitigation_strategies' in analysis_results:
        print("RECOMMENDED MITIGATION STRATEGIES:")
        print("-" * 80)

        for key, strategies in analysis_results['mitigation_strategies'].items():
            if not strategies:
                continue

            parts = key.split('_')
            feature = parts[0]
            horizon = parts[1]

            print(f"\n{feature.capitalize()} ({horizon}):")

            for strategy in strategies:
                print(f"  • Issue: {strategy['issue']}")
                print("    Strategies:")
                for recommendation in strategy['strategies']:
                    print(f"      - {recommendation}")

    print("\n" + "=" * 80)
    print("Analysis complete. See output directory for visualization plots and detailed reports.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze fairness and bias in glucose prediction models'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='xgboost_models',
        help='Directory containing trained XGBoost models'
    )

    parser.add_argument(
        '--patient-files',
        type=str,
        nargs='+',
        required=True,
        help='CSV files with patient data (must include demographic info in filename, e.g., 45_female.csv)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='fairness_bias_analysis',
        help='Directory to save analysis results and plots'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=150,
        help='Glucose threshold for binary classification (mg/dL)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Generate more detailed analysis with additional metrics'
    )

    args = parser.parse_args()

    # Add this code to expand glob patterns
    import glob
    expanded_files = []
    for pattern in args.patient_files:
        matches = glob.glob(pattern)
        if matches:
            expanded_files.extend(matches)
        else:
            expanded_files.append(pattern)

    if not expanded_files:
        print(f"Error: No files found matching patterns: {args.patient_files}")
        return

    print(f"Analyzing fairness and bias in models from: {args.model_dir}")
    print(f"Using patient files: {expanded_files}")
    print(f"Using glucose threshold: {args.threshold} mg/dL")
    print(f"Outputs will be saved to: {args.output_dir}")

    analysis_results = analyze_model_fairness_bias(
        args.model_dir,
        expanded_files,  # Use the expanded list
        args.output_dir,
        args.threshold,
    )

    if analysis_results:
        print_results(analysis_results)
    else:
        print("Analysis failed or produced no results")


if __name__ == "__main__":
    main()