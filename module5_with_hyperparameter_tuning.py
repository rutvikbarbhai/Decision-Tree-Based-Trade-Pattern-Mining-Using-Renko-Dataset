# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised Decision Tree Clustering for Trading Data with Hyperparameter Tuning

Builds interpretable clusters using decision trees with your cleaned features.
Includes optional hyperparameter tuning with temporal cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import time

sns.set_style('whitegrid')


class SupervisedClusteringModel:
    """
    Supervised clustering using decision trees for interpretable trade quality prediction
    """

    def __init__(self, name, max_depth=8, min_samples_split=50, min_samples_leaf=20):
        """
        Args:
            max_depth: Maximum tree depth (controls cluster granularity)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf (controls cluster size)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.feature_names = None
        self.class_names = None
        self.name = name

    def train(self, X, y, feature_names=None, class_names=None):
        """Train the decision tree model"""

        self.feature_names = feature_names if feature_names else X.columns.tolist()
        self.class_names = class_names if class_names else [str(c) for c in sorted(y.unique())]

        print("Training Decision Tree Classifier...")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Classes: {len(self.class_names)}")
        print(f"  Max depth: {self.max_depth}")

        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X, y)

        train_acc = self.model.score(X, y)
        print(f"\nTraining Accuracy: {train_acc:.4f}")

        return self

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""

        y_pred = self.model.predict(X_test)

        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {acc:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': acc,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred,
                                                           target_names=self.class_names,
                                                           output_dict=True)
        }

    def get_cluster_rules(self, max_rules=10):
        """Extract interpretable rules from decision tree"""

        from sklearn.tree import _tree

        tree = self.model.tree_
        feature_names = self.feature_names

        def recurse(node, depth, path_conditions):
            """Recursively extract rules"""

            if tree.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                feature_name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                # Left child (<=)
                left_conditions = path_conditions + [f"{feature_name} <= {threshold:.3f}"]
                recurse(tree.children_left[node], depth + 1, left_conditions)

                # Right child (>)
                right_conditions = path_conditions + [f"{feature_name} > {threshold:.3f}"]
                recurse(tree.children_right[node], depth + 1, right_conditions)
            else:
                # Leaf node
                samples = tree.n_node_samples[node]
                class_counts = tree.value[node][0]
                predicted_class = np.argmax(class_counts)
                confidence = class_counts[predicted_class] / class_counts.sum()

                rules.append({
                    'conditions': path_conditions,
                    'predicted_class': self.class_names[predicted_class],
                    'samples': int(samples),
                    'confidence': float(confidence),
                    'depth': depth
                })

        rules = []
        recurse(0, 0, [])

        # Sort by number of samples (larger clusters first)
        rules = sorted(rules, key=lambda x: x['samples'], reverse=True)

        print("\n" + "=" * 70)
        print(f"TOP {min(max_rules, len(rules))} CLUSTER RULES")
        print("=" * 70)

        for i, rule in enumerate(rules[:max_rules], 1):
            print(f"\nCluster {i}:")
            print(f"  Predicted Class: {rule['predicted_class']}")
            print(f"  Confidence: {rule['confidence']:.2%}")
            print(f"  Samples: {rule['samples']:,}")
            print(f"  Depth: {rule['depth']}")
            print("  Conditions:")
            for condition in rule['conditions']:
                print(f"    • {condition}")

        return rules

    def plot_tree_visualization(self, max_depth_display=3):
        """Visualize the decision tree"""

        fig, ax = plt.subplots(figsize=(20, 12))

        plot_tree(
            self.model,
            max_depth=max_depth_display,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )

        plt.title(f'Decision Tree Visualization (Top {max_depth_display} Levels)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        return fig

    def get_feature_importance(self, top_n=20):
        """Get feature importance from the tree"""

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save(self, output_dir):
        """Save model and metadata"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        joblib.dump(self.model, output_dir / f'decision_tree_model_{self.name}.pkl')

        # Save metadata
        metadata = {
            'max_depth': int(self.max_depth),
            'min_samples_split': int(self.min_samples_split),
            'min_samples_leaf': int(self.min_samples_leaf),
            'n_features': int(len(self.feature_names)),
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'n_leaves': int(self.model.get_n_leaves()),
            'tree_depth': int(self.model.get_depth())
        }

        with open(output_dir / f'model_metadata_{self.name}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n Model saved to: {output_dir}")

    @classmethod
    def load(cls, output_dir, name):
        """Load saved model"""

        output_dir = Path(output_dir)

        # Load metadata
        with open(output_dir / f'model_metadata_{name}.json', 'r') as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            name=name,
            max_depth=metadata['max_depth'],
            min_samples_split=metadata['min_samples_split'],
            min_samples_leaf=metadata['min_samples_leaf']
        )

        # Load model
        instance.model = joblib.load(output_dir / f'decision_tree_model_{name}.pkl')
        instance.feature_names = metadata['feature_names']
        instance.class_names = metadata['class_names']

        return instance


def tune_hyperparameters(X_train, y_train, output_dir, name, quick_mode=False):
    """
    Find optimal hyperparameters using GridSearchCV with TimeSeriesSplit

    Args:
        X_train: Training features
        y_train: Training labels
        output_dir: Directory to save tuning results
        name: Name for saving files
        quick_mode: If True, use smaller search space (faster)
    Returns:
        dict: Best parameters found
    """
    print("\n" + "=" * 80)
    print("🔍 HYPERPARAMETER TUNING WITH TEMPORAL CROSS-VALIDATION")
    print("=" * 80)
    start_time = time.time()
    # Define search space
    if quick_mode:
        print(" Quick mode: Testing limited parameter combinations")
        param_grid = {
            'max_depth': [5, 7, 10],
            'min_samples_split': [50, 100],
            'min_samples_leaf': [20, 30]
        }
    else:
        print(" Full mode: Testing extensive parameter combinations")
        param_grid = {
            'max_depth': [5, 7, 10, 12, 15],
            'min_samples_split': [30, 50, 100, 150],
            'min_samples_leaf': [10, 20, 25, 30, 50]
        }
    total_combinations = len(param_grid['max_depth']) * \
        len(param_grid['min_samples_split']) * \
        len(param_grid['min_samples_leaf'])
    print("\n Search space:")
    print(f"   max_depth: {param_grid['max_depth']}")
    print(f"   min_samples_split: {param_grid['min_samples_split']}")
    print(f"   min_samples_leaf: {param_grid['min_samples_leaf']}")
    print(f"   Total combinations: {total_combinations}")
    print(f"\n  Estimated time: {total_combinations * 5 // 60} - {total_combinations * 10 // 60} minutes")
    # Initialize model
    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    #  Use TimeSeriesSplit for temporal data
    tscv = TimeSeriesSplit(n_splits=5)
    print("\n  Using TimeSeriesSplit (respects temporal order, no shuffle)")
    print("   This ensures train data always comes before test data\n")
    # Run grid search
    grid_search = GridSearchCV(
        dt,
        param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    print(" Starting grid search...")
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    # Extract results
    results = grid_search.cv_results_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("\n" + "=" * 80)
    print(" HYPERPARAMETER TUNING RESULTS")
    print("=" * 80)
    print(f"\n  Time elapsed: {elapsed_time / 60:.1f} minutes")
    print("\n Best Parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"\n Best CV Score: {best_score:.4f}")

    # Show top 5 parameter combinations
    print("\n Top 5 Parameter Combinations:")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rank_test_score')

    for i, row in results_df.head(5).iterrows():
        print(f"\n   Rank {int(row['rank_test_score'])}:")
        print(f"      max_depth: {row['param_max_depth']}")
        print(f"      min_samples_split: {row['param_min_samples_split']}")
        print(f"      min_samples_leaf: {row['param_min_samples_leaf']}")
        print(f"      CV Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"      Train Score: {row['mean_train_score']:.4f}")
        gap = row['mean_train_score'] - row['mean_test_score']
        print(f"      Overfitting Gap: {gap:.4f}")

    # Save detailed results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save best parameters
    with open(output_dir / f'best_hyperparameters_{name}.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_cv_score': float(best_score),
            'tuning_time_minutes': elapsed_time / 60,
            'total_combinations_tested': total_combinations
        }, f, indent=2)

    # Save all results as CSV
    results_summary = pd.DataFrame({
        'max_depth': results['param_max_depth'],
        'min_samples_split': results['param_min_samples_split'],
        'min_samples_leaf': results['param_min_samples_leaf'],
        'mean_cv_score': results['mean_test_score'],
        'std_cv_score': results['std_test_score'],
        'mean_train_score': results['mean_train_score'],
        'rank': results['rank_test_score']
    }).sort_values('rank')

    results_summary.to_csv(output_dir / f'hyperparameter_tuning_results_{name}.csv', index=False)

    print("\n Tuning results saved:")
    print(f"   • {output_dir / f'best_hyperparameters_{name}.json'}")
    print(f"   • {output_dir / f'hyperparameter_tuning_results_{name}.csv'}")

    # Create visualization of top parameters
    plot_tuning_results(results_summary, output_dir, name)

    return best_params


def plot_tuning_results(results_df, output_dir, name):
    """Create visualization of hyperparameter tuning results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: CV Score by max_depth
    ax1 = axes[0, 0]
    depth_scores = results_df.groupby('max_depth')['mean_cv_score'].agg(['mean', 'max'])
    ax1.plot(depth_scores.index, depth_scores['mean'], 'o-', label='Mean CV Score', linewidth=2, markersize=8)
    ax1.plot(depth_scores.index, depth_scores['max'], 's--', label='Best CV Score', linewidth=2, markersize=6)
    ax1.set_xlabel('Max Depth', fontweight='bold', fontsize=12)
    ax1.set_ylabel('CV Score', fontweight='bold', fontsize=12)
    ax1.set_title('Performance vs Tree Depth', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: CV Score by min_samples_split
    ax2 = axes[0, 1]
    split_scores = results_df.groupby('min_samples_split')['mean_cv_score'].agg(['mean', 'max'])
    ax2.plot(split_scores.index, split_scores['mean'], 'o-', label='Mean CV Score', linewidth=2, markersize=8)
    ax2.plot(split_scores.index, split_scores['max'], 's--', label='Best CV Score', linewidth=2, markersize=6)
    ax2.set_xlabel('Min Samples Split', fontweight='bold', fontsize=12)
    ax2.set_ylabel('CV Score', fontweight='bold', fontsize=12)
    ax2.set_title('Performance vs Min Samples Split', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: CV Score by min_samples_leaf
    ax3 = axes[1, 0]
    leaf_scores = results_df.groupby('min_samples_leaf')['mean_cv_score'].agg(['mean', 'max'])
    ax3.plot(leaf_scores.index, leaf_scores['mean'], 'o-', label='Mean CV Score', linewidth=2, markersize=8)
    ax3.plot(leaf_scores.index, leaf_scores['max'], 's--', label='Best CV Score', linewidth=2, markersize=6)
    ax3.set_xlabel('Min Samples Leaf', fontweight='bold', fontsize=12)
    ax3.set_ylabel('CV Score', fontweight='bold', fontsize=12)
    ax3.set_title('Performance vs Min Samples Leaf', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Top 10 parameter combinations
    ax4 = axes[1, 1]
    top10 = results_df.head(10).copy()
    top10['params'] = top10.apply(
        lambda x: f"D{x['max_depth']}_S{x['min_samples_split']}_L{x['min_samples_leaf']}",
        axis=1
    )
    ax4.barh(range(len(top10)), top10['mean_cv_score'], color='steelblue', alpha=0.7)
    ax4.set_yticks(range(len(top10)))
    ax4.set_yticklabels(top10['params'], fontsize=9)
    ax4.set_xlabel('CV Score', fontweight='bold', fontsize=12)
    ax4.set_title('Top 10 Parameter Combinations', fontweight='bold', fontsize=14)
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'hyperparameter_tuning_plots_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   • {output_dir / f'hyperparameter_tuning_plots_{name}.png'}")


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix heatmap"""

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)

    ax.set_xlabel('Predicted', fontweight='bold', fontsize=12)
    ax.set_ylabel('Actual', fontweight='bold', fontsize=12)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14, pad=15)

    plt.tight_layout()
    return fig


def run_supervised_clustering(train_path, test_path, features_path, output_dir, name,
                              enable_tuning=False, quick_tuning=False,
                              manual_params=None):
    """
    Train and evaluate supervised clustering for a given dataset.

    Args:
        train_path: Path to training parquet file
        test_path: Path to test parquet file
        features_path: Path to features text file
        output_dir: Directory to save outputs
        name: Name for this run (long/short)
        enable_tuning: If True, run hyperparameter tuning
        quick_tuning: If True, use quick tuning mode (fewer combinations)
        manual_params: Dict of manual parameters (if not tuning)

    Returns:
        tuple: (model, results, best_params)
    """
    name = name.lower().replace(" trades", "")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print(f"SUPERVISED CLUSTERING: {output_dir}")
    print("=" * 80)

    # === Load data ===
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f" Train: {len(train_df):,} | Test: {len(test_df):,}")

    # === Temporal validation ===
    if 'entry_date' in train_df.columns and 'entry_date' in test_df.columns:
        train_start = train_df['entry_date'].min()
        train_end = train_df['entry_date'].max()
        test_start = test_df['entry_date'].min()
        test_end = test_df['entry_date'].max()

        print("\n Temporal Check:")
        print(f"   Train: {train_start} to {train_end}")
        print(f"   Test:  {test_start} to {test_end}")

        if train_end >= test_start:
            print(f"     WARNING: Train ends at {train_end}, test starts at {test_start}")
            print("   This may cause lookahead bias!")
        else:
            print("    No temporal leak (train ends before test starts)")

        # Sort by time
        train_df = train_df.sort_values('entry_date').reset_index(drop=True)
        test_df = test_df.sort_values('entry_date').reset_index(drop=True)
        print("    Data sorted by entry_date")

    # === Load features ===
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]

    print(f"\n Using {len(features)} features")

    # === Prepare data ===
    X_train = train_df[features].fillna(0)
    y_train = train_df['label']
    X_test = test_df[features].fillna(0)
    y_test = test_df['label']

    class_names = sorted(y_train.unique().astype(str))

    # ================================================================
    # HYPERPARAMETER TUNING (OPTIONAL)
    # ================================================================
    best_params = None

    if enable_tuning:
        best_params = tune_hyperparameters(
            X_train, y_train, output_dir, name, quick_mode=quick_tuning
        )
    elif manual_params is not None:
        best_params = manual_params
        print("\n Using manual parameters:")
        for k, v in manual_params.items():
            print(f"   {k}: {v}")
    else:
        # Use default parameters
        best_params = {
            'max_depth': 7,
            'min_samples_split': 50,
            'min_samples_leaf': 25
        }
        print("\n Using default parameters:")
        for k, v in best_params.items():
            print(f"   {k}: {v}")

    # ================================================================
    # TRAIN MODEL WITH BEST PARAMETERS
    # ================================================================
    print("\n" + "=" * 80)
    print(" TRAINING MODEL WITH SELECTED PARAMETERS")
    print("=" * 80)

    model = SupervisedClusteringModel(
        name,
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf']
    )
    model.train(X_train, y_train, feature_names=features, class_names=class_names)

    # === Cross-validation check (to verify generalization) ===
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION CHECK (TEMPORAL)")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model.model, X_train, y_train, cv=tscv, scoring='accuracy')
    print(f"  Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"  Std Dev:          {cv_scores.std():.4f}")
    print(f"  Train Accuracy:   {model.model.score(X_train, y_train):.4f}")
    gap = abs(model.model.score(X_train, y_train) - cv_scores.mean())
    print(f"  Gap:              {gap:.4f}")

    if gap < 0.05:
        print("   GOOD: Model generalizes well (gap < 5%)")
    elif gap < 0.10:
        print("    FAIR: Some overfitting (gap 5-10%)")
    else:
        print("   BAD: Significant overfitting (gap > 10%)")

    print("\n  Note: Using TimeSeriesSplit (respects temporal order)")
    print("=" * 70)

    # === Evaluate ===
    results = model.evaluate(X_test, y_test)

    # === Extract rules and importance ===
    rules = model.get_cluster_rules(max_rules=10)
    importance_df = model.get_feature_importance(top_n=15)

    # === Visualization ===
    print("\n Generating visualizations...")

    fig1 = model.plot_tree_visualization(max_depth_display=3)
    fig1.savefig(output_dir / f'decision_tree_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig2 = plot_confusion_matrix(results['confusion_matrix'], class_names)
    fig2.savefig(output_dir / f'confusion_matrix_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig3, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df['importance'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_title('Feature Importance', fontweight='bold', fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig3.savefig(output_dir / f'feature_importance_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === Save outputs ===
    model.save(output_dir)
    with open(output_dir / f'cluster_rules_{name}.json', 'w') as f:
        json.dump(rules, f, indent=2)

    # Save final parameters used
    with open(output_dir / f'final_parameters_{name}.json', 'w') as f:
        json.dump({
            'parameters': best_params,
            'train_accuracy': float(model.model.score(X_train, y_train)),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'test_accuracy': float(results['accuracy']),
            'tree_depth': int(model.model.get_depth()),
            'n_leaves': int(model.model.get_n_leaves())
        }, f, indent=2)

    print(f"\n Clustering complete for: {output_dir}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Tree depth: {model.model.get_depth()}")
    print(f"Leaves: {model.model.get_n_leaves()}")
    print(f"All results saved in: {output_dir}\n")

    return model, results, best_params


def main():
    """
    Run supervised clustering for both LONG and SHORT datasets.

    Set ENABLE_TUNING = True to run hyperparameter tuning.
    """

    # ================================================================
    # CONFIGURATION
    # ================================================================
    test_name = "base_ind_osc"

    # HYPERPARAMETER TUNING SETTINGS
    ENABLE_TUNING = False  # Set to True to enable hyperparameter tuning and False to disable hyperparameter tuning
    QUICK_TUNING = False    # Set to True for quick tuning (fewer combinations)

    # Manual parameters (only used if ENABLE_TUNING = False)
    MANUAL_PARAMS = {
        'max_depth': 7,
        'min_samples_split': 150,
        'min_samples_leaf': 25
    }

    # ================================================================

    configs = [
        {
            "name": "Long Trades",
            "train": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/train_prepared_long.parquet",
            "test": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/test_prepared_long.parquet",
            "features": f"./outputs/test_{test_name}/long_trades/module4_feature_reduction_long_trades/final_features_long.txt",
            "output": f"./outputs/test_{test_name}/long_trades/module5_supervised_clustering_long_trades"
        },
        {
            "name": "Short Trades",
            "train": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/train_prepared_short.parquet",
            "test": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/test_prepared_short.parquet",
            "features": f"./outputs/test_{test_name}/short_trades/module4_feature_reduction_short_trades/final_features_short.txt",
            "output": f"./outputs/test_{test_name}/short_trades/module5_supervised_clustering_short_trades"
        }
    ]

    results_summary = {}
    best_params_summary = {}

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f" RUNNING {cfg['name'].upper()} CLUSTERING")
        print("=" * 80)

        model, results, best_params = run_supervised_clustering(
            cfg["train"],
            cfg["test"],
            cfg["features"],
            cfg["output"],
            cfg['name'],
            enable_tuning=ENABLE_TUNING,
            quick_tuning=QUICK_TUNING,
            manual_params=MANUAL_PARAMS if not ENABLE_TUNING else None
        )

        results_summary[cfg["name"]] = results
        best_params_summary[cfg["name"]] = best_params

    print("\n" + "=" * 80)
    print(" ALL SUPERVISED CLUSTERING COMPLETE")
    print("=" * 80)

    # Print summary
    print("\n FINAL SUMMARY:")
    for name, results in results_summary.items():
        params = best_params_summary[name]
        print(f"\n{name}:")
        print(f"   Test Accuracy: {results['accuracy']:.4f}")
        print(f"   Parameters: depth={params['max_depth']}, "
              f"split={params['min_samples_split']}, "
              f"leaf={params['min_samples_leaf']}")

    return results_summary, best_params_summary


if __name__ == "__main__":
    results_summary, best_params_summary = main()
