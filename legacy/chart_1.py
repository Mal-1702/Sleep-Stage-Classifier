import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from collections import Counter
import warnings
# full code for the sleep stage classifier - 

warnings.filterwarnings('ignore')
def plot_epoch_curve(df, stage_col='label', title='Sleep Stage Epoch Curve'):
    """
    Plots an epoch curve (hypnogram-style) for sleep stages.
    """
    plt.figure(figsize=(12,4))
    stages = df[stage_col].values
    plt.plot(stages, marker='o', linestyle='-', markersize=3)
    plt.yticks(np.unique(stages))
    plt.xlabel('Epoch')
    plt.ylabel('Sleep Stage')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_f1_comparison(results, option_labels=['Option A', 'Option B (best)']):
    """
    Plots a simple bar chart comparing F1 macro scores between options.
    """
    option_a_f1 = results['Option_A']['test_f1'] if 'Option_A' in results else 0
    option_b_best = max(results['Option_B'].keys(),
                        key=lambda x: results['Option_B'][x]['test_f1']) if 'Option_B' in results else None
    option_b_f1 = results['Option_B'][option_b_best]['test_f1'] if option_b_best else 0

    plt.figure(figsize=(6,4))
    plt.bar(option_labels, [option_a_f1, option_b_f1], color=['teal','orange'])
    plt.title('Test F1 Macro: Option A vs Option B')
    plt.ylabel('F1 Macro')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

class AdvancedSleepStageClassifier:
    def __init__(self, data_path=None, df=None):
        """
        Advanced classifier with multiple resampling strategies
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either provide data_path or df parameter")

        self.feature_cols = [col for col in self.df.columns if col != 'label']

        print("🔬 ADVANCED RESAMPLING FOR EXTREME IMBALANCE")
        print("=" * 60)
        print(f"Dataset: {len(self.df)} samples, {len(self.feature_cols)} features")
        print("\nClass distribution:")
        class_dist = self.df['label'].value_counts()
        for stage, count in class_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {stage}: {count} samples ({percentage:.1f}%)")

    def option_a_combine_deep_sleep(self):
        """
        OPTION A: Combine Stage 3 + Stage 4 as Deep Sleep (Clinically Standard)
        """
        print("\n" + "=" * 60)
        print("OPTION A: COMBINE STAGE 3+4 AS DEEP SLEEP")
        print("=" * 60)

        df_combined = self.df.copy()

        # Combine Stage 3 and 4
        df_combined['label'] = df_combined['label'].replace({
            'Sleep stage 3': 'Deep Sleep (N3)',
            'Sleep stage 4': 'Deep Sleep (N3)'
        })

        print("✅ Combined Stage 3 + Stage 4 → Deep Sleep (N3)")
        print("\nNew class distribution:")

        new_dist = df_combined['label'].value_counts()
        for stage, count in new_dist.items():
            percentage = (count / len(df_combined)) * 100
            print(f"  {stage}: {count} samples ({percentage:.1f}%)")

        # Calculate balance improvement
        old_ratio = self.df['label'].value_counts().min() / self.df['label'].value_counts().max()
        new_ratio = new_dist.min() / new_dist.max()

        print(f"\n📊 Balance improvement:")
        print(f"  Before: {old_ratio:.4f} (worst: 9/{self.df['label'].value_counts().max()})")
        print(f"  After:  {new_ratio:.4f} (worst: {new_dist.min()}/{new_dist.max()})")
        print(f"  Improvement: {(new_ratio / old_ratio):.1f}x better")

        self.df_combined = df_combined
        return df_combined

    def option_b_advanced_resampling(self):
        """
        OPTION B: Keep all 6 classes with advanced resampling strategies
        """
        print("\n" + "=" * 60)
        print("OPTION B: ADVANCED RESAMPLING (KEEP ALL 6 CLASSES)")
        print("=" * 60)

        X = self.df[self.feature_cols].values
        y = self.df['label'].values

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print("Original training class distribution:")
        train_counts = Counter(y_train)
        for class_idx, count in train_counts.items():
            class_name = label_encoder.inverse_transform([class_idx])[0]
            print(f"  {class_name}: {count} samples")

        # Try different resampling strategies
        resampling_strategies = {
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=2),  # Reduced k_neighbors
            'ADASYN': ADASYN(random_state=42, n_neighbors=2),  # Adaptive synthetic sampling
            'Conservative_SMOTE': SMOTE(random_state=42, k_neighbors=2),  # Very conservative
            'Selective_SMOTE': None  # Custom strategy below
        }

        resampled_data = {}

        for strategy_name, resampler in resampling_strategies.items():
            print(f"\n🔄 Trying {strategy_name}...")

            if strategy_name == 'Selective_SMOTE':
                # Only oversample classes with >20 samples to avoid Stage 4 issues
                X_resampled, y_resampled = self._selective_smote(X_train, y_train, label_encoder)
            else:
                try:
                    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
                    print(f"  ✅ Success - {len(X_resampled)} samples generated")
                except Exception as e:
                    print(f"  ❌ Failed: {str(e)}")
                    continue

            # Show new distribution
            resampled_counts = Counter(y_resampled)
            print(f"  Resampled distribution:")
            for class_idx, count in resampled_counts.items():
                class_name = label_encoder.inverse_transform([class_idx])[0]
                print(f"    {class_name}: {count} samples")

            resampled_data[strategy_name] = {
                'X_train': X_resampled,
                'y_train': y_resampled,
                'X_test': X_test,
                'y_test': y_test
            }

        self.resampled_data = resampled_data
        self.label_encoder = label_encoder
        return resampled_data

    def _selective_smote(self, X_train, y_train, label_encoder):
        """
        Custom SMOTE that only oversamples classes with sufficient samples
        """
        train_counts = Counter(y_train)

        # Only apply SMOTE to classes with >15 samples
        safe_classes = [cls for cls, count in train_counts.items() if count > 15]
        unsafe_classes = [cls for cls, count in train_counts.items() if count <= 15]

        print(f"    Safe classes for SMOTE: {len(safe_classes)}")
        print(f"    Unsafe classes (will use class weights): {len(unsafe_classes)}")

        if len(safe_classes) > 1:
            # Apply SMOTE only to safe classes
            safe_mask = np.isin(y_train, safe_classes)
            X_safe = X_train[safe_mask]
            y_safe = y_train[safe_mask]

            smote = SMOTE(random_state=42, k_neighbors=3)
            X_safe_resampled, y_safe_resampled = smote.fit_resample(X_safe, y_safe)

            # Combine with unsafe classes (unchanged)
            unsafe_mask = ~safe_mask
            X_unsafe = X_train[unsafe_mask]
            y_unsafe = y_train[unsafe_mask]

            X_combined = np.vstack([X_safe_resampled, X_unsafe])
            y_combined = np.concatenate([y_safe_resampled, y_unsafe])

            return X_combined, y_combined
        else:
            # Fallback: no resampling, just return original
            return X_train, y_train

    def compare_approaches(self):
        """
        Train models on both approaches and compare
        """
        print("\n" + "=" * 60)
        print("COMPARING BOTH APPROACHES")
        print("=" * 60)

        results = {}

        # Option A: Combined Deep Sleep
        if hasattr(self, 'df_combined'):
            print("\n🔄 Training Option A (Combined Deep Sleep)...")
            results['Option_A'] = self._train_combined_model()

        # Option B: Advanced Resampling
        if hasattr(self, 'resampled_data'):
            print("\n🔄 Training Option B (Advanced Resampling)...")
            results['Option_B'] = self._train_resampled_models()

        return results

    def _train_combined_model(self):
        """
        Train model on combined deep sleep data
        """
        X = self.df_combined[self.feature_cols].values
        y = self.df_combined['label'].values

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Create pipeline with moderate SMOTE
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=5)),  # Now safe with larger classes
            ('pca', PCA(n_components=6, random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=42
            ))
        ])

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro')

        # Train and test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        test_f1 = f1_score(y_test, y_pred, average='macro')
        test_accuracy = accuracy_score(y_test, y_pred)

        # Per-class performance
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

        result = {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'cv_test_gap': abs(cv_scores.mean() - test_f1),
            'report': report,
            'label_encoder': label_encoder
        }

        print(f"  ✅ CV F1-macro: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  ✅ Test F1-macro: {test_f1:.3f}")
        print(f"  ✅ Test Accuracy: {test_accuracy:.3f}")
        print(f"  ✅ CV-Test Gap: {abs(cv_scores.mean() - test_f1):.3f}")

        return result

    def _train_resampled_models(self):
        """
        Train models on different resampled datasets
        """
        results = {}

        for strategy_name, data in self.resampled_data.items():
            print(f"\n  Training with {strategy_name}...")

            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']

            # Create pipeline (no SMOTE since already resampled)
            pipeline = Pipeline([
                ('pca', PCA(n_components=6, random_state=42)),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, class_weight='balanced', random_state=42
                ))
            ])

            try:
                # Cross-validation on original training data with pipeline including resampling
                if strategy_name != 'Selective_SMOTE':
                    # For comparison, use the resampling strategy in CV
                    if strategy_name == 'BorderlineSMOTE':
                        resampler = BorderlineSMOTE(random_state=42, k_neighbors=2)
                    elif strategy_name == 'ADASYN':
                        resampler = ADASYN(random_state=42, n_neighbors=2)
                    else:
                        resampler = SMOTE(random_state=42, k_neighbors=2)

                    cv_pipeline = ImbPipeline([
                        ('resampler', resampler),
                        ('pca', PCA(n_components=6, random_state=42)),
                        ('classifier', RandomForestClassifier(
                            n_estimators=100, class_weight='balanced', random_state=42
                        ))
                    ])

                    # Get original training data for CV
                    X_orig = self.resampled_data[list(self.resampled_data.keys())[0]]['X_test']  # Use test split logic
                    y_orig = self.resampled_data[list(self.resampled_data.keys())[0]]['y_test']

                    # Recreate original training split
                    X = self.df[self.feature_cols].values
                    y = self.df['label'].values
                    y_encoded = self.label_encoder.fit_transform(y)
                    X_orig_train, _, y_orig_train, _ = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                    )

                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds
                    cv_scores = cross_val_score(cv_pipeline, X_orig_train, y_orig_train,
                                                cv=cv, scoring='f1_macro')
                else:
                    # For selective SMOTE, just use the resampled data
                    cv_scores = np.array([0.0])  # Placeholder

                # Train on resampled data
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                test_f1 = f1_score(y_test, y_pred, average='macro')
                test_accuracy = accuracy_score(y_test, y_pred)

                results[strategy_name] = {
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'test_f1': test_f1,
                    'test_accuracy': test_accuracy,
                    'cv_test_gap': abs(cv_scores.mean() - test_f1) if cv_scores.mean() > 0 else 0
                }

                print(f"    ✅ Test F1-macro: {test_f1:.3f}")
                print(f"    ✅ Test Accuracy: {test_accuracy:.3f}")
                if cv_scores.mean() > 0:
                    print(f"    ✅ CV F1-macro: {cv_scores.mean():.3f}")

            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
                continue

        return results

    def generate_recommendation(self, results):
        """
        Generate final recommendation based on results
        """
        print("\n" + "=" * 60)
        print("FINAL RECOMMENDATION")
        print("=" * 60)

        # Compare Option A vs best of Option B
        option_a_f1 = results.get('Option_A', {}).get('test_f1', 0)
        option_a_gap = results.get('Option_A', {}).get('cv_test_gap', 1)

        option_b_results = results.get('Option_B', {})
        if option_b_results:
            best_b_strategy = max(option_b_results.keys(), key=lambda x: option_b_results[x]['test_f1'])
            option_b_f1 = option_b_results[best_b_strategy]['test_f1']
            option_b_gap = option_b_results[best_b_strategy]['cv_test_gap']
        else:
            best_b_strategy = None
            option_b_f1 = 0
            option_b_gap = 1

        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"  Option A (Combined Deep Sleep):")
        print(f"    • Test F1-macro: {option_a_f1:.3f}")
        print(f"    • CV-Test Gap: {option_a_gap:.3f}")
        print(f"    • Classes: 5 (clinically standard)")

        if best_b_strategy:
            print(f"  Option B (Best: {best_b_strategy}):")
            print(f"    • Test F1-macro: {option_b_f1:.3f}")
            print(f"    • CV-Test Gap: {option_b_gap:.3f}")
            print(f"    • Classes: 6 (includes rare Stage 4)")

        print(f"\n🏆 RECOMMENDATION:")

        # Decision logic
        if option_a_f1 > option_b_f1 + 0.02 or option_a_gap < option_b_gap - 0.02:
            print("✅ CHOOSE OPTION A: Combined Deep Sleep")
            print("\n💡 REASONS:")
            print("  • Better or comparable performance")
            print("  • More stable generalization")
            print("  • Clinically standard approach")
            print("  • Statistically sound (no extreme minorities)")
            print("  • Easier to deploy and interpret")

        elif option_b_f1 > option_a_f1 + 0.05:
            print("✅ CHOOSE OPTION B: Advanced Resampling")
            print(f"  Best strategy: {best_b_strategy}")
            print("\n💡 REASONS:")
            print("  • Significantly better performance")
            print("  • Preserves all clinical classes")
            print("  • Successfully handles extreme imbalance")

        else:
            print("⚖  BOTH OPTIONS VIABLE - RECOMMEND OPTION A")
            print("\n💡 REASONS:")
            print("  • Similar performance but simpler approach")
            print("  • Follows clinical standards")
            print("  • More robust and deployable")
            print("  • Avoids statistical issues with extreme minorities")

        print(f"\n🎯 PRACTICAL ADVICE:")
        print("  • Stage 4 (9 samples) is not statistically learnable")
        print("  • Combined Stage 3+4 is clinically accepted")
        print("  • Focus on robust, deployable solutions")
        print("  • Your performance expectations should be realistic (0.65-0.75 F1)")

        return {
            'recommended': 'option_a' if option_a_f1 >= option_b_f1 - 0.02 else 'option_b',
            'option_a_f1': option_a_f1,
            'option_b_f1': option_b_f1,
            'best_b_strategy': best_b_strategy
        }

# ==============================
# IMPORTS
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ==============================
# HELPER FUNCTIONS
# ==============================

def plot_epoch_curve(df, stage_col='label', title='Sleep Stage Epoch Curve'):
    """
    Plots an epoch curve (hypnogram-style) for sleep stages.
    """
    plt.figure(figsize=(12,4))
    stages = df[stage_col].values
    plt.plot(stages, marker='o', linestyle='-', markersize=3)
    plt.yticks(np.unique(stages))
    plt.xlabel('Epoch')
    plt.ylabel('Sleep Stage')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_f1_comparison(results, option_labels=['Option A', 'Option B (best)']):
    """
    Plots a simple bar chart comparing F1 macro scores between options.
    """
    option_a_f1 = results['Option_A']['test_f1'] if 'Option_A' in results else 0
    option_b_best = max(results['Option_B'].keys(),
                        key=lambda x: results['Option_B'][x]['test_f1']) if 'Option_B' in results else None
    option_b_f1 = results['Option_B'][option_b_best]['test_f1'] if option_b_best else 0

    plt.figure(figsize=(6,4))
    plt.bar(option_labels, [option_a_f1, option_b_f1], color=['teal','orange'])
    plt.title('Test F1 Macro: Option A vs Option B')
    plt.ylabel('F1 Macro')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


# ==============================
# CLASS PLACEHOLDER
# ==============================
class AdvancedSleepStageClassifier:
    # ----- Paste your full class code here -----
    # No changes needed. Your existing methods remain intact.
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        # Add your existing initialization code here
        pass

    # Placeholder methods
    def option_a_combine_deep_sleep(self):
        return self.df  # Replace with actual method

    def option_b_advanced_resampling(self):
        pass

    def compare_approaches(self):
        return {}  # Replace with actual results

    def generate_recommendation(self, results):
        return "Recommendation"  # Replace with actual logic


# ==============================
# MAIN FUNCTION
# ==============================
def main():
    """
    Main execution function
    """
    print("ADVANCED SLEEP STAGE CLASSIFICATION")
    print("Balancing Clinical Value with Statistical Reality")
    print("=" * 70)

    # Initialize classifier - UPDATE PATH HERE
    data_path = r"C:\Users\Malha\Downloads\dream_features.csv"

    try:
        # -----------------------------
        # Initialize classifier
        # -----------------------------
        classifier = AdvancedSleepStageClassifier(data_path=data_path)

        # -----------------------------
        # Option A + Option B
        # -----------------------------
        df_combined = classifier.option_a_combine_deep_sleep()
        classifier.option_b_advanced_resampling()

        # Compare approaches
        results = classifier.compare_approaches()

        # Generate recommendation
        recommendation = classifier.generate_recommendation(results)

        print(f"\n" + "=" * 70)
        print("ANALYSIS COMPLETED!")
        print("=" * 70)

        # -----------------------------
        # PLOTTING SECTION
        # -----------------------------

        # 1️⃣ Original class distribution
        plt.figure(figsize=(8,5))
        sns.countplot(data=classifier.df, x='label', palette='viridis')
        plt.title('Original Sleep Stage Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()

        # 2️⃣ Combined Stage 3+4 distribution
        plt.figure(figsize=(8,5))
        sns.countplot(data=df_combined, x='label', palette='magma')
        plt.title('Combined Deep Sleep Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()

        # 3️⃣ Epoch curves
        print("Plotting epoch curves...")
        plot_epoch_curve(classifier.df, stage_col='label', title='Original Sleep Stage Epoch Curve')
        plot_epoch_curve(df_combined, stage_col='label', title='Combined Deep Sleep Epoch Curve')

        # 4️⃣ Option A vs Option B F1 bar chart
        plot_f1_comparison(results)

        # 5️⃣ Confusion matrix (Option A model)
        if 'Option_A' in results:
            y_labels = results['Option_A']['label_encoder'].classes_
            # Predict on Option A test set again
            X = classifier.df_combined[classifier.feature_cols].values
            y = classifier.df_combined['label'].values
            y_enc = results['Option_A']['label_encoder'].transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
            )
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=5)),
                ('pca', PCA(n_components=6, random_state=42)),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, class_weight='balanced', random_state=42
                ))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            plot_confusion_matrix(y_test, y_pred, labels=y_labels, title='Confusion Matrix (Option A)')

        return classifier, results, recommendation

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


# ==============================
# RUN MAIN
# ==============================
if __name__ == "__main__":
    classifier, results, recommendation = main()

