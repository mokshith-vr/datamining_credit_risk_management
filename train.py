import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

os.makedirs('models', exist_ok=True)

# ── Stage 1: Load & EDA ──────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 1 — Data Loading & EDA")
print("=" * 60)

df = pd.read_csv('loan_data.csv')
print(f"Shape: {df.shape}")
print(f"Nulls:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"\nClass distribution:\n{df['loan_status'].value_counts()}")
print(df['loan_status'].value_counts(normalize=True).round(3))

# Class distribution bar
fig, ax = plt.subplots(figsize=(5, 4))
df['loan_status'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'], edgecolor='black')
ax.set_xticklabels(['No Default (0)', 'Default (1)'], rotation=0)
ax.set_title('Class Distribution — loan_status')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.close()
print("Saved: class_distribution.png")

# Histograms
numerical_cols = ['person_age', 'person_income', 'loan_amnt', 'credit_score']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flatten(), numerical_cols):
    ax.hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='black')
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.suptitle('Histograms — Key Numerical Features', fontsize=13)
plt.tight_layout()
plt.savefig('histograms.png', dpi=150)
plt.close()
print("Saved: histograms.png")

# Boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ['loan_amnt', 'credit_score']):
    df.boxplot(column=col, by='loan_status', ax=ax)
    ax.set_title(f'{col} by loan_status')
    ax.set_xlabel('loan_status (0=No Default, 1=Default)')
plt.suptitle('')
plt.tight_layout()
plt.savefig('boxplots.png', dpi=150)
plt.close()
print("Saved: boxplots.png")

# Correlation heatmap
num_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                'credit_score', 'loan_status']
corr = df[num_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
ax.set_title('Correlation Matrix — Numerical Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")
print("\nTop correlations with loan_status:")
print(corr['loan_status'].sort_values(key=abs, ascending=False))

# ── Stage 2: Preprocessing ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2 — Preprocessing")
print("=" * 60)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

numerical_features = ['person_age', 'person_income', 'person_emp_exp',
                       'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                       'cb_person_cred_hist_length', 'credit_score']
ordinal_features = ['person_education']
ordinal_categories = [['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']]
onehot_features = ['person_gender', 'person_home_ownership',
                   'loan_intent', 'previous_loan_defaults_on_file']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features)
])
print("Preprocessor defined.")

# ── Stage 3: Train/Test Split ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 3 — Train/Test Split")
print("=" * 60)

from sklearn.model_selection import train_test_split

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(preprocessor, 'models/preprocessor.pkl')

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Feature count after encoding: {X_train_processed.shape[1]}")
print(f"Train class: {dict(y_train.value_counts())}")
print(f"Test class:  {dict(y_test.value_counts())}")
print("Saved: models/preprocessor.pkl")

# ── Approach 1: Logistic Regression ─────────────────────────────────────────
print("\n" + "=" * 60)
print("APPROACH 1 — Logistic Regression (GridSearchCV)")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000]
}
lr = LogisticRegression(class_weight='balanced', random_state=42)
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='f1', n_jobs=1, verbose=2)
grid_lr.fit(X_train_processed, y_train)
best_lr = grid_lr.best_estimator_

joblib.dump(best_lr, 'models/lr_model.pkl')
print(f"\nBest params: {grid_lr.best_params_}")
print(f"Best CV F1:  {grid_lr.best_score_:.4f}")
print("Saved: models/lr_model.pkl")

# ── Approach 2: Random Forest ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("APPROACH 2 — Random Forest (GridSearchCV)")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1', n_jobs=1, verbose=2)
grid_rf.fit(X_train_processed, y_train)
best_rf = grid_rf.best_estimator_

joblib.dump(best_rf, 'models/rf_model.pkl')
print(f"\nBest params: {grid_rf.best_params_}")
print(f"Best CV F1:  {grid_rf.best_score_:.4f}")
print("Saved: models/rf_model.pkl")

# ── Approach 3: XGBoost + SMOTE-ENN ─────────────────────────────────────────
print("\n" + "=" * 60)
print("APPROACH 3 — XGBoost + SMOTE-ENN (GridSearchCV)")
print("=" * 60)

from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN

print("Applying SMOTE-ENN on training data...")
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_processed, y_train)

print(f"Before SMOTE-ENN: {dict(pd.Series(y_train).value_counts())}")
print(f"After  SMOTE-ENN: {dict(pd.Series(y_train_resampled).value_counts())}")

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [1, 2]
}
xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='f1', n_jobs=1, verbose=2)
grid_xgb.fit(X_train_resampled, y_train_resampled)
best_xgb = grid_xgb.best_estimator_

joblib.dump(best_xgb, 'models/xgb_model.pkl')
print(f"\nBest params: {grid_xgb.best_params_}")
print(f"Best CV F1:  {grid_xgb.best_score_:.4f}")
print("Saved: models/xgb_model.pkl")

# ── Approach 4: SHAP-Guided Stacking Ensemble ────────────────────────────────
print("\n" + "=" * 60)
print("APPROACH 4 — SHAP-Guided Heterogeneous Stacking Ensemble")
print("=" * 60)

import shap
from sklearn.ensemble import StackingClassifier

print("Computing SHAP values from XGBoost...")
feature_names = preprocessor.get_feature_names_out()
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_train_resampled)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

print("\nTop 10 features by SHAP importance:")
print(importance_df.head(10).to_string(index=False))

# SHAP bar chart
fig, ax = plt.subplots(figsize=(8, 6))
importance_df.head(10).sort_values('shap_importance').plot(
    kind='barh', x='feature', y='shap_importance', ax=ax,
    color='steelblue', legend=False
)
ax.set_title('Top 10 Features by SHAP Importance (XGBoost)')
ax.set_xlabel('Mean |SHAP Value|')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150)
plt.close()
print("Saved: shap_importance.png")

top_features_idx = importance_df.head(10).index.tolist()
X_train_shap = X_train_resampled[:, top_features_idx]
X_test_shap = X_test_processed[:, top_features_idx]

print("\nBuilding stacking ensemble...")
estimators = [
    ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=42),
    cv=5,
    n_jobs=1
)
stacking.fit(X_train_shap, y_train_resampled)

joblib.dump(stacking, 'models/stacking_model.pkl')
joblib.dump(top_features_idx, 'models/top_features_idx.pkl')
print("Saved: models/stacking_model.pkl")
print("Saved: models/top_features_idx.pkl")

# ── Stage 5: Benchmarking ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 5 — Benchmarking")
print("=" * 60)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    RocCurveDisplay
)

models_eval = {
    'Logistic Regression': (best_lr, X_test_processed),
    'Random Forest': (best_rf, X_test_processed),
    'XGBoost + SMOTE-ENN': (best_xgb, X_test_processed),
    'SHAP Stacking': (stacking, X_test_shap)
}

results = []
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for name, (model, X_test_used) in models_eval.items():
    y_pred = model.predict(X_test_used)
    y_prob = model.predict_proba(X_test_used)[:, 1]
    results.append({
        'Approach': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1-Score': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)
    })
    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_roc)

ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
ax_roc.set_title('ROC Curves — All 4 Approaches')
ax_roc.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.close()
print("Saved: roc_curves.png")

results_df = pd.DataFrame(results)
print("\n===== BENCHMARK RESULTS =====")
print(results_df.to_string(index=False))

# Confusion matrices
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, (model, X_test_used)) in zip(axes_cm.flatten(), models_eval.items()):
    y_pred = model.predict(X_test_used)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrices — All 4 Approaches', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.close()
print("Saved: confusion_matrices.png")

print("\n===== ALL DONE =====")
print("Models saved:")
for f in sorted(os.listdir('models')):
    print(f"  models/{f}")
