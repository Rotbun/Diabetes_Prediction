import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
import lime
import lime.lime_tabular

# Load data
df = pd.read_csv('data_diabetes.csv')


# Scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
df[numerical_features] = scaler.fit_transform(df[numerical_features])


# Data Exploration

# Bar chart for categorical variables
def plot_bar(df, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column)
    plt.title(f'Bar Chart of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()


# Pie chart for categorical variables
def plot_pie(df, column):
    plt.figure(figsize=(10, 6))
    df[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie Chart of {column}')
    plt.ylabel('')
    plt.show()

# Histogram for continuous variables
def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Box plot for continuous variables
def plot_box(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y=column)
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.show()

# Plotting categorical variables
plot_bar(df, 'gender')
plot_pie(df, 'smoking_history')
plot_bar(df,'hypertension')
plot_pie(df,'heart_disease')


# Plotting continuous variables
plot_histogram(df, 'age')
plot_box(df, 'bmi')
plot_box(df,'blood_glucose_level')
plot_histogram(df, 'HbA1c_level')

# Display summary statistics
print(df.describe())

label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['smoking_history'] = label_encoder.fit_transform(df['smoking_history'])
df['hypertension'] = label_encoder.fit_transform(df['hypertension'])
df['heart_disease'] = label_encoder.fit_transform(df['heart_disease'])


# Scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Define features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from imblearn.over_sampling import SMOTE
# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check the distribution of the target variable after balancing
print("Distribution of target variable in the training set after balancing:")
print(y_train.value_counts())

# Encode the labels in y_train and y_test
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Define a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    return accuracy, precision, recall, f1, auc


# Classical Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}


# Ensemble Models
ensemble_models = {
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "CatBoost": CatBoostClassifier(verbose=0)
}

param_grids = {
    "Logistic Regression": {"C": np.logspace(-4, 4, 20)},
    "Decision Tree": {"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10]},
    "SVM": {"C": np.logspace(-4, 4, 20), "kernel": ["linear", "rbf"]},
    "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
    "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
    "CatBoost": {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "depth": [3, 5, 7]}
}

# Initialize results dictionary
# Initialize results dictionary
results = {}
confusion_matrices = {}

# Train classical and ensemble models using RandomizedSearchCV
for name, model in {**models, **ensemble_models}.items():
    print(f"Training {name}...")
    random_search = RandomizedSearchCV(model, param_grids[name], n_iter=10, scoring='accuracy', cv=5, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # StackingClassifier for ensemble of all models
    estimators = [(name, RandomizedSearchCV(model, param_grids[name], n_iter=10, scoring='accuracy', cv=5, verbose=1, random_state=42, n_jobs=-1).fit(X_train, y_train).best_estimator_) for name, model in {**models, **ensemble_models}.items()]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stacking_model.fit(X_train, y_train)

    stacking_accuracy, stacking_precision, stacking_recall, stacking_f1, stacking_auc, stacking_y_pred = evaluate_model(stacking_model, X_test, y_test)
    results["Stacking Model"] = (stacking_accuracy, stacking_precision, stacking_recall, stacking_f1, stacking_auc, stacking_y_pred)
    confusion_matrices["Stacking Model"] = confusion_matrix(y_test, stacking_y_pred)

    # Evaluate the best model and store metrics
    accuracy, precision, recall, f1, auc = evaluate_model(best_model, X_test, y_test)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Save results
    results[name] = (accuracy, precision, recall, f1, auc, y_pred)
    confusion_matrices[name] = cm

# Check if the results dictionary is not empty
if results:
    # Identify the best model
    best_model_name = max(results, key=lambda k: results[k][0])
    best_model_metrics = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy={best_model_metrics[0]}, Precision={best_model_metrics[1]}, Recall={best_model_metrics[2]}, F1-score={best_model_metrics[3]}, AUC-ROC={best_model_metrics[4]}")
else:
    print("No models were evaluated. Please check the model training process.")

# Plotting confusion matrices
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

for name, cm in confusion_matrices.items():
    plot_confusion_matrix(cm, name)
    

# Comparison of predictions with actual using Bar chart
def plot_predictions_vs_actual_bar(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    
    # Calculate frequencies
    unique, counts_true = np.unique(y_true, return_counts=True)
    unique, counts_pred = np.unique(y_pred, return_counts=True)
    
    # Bar width
    width = 0.4
    
    # Indices for actual and predicted bars
    indices = np.arange(len(unique))
    
    # Bar charts
    plt.bar(indices, counts_true, width, label='Actual', color='blue', alpha=0.6)
    plt.bar(indices + width, counts_pred, width, label='Predicted', color='red', alpha=0.6)
    
    # Titles and labels
    plt.title(f'Frequency of Predictions vs Actual for {model_name}')
    plt.xlabel('Class (0: No Diabetes, 1: Diabetes)')
    plt.ylabel('Frequency')
    plt.xticks(indices + width / 2, unique)  # Set x-ticks to be at the center of the bars
    plt.legend()
    plt.show()

# Iterating through the results to plot for each model
for name, (_, _, _, _, _, y_pred) in results.items():
    plot_predictions_vs_actual_bar(y_test, y_pred, name)
    

# Ensure best_model is trained before using SHAP explainer
best_model.fit(X_train, y_train)

# Use the appropriate SHAP explainer based on the type of the best model
if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, CatBoostClassifier, DecisionTreeClassifier)):
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.KernelExplainer(best_model.predict, X_train)

shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns)


# Performance measure graphs
def plot_performance_measures(results):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for model_name, scores in results.items():
            plt.bar(model_name, scores[metrics.index(metric)], label=metric)
        plt.title(f'Model Comparison: {metric}')
        plt.ylabel(metric)
        plt.show()

plot_performance_measures(results)



# Explanation with LIME (example for Logistic Regression)
import lime
import lime.lime_tabular

def explain_with_lime(model, X_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], discretize_continuous=True)
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(X_test.iloc[i], model.predict_proba, num_features=5)
    exp.show_in_notebook(show_table=True)
    

import lime
import lime.lime_tabular

def explain_with_lime(model, X_train, X_test, feature_names, categorical_features):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['No Diabetes', 'Diabetes'],
        categorical_features=categorical_features,
        discretize_continuous=True
    )

    # Select a random instance from the test set for explanation
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(
        data_row=X_test.values[i],
        predict_fn=model.predict_proba,
        num_features=5
    )
    
    # Display explanation in notebook
    exp.show_in_notebook(show_table=True)

# Example usage with the best model
categorical_features_indices = [X_train.columns.get_loc(col) for col in ['gender', 'smoking_history', 'hypertension', 'heart_disease']]
explain_with_lime(best_model, X_train, X_test, X_train.columns.tolist(), categorical_features_indices)


# Imports for additional plots
# Import the new PartialDependenceDisplay from sklearn.inspection
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.inspection import PartialDependenceDisplay

# Function to plot all performance measures in one figure
def plot_performance_measures_combined(results):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]
    num_metrics = len(metrics)
    
    plt.figure(figsize=(40, 30))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i + 1)
        values = [scores[metrics.index(metric)] for model_name, scores in results.items()]
        plt.bar(results.keys(), values, label=metric)
        plt.title(f'Model Comparison: {metric}')
        plt.ylabel(metric)
        plt.xlabel
    
    plt.tight_layout()
    plt.show()

plot_performance_measures_combined(results)

# Function to plot predictions vs actuals in a single figure
def plot_predictions_vs_actual_bar_combined(y_test, results):
    plt.figure(figsize=(40, 30))
    
    for i, (model_name, (_, _, _, _, _, y_pred)) in enumerate(results.items()):
        plt.subplot(3, 3, i + 1)
        unique, counts_true = np.unique(y_test, return_counts=True)
        unique, counts_pred = np.unique(y_pred, return_counts=True)
        
        width = 1.0
        indices = np.arange(len(unique))
        plt.bar(indices, counts_true, width, label='Actual', color='blue', alpha=0.6)
        plt.bar(indices + width, counts_pred, width, label='Predicted', color='red', alpha=0.6)
        
        plt.title(f'Predictions vs Actual for {model_name}')
        plt.xlabel('Class (0: No Diabetes, 1: Diabetes)')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_predictions_vs_actual_bar_combined(y_test, results)

# SHAP Explanation for Top 3 Models
top_3_models = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[:3]

for model_name, (accuracy, precision, recall, f1, auc, y_pred) in top_3_models:
    print(f"Explaining {model_name} using SHAP...")
    
    model = {**models, **ensemble_models}[model_name]
    model.fit(X_train, y_train)
    
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, CatBoostClassifier, DecisionTreeClassifier)):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)
    
    shap_values = explainer.shap_values(X_test)
    
    # SHAP Summary Plot
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)
    
    # SHAP Waterfall Plot for the first observation
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test.iloc[0].values, feature_names=X.columns.tolist()))
    
    # SHAP Force Plot
    shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)

# LIME Explanation for Top 3 Models
for model_name, _ in top_3_models:
    print(f"Explaining {model_name} using LIME...")
    
    model = {**models, **ensemble_models}[model_name]
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X.columns.tolist(),
        class_names=['No Diabetes', 'Diabetes'],
        discretize_continuous=True
    )
    
    # Select a random instance from the test set for explanation
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(
        data_row=X_test.values[i],
        predict_fn=model.predict_proba,
        num_features=5
    )
    
    exp.show_in_notebook(show_table=True)

# PDP and ICE Plots for Advanced Explanations
for model_name, _ in top_3_models:
    print(f"Generating Partial Dependence and ICE Plots for {model_name}...")
    
    model = {**models, **ensemble_models}[model_name]
    model.fit(X_train, y_train)
    
    features = [0, 1]  # Example feature indices to plot
    display = PartialDependenceDisplay.from_estimator(model, X_train, features)
    display.plot()

# Permutation Feature Importance
for model_name, _ in top_3_models:
    print(f"Computing Permutation Feature Importance for {model_name}...")
    
    model = {**models, **ensemble_models}[model_name]
    model.fit(X_train, y_train)
    
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
    plt.title(f"Permutation Importances for {model_name}")
    plt.show()


# Import the new PartialDependenceDisplay from sklearn.inspection
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

# Code to plot Partial Dependence Plots using PartialDependenceDisplay
def plot_partial_dependence_plots(best_model, X_train, features):
    """
    Function to plot Partial Dependence Plots for the best model.
    
    Parameters:
    - best_model: The trained best model for which to plot the partial dependence.
    - X_train: Training data to fit the model.
    - features: List of feature indices or feature names to plot.
    """
    # Ensure the model is fitted and features are provided
    if best_model and len(features) > 0:
        print(f"\nPlotting Partial Dependence for: {features}")
        # Use PartialDependenceDisplay to plot partial dependence plots
        display = PartialDependenceDisplay.from_estimator(
            best_model,
            X_train,
            features,
            kind="both",  # both individual (ICE) and average (PDP) curves
            grid_resolution=50,
            n_jobs=-1
        )
        display.figure_.suptitle('Partial Dependence Plots')
        display.figure_.tight_layout()
    else:
        print("Model or features not available for Partial Dependence Plots.")

# Example of usage
features_to_plot = [0, 1, 2]  # Indices of features you want to plot (or use feature names)
plot_partial_dependence_plots(best_model, X_train, features_to_plot)



# # Generate explanations for CatBoost model
# catboost_explanation = explainer.explain_instance(data_instance, catboost_model.predict_proba)
# xgboost_explanation = explainer.explain_instance(data_instance, xgboost_model.predict_proba)

# # Display the LIME explanations in Jupyter Notebook
# display(catboost_explanation)
# display(xgboost_explanation)

# # Save explanations as HTML files
# with open('lime_explanation_catboost.html', 'w') as file:
#     file.write(catboost_explanation.as_html())

# with open('lime_explanation_xgboost.html', 'w') as file:
#     file.write(xgboost_explanation.as_html())

# # Open HTML files in a web browser
# webbrowser.open('lime_explanation_catboost.html')
# webbrowser.open('lime_explanation_xgboost.html')
