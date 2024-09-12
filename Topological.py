import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np

# Data snippet
df = pd.read_csv('data_diabetes.csv')

# Data exploration analysis
print(df.describe())
print(df.info())

# Charting the Categorical Variables

plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df)
plt.title('Distribution of Gender')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='hypertension', data=df)
plt.title('Distribution of Hypertension')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='diabetes', data=df)
plt.title('Distribution of Diabetes')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='smoking_history', data=df)
plt.title('Distribution of Smoking History')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='heart_disease', data=df)
plt.title('Distribution of Heart Disease')
plt.show()


# Charting the Continuous Variables

plt.figure(figsize=(8, 6))
sns.histplot(df['bmi'], kde=False)
plt.title('Distribution of BMI')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['age'], kde=False)
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['HbA1c_level'], kde=False)
plt.title('Distribution of HbA1c Level')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['blood_glucose_level'], kde=False)
plt.title('Distribution of Blood Glucose Level')
plt.show()

# Correlation Plot of Continuous Variables

# Select only the continuous variables
continuous_vars = ['bmi', 'age', 'HbA1c_level', 'blood_glucose_level']

# Compute the correlation matrix
corr_matrix = df[continuous_vars].corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Continuous Variables')
plt.show()


# Display summary statistics
print(df.describe())


# Data Preprocessing
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['hypertension'] = le.fit_transform(df['hypertension'])
df['heart_disease'] = le.fit_transform(df['heart_disease'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])

# Split the data
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for Topological Model
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# Topological Machine Learning Model - Vietoris-Rips Complex
topological_pipeline = Pipeline([
    ("vietoris_rips", VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1])),
    ("persistence_entropy", PersistenceEntropy()),
    ("svc", SVC(probability=True))
])

# Fit and predict with reshaped input
topological_pipeline.fit(X_train_reshaped, y_train)
y_pred_topological = topological_pipeline.predict(X_test_reshaped)
y_prob_topological = topological_pipeline.predict_proba(X_test_reshaped)[:, 1]

# Evaluation of the Topological model
print("Topological Model Evaluation:")
print(classification_report(y_test, y_pred_topological))

# Confusion Matrix for Topological Model
cm_topological = confusion_matrix(y_test, y_pred_topological)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_topological, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Topological Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Additional Plot: Bar Chart of Topological Model Prediction Probabilities
plt.figure(figsize=(10, 6))

# Define the bins and calculate the counts
bins = np.linspace(0, 1, 11)  # Creating 10 bins between 0 and 1
counts, bin_edges = np.histogram(y_prob_topological, bins=bins)

# Plot the bar chart
plt.bar(bin_edges[:-1], counts, width=0.1, alpha=0.75, color='blue', edgecolor='black', align='edge')
plt.title('Bar Chart of Predicted Probabilities - Topological Model')
plt.xlabel('Predicted Probability Range')
plt.ylabel('Frequency')
plt.xticks(bin_edges)  # Set x-ticks to show the bin edges

plt.show()


# ROC Curve for Topological Model
plt.figure(figsize=(12, 6))
fpr_topological, tpr_topological, _ = roc_curve(y_test, y_prob_topological)
plt.plot(fpr_topological, tpr_topological, label='Topological Model (AUC = %0.2f)' % auc(fpr_topological, tpr_topological))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Topological Model')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve for Topological Model
plt.figure(figsize=(12, 6))
precision_topological, recall_topological, _ = precision_recall_curve(y_test, y_prob_topological)
plt.plot(recall_topological, precision_topological, label='Topological Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Topological Model')
plt.legend(loc="lower left")
plt.show()

# Plotting Persistence Diagrams (Topological Model Feature Analysis)
from gtda.plotting import plot_diagram

# Assuming Vietoris-Rips persistence step output is accessible
persistence_diagrams = topological_pipeline.named_steps["vietoris_rips"].fit_transform(X_train_reshaped)

plt.figure(figsize=(12, 6))
plot_diagram(persistence_diagrams[0], homology_dimensions=[0, 1])
plt.title('Persistence Diagram - Vietoris-Rips Complex (Sample)')
plt.show()

# Comparison with other models (RandomForest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Evaluation:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest Model
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Random Forest Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plotting evaluation and predictions
plt.figure(figsize=(12, 6))

# ROC Curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_topological, tpr_topological, label='Topological Model (AUC = %0.2f)' % auc(fpr_topological, tpr_topological))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %0.2f)' % auc(fpr_rf, tpr_rf))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curves
plt.figure(figsize=(12, 6))
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)

plt.plot(recall_topological, precision_topological, label='Topological Model')
plt.plot(recall_rf, precision_rf, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower left")
plt.show()

# Feature Importance for Random Forest
plt.figure(figsize=(10, 6))
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.title('Feature Importance - Random Forest')
plt.show()
