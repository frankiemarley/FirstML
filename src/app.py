import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Step 1: Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', delimiter=';')

# Step 2: Perform a full EDA
# Drop duplicates
data = data.drop_duplicates().reset_index(drop=True)

# Convert categorical variables into dummy/indicator variables
data_encoded = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'])

# Encode the target variable
data_encoded['y'] = data_encoded['y'].map({'no': 0, 'yes': 1})

# Split the data into features (X) and target (y)
X = data_encoded.drop('y', axis=1)
y = data_encoded['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 3: Build a logistic regression model
model_balanced = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, solver='liblinear')
model_balanced.fit(X_train_resampled, y_train_resampled)

# Make predictions on the training set
y_train_pred = model_balanced.predict(X_train_scaled)

# Evaluate the model on the training set
print("Classification Report for Training Set:")
print(classification_report(y_train, y_train_pred))
print("\nConfusion Matrix for Training Set:")
print(confusion_matrix(y_train, y_train_pred))

# Make predictions on the test set
y_pred = model_balanced.predict(X_test_scaled)

# Evaluate the model on the test set
print("\nClassification Report for Test Set:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix for Test Set:")
print(confusion_matrix(y_test, y_pred))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model_balanced.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# ROC AUC
roc_auc = roc_auc_score(y_test, model_balanced.predict_proba(X_test_scaled)[:, 1])
print(f"\nROC AUC: {roc_auc}")

# Feature Importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model_balanced.coef_[0]})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Most Important Features')
plt.show()

# Step 4: Optimize the previous model using GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # Only 'liblinear' solver supports both 'l1' and 'l2' penalties
}

grid_search = GridSearchCV(LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000), param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train_scaled, y_train)

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_model.predict(X_test_scaled)
print("\nBest Model - Classification Report:")
print(classification_report(y_test, y_pred_best))
print("\nBest Model - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# ROC AUC
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
print(f"\nROC AUC: {roc_auc}")

# Feature Importance (for Logistic Regression with L1 penalty)
if 'l1' in best_model.get_params()['penalty']:
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': best_model.coef_[0]})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Top 10 Most Important Features')
    plt.show()
