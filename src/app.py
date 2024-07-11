import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE

data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', delimiter=';')

# # 1. Visión general de los datos
# print(data)
# print(data.info())
# print(data.shape)

# # 2. Análisis de valores faltantes
# print("\nValores faltantes:")
# print(data.isnull().sum())

# # 3. Estadísticas descriptivas
# print("\nEstadísticas descriptivas:")
# print(data.describe())


duplicadas_incluyendo_primera = data[data.duplicated(keep=False)]
print(duplicadas_incluyendo_primera)

# Drop duplicates
data_no_duplicates = data.drop_duplicates()

# If you want to reset the index after dropping duplicates:
data_no_duplicates = data_no_duplicates.reset_index(drop=True)
print(data.duplicated().sum())

# Print the shape before and after to see how many duplicates were removed
print("Shape before dropping duplicates:", data.shape)
print("Shape after dropping duplicates:", data_no_duplicates.shape)

# # Analyze categorical variables
# for col in data.select_dtypes(include=['object']).columns:
#     print(f"\n{col}:")
#     print(data[col].value_counts(normalize=True))

# # Visualize the distribution of the target variable
# plt.figure(figsize=(8, 6))
# sns.countplot(x='y', data=data)
# plt.title('Distribution of Target Variable')
# plt.show()

# # Visualize correlations between numerical variables
# numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
# correlation_matrix = data[numerical_cols].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()

# # Analyze the relationship between age and the target variable
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='y', y='age', data=data)
# plt.title('Age Distribution by Target Variable')
# plt.show()

# # Analyze the relationship between job and the target variable
# plt.figure(figsize=(12, 6))
# sns.countplot(x='job', hue='y', data=data)
# plt.title('Job Distribution by Target Variable')
# plt.xticks(rotation=45)
# plt.show()

# # Encode categorical variables
# data_encoded = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'])

# # Encode the target variable
# data_encoded['y'] = data_encoded['y'].map({'no': 0, 'yes': 1})

# # Split the data into features (X) and target (y)
# X = data_encoded.drop('y', axis=1)
# y = data_encoded['y']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)    

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# # Train the model with balanced classes
# model_balanced = LogisticRegression(random_state=42, class_weight='balanced')
# model_balanced.fit(X_train_resampled, y_train_resampled)


# # Make predictions on the training set
# y_train_pred = model_balanced.predict(X_train_scaled)

# # Evaluate the model on the training set
# print("Classification Report for Training Set:")
# print(classification_report(y_train, y_train_pred))
# print("\nConfusion Matrix for Training Set:")
# print(confusion_matrix(y_train, y_train_pred))

# # Make predictions on the test set
# y_pred = model_balanced.predict(X_test_scaled)

# # Evaluate the model
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix test:")
# print(confusion_matrix(y_test, y_pred))


# # Define the parameter grid
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga']
# }

# # Create the GridSearchCV object
# grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='f1')

# # Fit the GridSearchCV
# grid_search.fit(X_train_scaled, y_train)

# # Print the best parameters and score
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score:", grid_search.best_score_)

# # Use the best model to make predictions
# best_model = grid_search.best_estimator_
# y_pred_best = best_model.predict(X_test_scaled)

# # Evaluate the best model
# print("\nClassification Report for Best Model:")
# print(classification_report(y_test, y_pred_best))
# print("\nConfusion Matrix for Best Model:")
# print(confusion_matrix(y_test, y_pred_best))


# Encode categorical variables
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

# Train the model with balanced classes
model_balanced = LogisticRegression(random_state=42, class_weight='balanced')
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

# Feature Selection
selector = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42))
selector.fit(X_train_scaled, y_train)

X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Hyperparameter Tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train_selected, y_train)

best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_model.predict(X_test_selected)
print("\nBest Model - Classification Report:")
print(classification_report(y_test, y_pred_best))
print("\nBest Model - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# Try Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
print("\nRandom Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nRandom Forest - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test_selected)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# ROC AUC
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_selected)[:, 1])
print(f"\nROC AUC: {roc_auc}")

# Feature Importance (for Random Forest)
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Most Important Features')
plt.show()


