import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Download from Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
data = pd.read_csv('diabetes.csv')  # Ensure the file is in the same directory

# Data Exploration
print("--- Dataset Head ---")
print(data.head())
print("\n--- Missing Values ---")
print(data.isnull().sum())
print("\n--- Data Description ---")
print(data.describe())

# Replace zeros with mean (for Glucose, BloodPressure, etc.)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[zero_cols] = data[zero_cols].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)

# Split features (X) and target (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Compare model performance
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.7, 0.9)
plt.show()

# Feature Importance (for Decision Tree)
dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance (Decision Tree)")
plt.show()