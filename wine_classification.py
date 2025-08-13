# Task 4: Machine Learning Model Implementation with the Wine dataset

# 1. Import necessary libraries
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the dataset
print("Loading the Wine dataset...")
wine = load_wine()
X = wine.data
y = wine.target

# Print dataset details
print("\nDataset loaded successfully.")
print("Features (X) shape:", X.shape)
print("Target labels (y) shape:", y.shape)
print("Target names:", wine.target_names)
print("Feature names:", wine.feature_names)

# 3. Split the data into training and testing sets
print("\nSplitting data into training and testing sets (70/30 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# 4. Create and train the machine learning model
print("\nCreating and training a Decision Tree Classifier model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# 5. Make predictions on the test set
print("\nMaking predictions on the test set...")
y_pred = model.predict(X_test)
print("Predictions complete.")

# 6. Evaluate the model's performance
print("\nEvaluating the model's performance:")

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy Score: {accuracy * 100:.2f}%")

# Print the classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 7. Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=wine.target_names,
    yticklabels=wine.target_names
)
plt.title('Confusion Matrix – Decision Tree Classifier (Wine Dataset)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\n✅ Task complete. The model was successfully implemented and evaluated.")
