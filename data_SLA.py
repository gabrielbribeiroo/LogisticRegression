import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# Load data from CSV file
csv_file = "Dados-SLA.csv"
raw_data = pd.read_csv(csv_file, header=None)

num_clients = 119
num_features = 12  # 1 ID + 11 responses

# Reshape the data to the correct format
reshaped_data = raw_data.values.reshape(num_clients, num_features)

# Define column names
columns = [
    "Client ID", "Service Rate", "Completed Orders", "Speed", "Consistency",
    "Flexibility", "Failure Recovery", "Information", "Correct Invoices",
    "Conforming Products", "Correct Product Quantity", "Overall Satisfaction"
]

# Create DataFrame with properly distributed data
data = pd.DataFrame(reshaped_data, columns=columns)

# Create target variable: satisfied clients (1) or unsatisfied (0)
data["Satisfied"] = (data["Overall Satisfaction"].astype(int) >= 4).astype(int)

# Separate independent variables (X) and dependent variable (y)
X = data.drop(columns=["Client ID", "Overall Satisfaction", "Satisfied"])
y = data["Satisfied"]

# Check label distribution before training
print("Original client distribution:", np.bincount(y))

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train with Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.7,
    random_state=42
)
gb_model.fit(X_scaled, y)

# Prediction
y_pred_gb = gb_model.predict(X_scaled)

# Model evaluation
print("Predicted label distribution:", np.bincount(y_pred_gb))
print("Accuracy (Gradient Boosting):", accuracy_score(y, y_pred_gb))
print("\nClassification Report (Gradient Boosting):\n", classification_report(y, y_pred_gb))

# Error analysis
data["Predicted"] = y_pred_gb
data["Correct"] = (data["Satisfied"] == data["Predicted"]).astype(int)
wrong_clients = data[data["Correct"] == 0][["Client ID", "Satisfied", "Predicted"]]
print("\nClients that reduced model accuracy:")
print(wrong_clients)

# Feature importance
importances = gb_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).round(3)

# Display importance table
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis("tight")
ax.axis("off")
table = ax.table(
    cellText=importance_df.values,
    colLabels=importance_df.columns,
    cellLoc='center',
    loc='center',
    colColours=["#D3D3D3", "#D3D3D3"]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.title("Feature Importance - Gradient Boosting")
plt.show()

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y, y_pred_gb), annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()

# Normalized confusion matrix
ConfusionMatrixDisplay.from_predictions(y, y_pred_gb, normalize="true", cmap="Blues")
plt.title("Adjusted Confusion Matrix (Normalized)")
plt.show()