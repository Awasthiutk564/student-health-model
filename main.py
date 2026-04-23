import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("student_burnout.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")

# -----------------------------
# STEP A: EDA Graphs
# -----------------------------

plt.figure(figsize=(8, 5))
sns.countplot(x="burnout_level", hue="burnout_level", data=df, palette="Set2", legend=False)
plt.title("Burnout Level Distribution")
plt.xlabel("Burnout Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/burnout_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(x="stress_level", hue="burnout_level", data=df, palette="Set1")
plt.title("Stress Level vs Burnout Level")
plt.xlabel("Stress Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/stress_vs_burnout.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x="burnout_level", y="anxiety_score", hue="burnout_level", data=df, palette="Pastel1", legend=False)
plt.title("Anxiety Score by Burnout Level")
plt.xlabel("Burnout Level")
plt.ylabel("Anxiety Score")
plt.tight_layout()
plt.savefig("outputs/anxiety_vs_burnout.png")
plt.close()

print("Graphs saved successfully in the outputs folder")

# -----------------------------
# STEP B: Data Preparation
# -----------------------------

X = df.drop(columns=["burnout_level", "student_id"])
y = df["burnout_level"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# STEP C: Logistic Regression
# -----------------------------

log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)

print("\nLogistic Regression Accuracy:", round(log_accuracy * 100, 2), "%")

# -----------------------------
# STEP D: Random Forest
# -----------------------------

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# -----------------------------
# STEP E: Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, rf_pred, labels=["Low", "Medium", "High"])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_rf.png")
plt.close()

print("Random Forest confusion matrix saved successfully in the outputs folder")

# -----------------------------
# STEP F: Accuracy Comparison Graph
# -----------------------------

accuracy_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [log_accuracy * 100, rf_accuracy * 100]
})

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=accuracy_df, x="Model", y="Accuracy", palette="Set3")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")

for i, value in enumerate(accuracy_df["Accuracy"]):
    ax.text(i, value + 0.3, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("outputs/model_accuracy_comparison.png")
plt.close()

print("Model accuracy comparison graph saved successfully")

# -----------------------------
# STEP G: Feature Importance Graph
# -----------------------------

feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importances = rf_model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

top_features = importance_df.sort_values("Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Important Features - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("outputs/feature_importance_rf.png")
plt.close()

print("Feature importance graph saved successfully")