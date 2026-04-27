import os
import argparse
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

parser = argparse.ArgumentParser(description="Student burnout model training and visualization")
parser.add_argument("--data", default="student_burnout.csv", help="Path to dataset CSV file")
parser.add_argument("--target", default="burnout_level", help="Target column name")
parser.add_argument("--id-column", default="student_id", help="ID column to drop if present")
parser.add_argument(
    "--class-order",
    default="Low,Medium,High",
    help="Comma-separated class order for confusion matrix, for example: Low,Medium,High"
)
args = parser.parse_args()

df = pd.read_csv(args.data)

if args.target not in df.columns:
    raise ValueError(
        f"Target column '{args.target}' not found in dataset. Available columns: {list(df.columns)}"
    )

class_labels = [label.strip() for label in args.class_order.split(",") if label.strip()]
if not class_labels:
    class_labels = sorted(df[args.target].dropna().unique().tolist())

print("Dataset loaded successfully")
print("Shape:", df.shape)

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")

# -----------------------------
# STEP A: EDA Graphs
# -----------------------------

plt.figure(figsize=(8, 5))
sns.countplot(x=args.target, hue=args.target, data=df, palette="Set2", legend=False)
plt.title(f"{args.target} Distribution")
plt.xlabel(args.target)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/burnout_distribution.png")
plt.close()

if "stress_level" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x="stress_level", hue=args.target, data=df, palette="Set1")
    plt.title(f"Stress Level vs {args.target}")
    plt.xlabel("Stress Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/stress_vs_burnout.png")
    plt.close()
else:
    print("Skipped stress vs target plot: 'stress_level' column not found")

if "anxiety_score" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=args.target, y="anxiety_score", hue=args.target, data=df, palette="Pastel1", legend=False)
    plt.title(f"Anxiety Score by {args.target}")
    plt.xlabel(args.target)
    plt.ylabel("Anxiety Score")
    plt.tight_layout()
    plt.savefig("outputs/anxiety_vs_burnout.png")
    plt.close()
else:
    print("Skipped anxiety vs target plot: 'anxiety_score' column not found")

print("Graphs saved successfully in the outputs folder")

# -----------------------------
# STEP B: Data Preparation
# -----------------------------

drop_columns = [args.target]
if args.id_column in df.columns:
    drop_columns.append(args.id_column)

X = df.drop(columns=drop_columns)
y = df[args.target]

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

missing_in_test = [label for label in class_labels if label not in set(y_test)]
if missing_in_test:
    print(f"Warning: some class labels are not present in y_test: {missing_in_test}")

cm = confusion_matrix(y_test, rf_pred, labels=class_labels)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
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