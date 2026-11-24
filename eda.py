import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cleaned dataset path
DATA_PATH = r"C:\Users\manik\Downloads\cleaned_healthcare_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("📊 Starting Exploratory Data Analysis...\n")

print("Dataset Shape:")
print(df.shape)

print("\n📋 Dataset Columns:")
print(df.columns.tolist())


# ==========================================================
# 1. Basic Statistics
# ==========================================================

print("\n📈 Basic Statistics:")
print(df.describe())


# ==========================================================
# 2. Risk Distribution
# ==========================================================

risk_columns = [
    "heart_risk",
    "diabetes_risk",
    "hypertension_risk",
    "kidney_risk",
    "stroke_risk",
    "cancer_screening_risk",
    "liver_risk",
    "respiratory_risk",
    "overall_health_risk"
]

for column in risk_columns:

    print(f"\n{column} distribution:")
    print(df[column].value_counts())


# ==========================================================
# 3. Age Distribution
# ==========================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["age"],
    bins=20,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()


# ==========================================================
# 4. BMI Distribution
# ==========================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["bmi"],
    bins=25,
    kde=True
)

plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Number of Patients")
plt.show()


# ==========================================================
# 5. Blood Pressure
# ==========================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="systolic_bp"
)

plt.title("Age vs Systolic Blood Pressure")
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.show()


# ==========================================================
# 6. Glucose Distribution
# ==========================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["glucose_mg_dl"],
    bins=30,
    kde=True
)

plt.title("Glucose Distribution")
plt.xlabel("Glucose (mg/dL)")
plt.ylabel("Number of Patients")
plt.show()


# ==========================================================
# 7. Cholesterol Distribution
# ==========================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["cholesterol_mg_dl"],
    bins=30,
    kde=True
)

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol (mg/dL)")
plt.ylabel("Number of Patients")
plt.show()


# ==========================================================
# 8. Overall Health Risk
# ==========================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="overall_health_risk"
)

plt.title("Overall Health Risk Distribution")
plt.xlabel("Health Risk")
plt.ylabel("Number of Patients")
plt.show()


# ==========================================================
# 9. Correlation Heatmap
# ==========================================================

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Healthcare Features Correlation Heatmap")
plt.show()


print("\n✅ EDA Completed Successfully!")