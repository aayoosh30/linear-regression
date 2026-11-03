import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df = pd.read_csv("dataset/immunization.csv.csv")
# Display first 5 records
print(df.head())

# Check number of rows and columns
print(df.shape)

# Display column names
print(df.columns)

# Check data types and missing values
df.info()

# Display statistical summary
print(df.describe())
# Display all unique indicator names
print(df["GHO (DISPLAY)"].unique())
# Filter indicators containing the word immunization
immunization_data = df[
    df["GHO (DISPLAY)"].str.contains(
        "immunization", case=False, na=False
    )
]

print(immunization_data["GHO (DISPLAY)"].unique())
vaccination_data = df[
    df["GHO (DISPLAY)"].str.contains(
        "vaccin", case=False, na=False
    )
]

print(vaccination_data["GHO (DISPLAY)"].unique())

# 1. Check which countries are present
print("\nCOUNTRIES:")
print(df["COUNTRY (DISPLAY)"].value_counts().head(10))


# 2. Filter immunization coverage indicators
immunization_data = df[
    df["GHO (DISPLAY)"].str.contains(
        "immunization coverage", case=False, na=False
    )
].copy()


# 3. Count records for each indicator
print("\nNUMBER OF RECORDS PER VACCINE:")
print(immunization_data["GHO (DISPLAY)"].value_counts())


# 4. Display selected columns
print("\nSAMPLE IMMUNIZATION RECORDS:")
print(immunization_data[
    ["GHO (CODE)", "GHO (DISPLAY)", "YEAR (DISPLAY)",
     "COUNTRY (DISPLAY)", "Numeric", "Value"]
].head(15))


# 5. Check available years
print("\nAVAILABLE YEARS:")
print(sorted(immunization_data["YEAR (DISPLAY)"].unique()))

# Filter immunization coverage records

immunization_data = df[
    df["GHO (DISPLAY)"].str.contains(
        "immunization coverage",
        case=False,
        na=False
    )
].copy()


# Display important columns
print("\nDETAILED IMMUNIZATION DATA:")

print(immunization_data[
    [
        "GHO (CODE)",
        "GHO (DISPLAY)",
        "YEAR (DISPLAY)",
        "DIMENSION (TYPE)",
        "DIMENSION (NAME)",
        "Numeric",
        "Value"
    ]
].head(20))


# Check how many records have missing numerical values
print("\nMISSING NUMERIC VALUES:")
print(immunization_data["Numeric"].isnull().sum())


# Check repeated records
print("\nDUPLICATE RECORDS:")
print(immunization_data.duplicated().sum())


# Check the number of records per year
print("\nRECORDS PER YEAR:")
print(immunization_data["YEAR (DISPLAY)"].value_counts().sort_index())

# Extract MCV1 records

mcv1_data = df[
    df["GHO (CODE)"] == "WHS4_543"
].copy()

# Sort records by year
mcv1_data = mcv1_data.sort_values("YEAR (DISPLAY)")

# Display selected columns
print("\nMCV1 DATA:")
print(mcv1_data[
    ["YEAR (DISPLAY)", "Numeric", "Value"]
].to_string(index=False))

# Check number of records
print("\nTOTAL MCV1 RECORDS:")
print(mcv1_data.shape)

# Check missing numerical values
print("\nMISSING VALUES:")
print(mcv1_data["Numeric"].isnull().sum())

# Sort data according to year
mcv1_data = mcv1_data.sort_values("YEAR (DISPLAY)")

# Create a line graph
plt.figure(figsize=(10, 5))

plt.plot(
    mcv1_data["YEAR (DISPLAY)"],
    mcv1_data["Numeric"],
    marker="o"
)

plt.title("MCV1 Immunization Coverage in Nepal (2000–2025)")
plt.xlabel("Year")
plt.ylabel("Immunization Coverage (%)")
plt.grid(True)

plt.show()

# Filter MCV1 data
# STEP 1: Filter MCV1 data again

# Extract MCV1 data

mcv1_data = df[
    df["GHO (DISPLAY)"] ==
    "Measles-containing-vaccine first-dose (MCV1) immunization coverage among 1-year-olds (%) (WUENIC)"
].copy()

# Sort by year
mcv1_data = mcv1_data.sort_values("YEAR (DISPLAY)")

# Display verified data
print("\n--- VERIFIED MCV1 DATA ---")

print(mcv1_data[[
    "GHO (CODE)",
    "YEAR (DISPLAY)",
    "Numeric",
    "Value"
]].to_string(index=False))

print("\nTotal records:", len(mcv1_data))

# Statistical analysis
print("\n--- MCV1 STATISTICAL ANALYSIS ---")

print("Mean:", mcv1_data["Numeric"].mean())
print("Median:", mcv1_data["Numeric"].median())
print("Standard Deviation:", mcv1_data["Numeric"].std())
print("Variance:", mcv1_data["Numeric"].var())
print("Minimum:", mcv1_data["Numeric"].min())
print("Maximum:", mcv1_data["Numeric"].max())

print("\n--- CHECKING MCV1 DATA SOURCES ---")

print(
    df[df["GHO (DISPLAY)"].str.contains("MCV1", na=False)]
    [["GHO (CODE)", "GHO (DISPLAY)", "YEAR (DISPLAY)", "Numeric", "Value"]]
    .sort_values(["GHO (CODE)", "YEAR (DISPLAY)"])
    .to_string(index=False)
)

# Prepare data for Linear Regression

X = mcv1_data[["YEAR (DISPLAY)"]]
y = mcv1_data["Numeric"]

print("\n--- DATA PREPARATION ---")

print("X (Independent Variable):")
print(X.head())

print("\ny (Dependent Variable):")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("\n--- TRAIN TEST SPLIT ---")

print("Training X shape:", X_train.shape)
print("Testing X shape:", X_test.shape)

print("Training y shape:", y_train.shape)
print("Testing y shape:", y_test.shape)

print("\nTraining years:")
print(X_train["YEAR (DISPLAY)"].tolist())

print("\nTesting years:")
print(X_test["YEAR (DISPLAY)"].tolist())

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model using training data
model.fit(X_train, y_train)

# Predict coverage for the testing years
y_pred = model.predict(X_test)

print("\n--- LINEAR REGRESSION RESULTS ---")

print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Compare actual and predicted values
results = pd.DataFrame({
    "Year": X_test["YEAR (DISPLAY)"].values,
    "Actual Coverage": y_test.values,
    "Predicted Coverage": y_pred
})

print("\nActual vs Predicted:")
print(results.to_string(index=False))

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

import matplotlib.pyplot as plt

# Plot training data
plt.scatter(
    X_train["YEAR (DISPLAY)"],
    y_train,
    color="blue",
    label="Training Data"
)

# Plot actual testing data
plt.scatter(
    X_test["YEAR (DISPLAY)"],
    y_test,
    color="green",
    label="Actual Testing Data"
)

# Plot predicted values
plt.scatter(
    X_test["YEAR (DISPLAY)"],
    y_pred,
    color="red",
    label="Predicted Values"
)

# Plot regression line
plt.plot(
    X["YEAR (DISPLAY)"],
    model.predict(X),
    color="orange",
    label="Regression Line"
)

# Add graph details
plt.title("Linear Regression: Nepal MCV1 Immunization Coverage")
plt.xlabel("Year")
plt.ylabel("MCV1 Immunization Coverage (%)")

plt.legend()
plt.grid(True)
plt.show()

# Create future years
future_years = pd.DataFrame({
    "YEAR (DISPLAY)": [2026, 2027, 2028, 2029, 2030]
})

# Predict future coverage
future_predictions = model.predict(future_years)

# Display predictions
for year, prediction in zip(future_years["YEAR (DISPLAY)"], future_predictions):
    print(f"Year: {year}, Predicted MCV1 Coverage: {prediction:.2f}%")

# Check missing values
print("Missing values:")
print(mcv1_data[["YEAR (DISPLAY)", "Numeric"]].isnull().sum())

# Check duplicate rows
print("Duplicate rows:", mcv1_data.duplicated().sum())

# Check data types
print("\nData types:")
print(mcv1_data[["YEAR (DISPLAY)", "Numeric"]].dtypes)

# Check dataset shape
print("\nDataset shape:", mcv1_data.shape)