import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Select only the 5 features used in the Streamlit app
features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany']
X = df[features]
y = df['Attrition'].map({'Yes': 1, 'No': 0})  # Encode target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = SVC(probability=True)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


