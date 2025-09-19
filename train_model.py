import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("career_data.csv")

# Encode categorical columns
le_interest = LabelEncoder()
le_brain = LabelEncoder()
le_career = LabelEncoder()

df['Interest'] = le_interest.fit_transform(df['Interest'])
df['Brain'] = le_brain.fit_transform(df['Brain'])
df['Career'] = le_career.fit_transform(df['Career'])

# Save encoders for later use
joblib.dump(le_interest, "interest_encoder.pkl")
joblib.dump(le_brain, "brain_encoder.pkl")
joblib.dump(le_career, "career_encoder.pkl")

# Train model
X = df[['IQ', 'Interest', 'Brain']]
y = df['Career']
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "career_model.pkl")

print("✅ Model training complete. Files saved: career_model.pkl + encoders")
