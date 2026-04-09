# D:\BreatheAI\utils\train_pipeline_model.py

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- 1. Simulated dataset for demonstration ---
# Features: AQI + 6 symptom severity values (1–5)
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'aqi': np.random.randint(50, 300, n_samples),
    'cough_severity': np.random.randint(1, 6, n_samples),
    'wheezing_severity': np.random.randint(1, 6, n_samples),
    'breathlessness_severity': np.random.randint(1, 6, n_samples),
    'chest_pain_severity': np.random.randint(1, 6, n_samples),
    'throat_irritation_severity': np.random.randint(1, 6, n_samples),
    'fatigue_severity': np.random.randint(1, 6, n_samples)
})

# Binary target: 0 = tolerant, 1 = intolerant (fake logic)
data['tolerance_label'] = (data['aqi'] > 150).astype(int)

# --- 2. Split features and target ---
X = data.drop(columns=['tolerance_label'])
y = data['tolerance_label']

# --- 3. Define preprocessing + model pipeline ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

# --- 4. Train the model ---
pipeline.fit(X, y)

# --- 5. Save model ---
joblib.dump(pipeline, "pipeline_model.pkl")

# --- 6. Save preprocessing columns ---
preprocessing_columns = list(X.columns)
joblib.dump(preprocessing_columns, "preprocessing_columns.pkl")

print("✅ pipeline_model.pkl and preprocessing_columns.pkl saved successfully!")
