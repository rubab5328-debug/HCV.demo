# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. APNA ACTUAL HCV DATA YAHAN LOAD KARO
# data = pd.read_csv('tumhara_hcv_data.csv')

# 2. Agar data nahi hai toh TEMPORARY ke liye sample data use karo:
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# 3. Model train karo
model = RandomForestClassifier()
model.fit(X, y)

# 4. Model save karo
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl!")