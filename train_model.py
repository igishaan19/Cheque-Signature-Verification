from sklearn.svm import SVC
import numpy as np
import joblib

X = [
    [1200, 300, 2.1, 4000, 0.60],
    [1300, 320, 2.0, 4200, 0.65],
    [1250, 310, 2.2, 4100, 0.62],
    [500, 150, 1.2, 1800, 0.30],
    [550, 170, 1.1, 1900, 0.35],
    [530, 160, 1.3, 1850, 0.32]
]

y = [1,1,1,0,0,0]

model = SVC(probability=True)
model.fit(X, y)

joblib.dump(model, "models/signature_model.pkl")

print("Model trained and saved successfully")