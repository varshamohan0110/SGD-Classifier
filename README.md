# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Set initial weights (theta) to zero.
2. Compute Predictions: Calculate predictions using the sigmoid function on the weighted inputs.
3. Calculate Cost: Compute the cost using the cross-entropy loss function.
4. Update Weights: Adjust weights by subtracting the gradient of the cost with respect to each weight.
5.  Repeat: Repeat steps 2–4 for a set number of iterations or until convergence is achieved. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Load Dataset
data = pd.read_csv("SDG.csv")

# 3️⃣ Separate Features and Target
X = data.drop(["status", "salary", "sl_no"], axis=1)
y = data["status"]   # Placed / Not Placed

# 4️⃣ Convert Categorical to Numerical (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7️⃣ Create SGD Classifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# 8️⃣ Train Model
model.fit(X_train, y_train)

# 9️⃣ Predict
y_pred = model.predict(X_test)

# 🔟 Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 1️⃣1️⃣ Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 1️⃣2️⃣ Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 1️⃣3️⃣ Predict for New Student (Example)
new_student = [[67, 91, 58, 0, 1, 1, 1, 0, 0, 88, 1, 67]]
new_student = scaler.transform(new_student)

pred = model.predict(new_student)
print("\nPredicted Status:", pred[0])


Developed by: Varsha M
RegisterNumber:  25001006
*/
```

## Output:
<img width="775" height="394" alt="image" src="https://github.com/user-attachments/assets/0e7cda75-8326-41fb-9770-172c1fe7f165" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
