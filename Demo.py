import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

# Load the dataset
file_path = 'diabetes.csv'  # Replace with your local file path if necessary
data = pd.read_csv(file_path)

# Step 1: Replace zero values with NaN in specific columns where zero is not a valid measurement
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)

# Step 2: Impute missing values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Step 3: Separate features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Step 4: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Multi-Class Instance Selection (MCIS)
def mcis_pipeline(X, y, k=3, r=3.5):
    selected_indices = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        # Separate positive and negative class instances
        X_pos = X[y == cls]
        X_neg = X[y != cls]
        
        # K-Means clustering on positive class
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_pos)
        centers = kmeans.cluster_centers_
        
        # Identify boundary instances
        for center in centers:
            distances = np.linalg.norm(X_neg - center, axis=1)
            boundary_indices = np.where(distances < r)[0]
            selected_indices.extend(boundary_indices)
    
    # Combine selected boundary instances with original positive instances
    selected_indices = np.unique(selected_indices)
    X_selected = np.vstack((X, X_neg[selected_indices]))
    y_selected = np.hstack((y, y[y != cls][selected_indices]))
    
    return X_selected, y_selected

# Apply MCIS
X_selected, y_selected = mcis_pipeline(X_scaled, y.values, k=3, r=3.5)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

# Step 7: Ho-Kashyap Algorithm
def ho_kashyap(X_train, y_train, max_iter=500, eta=0.01, b0=1):
    # Prepare data
    y_train = np.where(y_train == 0, -1, 1)  # Convert labels to {-1, 1}
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    A = y_train[:, np.newaxis] * X_train
    b = b0 * np.ones((A.shape[0], 1))

    # Initialize weight vector
    w = np.zeros((A.shape[1], 1))

    for _ in range(max_iter):
        # Compute margin
        e = A @ w - b
        # Update b and w
        b += eta * (e + np.abs(e))
        w = np.linalg.pinv(A) @ b

        # Check for convergence
        if np.all(e >= 0):
            break

    return w

# Train Ho-Kashyap
w_ho_kashyap = ho_kashyap(X_train, y_train)

# Step 8: Predict with Ho-Kashyap
def predict_ho_kashyap(X, w):
    X = np.hstack((X, np.ones((X.shape[0], 1))))  # Add bias term
    predictions = np.sign(X @ w)
    return np.where(predictions == -1, 0, 1)

y_pred = predict_ho_kashyap(X_test, w_ho_kashyap)

# Step 9: Evaluate
print(classification_report(y_test, y_pred))
