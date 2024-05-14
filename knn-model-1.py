from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual data)
data = [
    [175, 72, "Male"],  # Height, Weight, Gender
    [168, 65, "Female"],
    [180, 80, "Male"],
    [160, 50, "Female"],
    [172, 68, "Male"],
]

# Extract features (height, weight) and target variable (gender)
X = [[d[0], d[1]] for d in data]  # Features
y = [d[2] for d in data]  # Target variable (gender)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the KNN model with k=3 neighbors (adjust k as needed)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make prediction on a new data point (replace with your values)
new_data = [[170, 60]]  # Height, Weight
prediction = knn.predict(new_data)

# Print the predicted gender
print("Predicted gender:", prediction[0])
