# Function to calculate Hamming distance for categorical data
def hamming_distance(point1, point2):
    distance = 0
    for i in range(len(point1) - 1):  # Exclude class label
        if point1[i] != point2[i]:  # Count mismatches
            distance += 1
    return distance


# Function to find k nearest neighbors
def get_neighbors(training_data, test_instance, k):
    distances = []
    for train_instance in training_data:
        dist = hamming_distance(train_instance, test_instance)  # Compute Hamming distance
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])  # Sort by distance (ascending)
    neighbors = [distances[i][0] for i in range(k)]  # Select k closest neighbors
    return neighbors


# Function to predict class based on majority voting
def predict_classification(training_data, test_instance, k):
    neighbors = get_neighbors(training_data, test_instance, k)
    class_votes = {}

    for neighbor in neighbors:
        label = neighbor[-1]  # Extract class label
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1

    # Find the class with the highest vote
    max_vote = None
    max_count = 0
    for key in class_votes:
        if class_votes[key] > max_count:
            max_count = class_votes[key]
            max_vote = key

    return max_vote  # Return most common class


# Sample dataset (categorical features + class label)
dataset = [
    ["Red", "Small", "BrandX", "Class1"],
    ["Blue", "Medium", "BrandY", "Class2"],
    ["Green", "Large", "BrandZ", "Class1"],
    ["Red", "Medium", "BrandX", "Class1"],
    ["Blue", "Small", "BrandY", "Class2"],
    ["Yellow", "Small", "BrandZ", "Class1"],
    ["Red", "Large", "BrandX", "Class1"],
    ["Blue", "Large", "BrandY", "Class2"],
    ["Green", "Medium", "BrandZ", "Class1"],
    ["Yellow", "Medium", "BrandY", "Class2"],
    ["Red", "Small", "BrandY", "Class2"],
    ["Green", "Small", "BrandX", "Class1"],
    ["Blue", "Medium", "BrandZ", "Class2"],
    ["Yellow", "Large", "BrandX", "Class1"]
]

# Test instance (categorical features, no class label)
test_point = ["Red", "Small", "BrandY"]
k = 3  # Number of neighbors

# Predict the class for the test instance
predicted_class = predict_classification(dataset, test_point, k)
print("Predicted class for", test_point, ":", predicted_class)
