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


# Dataset with full names for color, size, and brand
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

# Mapping from short user input to full dataset values
color_map = {'r': 'Red', 'b': 'Blue', 'g': 'Green', 'y': 'Yellow'}
size_map = {'s': 'Small', 'm': 'Medium', 'l': 'Large'}
brand_map = {'x': 'BrandX', 'y': 'BrandY', 'z': 'BrandZ'}


# Function to prompt user for input and predict class
def main():
    k = 3  # Number of neighbors
    print("Let's predict the product's class based on its details. Please provide the following information:")
    while True:
        print("\nEnter the details of the new product:")

        # Color input with validation
        color = input("Color (r = Red, b = Blue, g = Green, y = Yellow): ").lower()
        while color not in color_map:
            print("Invalid input. Please enter 'r', 'b', 'g', or 'y'.")
            color = input("Color (r = Red, b = Blue, g = Green, y = Yellow): ").lower()

        # Size input with validation
        size = input("Size (s = Small, m = Medium, l = Large): ").lower()
        while size not in size_map:
            print("Invalid input. Please enter 's', 'm', or 'l'.")
            size = input("Size (s = Small, m = Medium, l = Large): ").lower()

        # Brand input with validation
        brand = input("Brand (x = BrandX, y = BrandY, z = BrandZ): ").lower()
        while brand not in brand_map:
            print("Invalid input. Please enter 'x', 'y', or 'z'.")
            brand = input("Brand (x = BrandX, y = BrandY, z = BrandZ): ").lower()

        # Convert the short input to the full values
        full_color = color_map[color]
        full_size = size_map[size]
        full_brand = brand_map[brand]

        # Create the test instance with full values
        test_point = [full_color, full_size, full_brand]
        predicted_class = predict_classification(dataset, test_point, k)
        print(f"Predicted class for {test_point}: {predicted_class}")

        more_predictions = input("\nDo you want to predict another one? (yes/no): ").strip().lower()
        if more_predictions != 'yes':
            print("Goodbye!")
            break


# Run the program
main()
