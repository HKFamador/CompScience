# Step 1: Initialize the centroids randomly (without using random or math packages)
def initialize_centroids(data, k):
    centroids = []
    data_copy = data[:]
    for i in range(k):
        index = i % len(data_copy)  # Pick centroids in a round-robin fashion if there is no random
        centroids.append(data_copy[index])
    return centroids


# Step 2: Assign each data point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]

    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid_index = min(range(len(distances)), key=distances.__getitem__)
        clusters[closest_centroid_index].append(point)

    return clusters


# Step 3: Calculate the Euclidean distance between two points (manually)
def euclidean_distance(point1, point2):
    sum_squared_diffs = 0
    for i in range(len(point1)):
        sum_squared_diffs += (point1[i] - point2[i]) ** 2
    return sum_squared_diffs ** 0.5


# Step 4: Recalculate centroids
def update_centroids(clusters):
    centroids = []

    for cluster in clusters:
        new_centroid = []
        for i in range(len(cluster[0])):  # assuming all points have the same number of dimensions
            dimension_values = [point[i] for point in cluster]
            new_centroid.append(sum(dimension_values) / len(cluster))
        centroids.append(new_centroid)

    return centroids


# Step 5: Check if centroids have changed
def has_converged(old_centroids, new_centroids, tolerance=1e-4):
    for old, new in zip(old_centroids, new_centroids):
        diff = euclidean_distance(old, new)
        if diff > tolerance:
            return False
    return True


# K-means clustering function
def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        clusters = assign_clusters(data, centroids)

        # Save the old centroids to check for convergence later
        old_centroids = centroids[:]

        # Update centroids
        centroids = update_centroids(clusters)

        # If centroids have converged, stop the algorithm
        if has_converged(old_centroids, centroids):
            print(f"Converged after {iteration + 1} iterations.")
            break
    return centroids, clusters


# Example data points (2D points)
# Example data points (2D points)
data_points = [[1, 2], [1, 3], [2, 2], [8, 8], [9, 9], [10, 8]]

# Number of clusters
k = 3

# Running the K-means algorithm
centroids, clusters = k_means(data_points, k)

# Output the results
print("\nFinal Centroids:")
for centroid in centroids:
    print(centroid)

print("\nClusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")

