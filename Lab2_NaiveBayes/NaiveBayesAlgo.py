class NaiveBayes:
    def __init__(self):
        self.probabilities = {}
        self.class_counts = {}
        self.total_samples = 0

    def train(self, X, y):
        self.total_samples = len(y)
        unique_classes = set(y)

        for c in unique_classes:
            self.class_counts[c] = y.count(c)
            self.probabilities[c] = {}

            for i in range(len(X[0])):
                self.probabilities[c][i] = {}

                feature_values = [X[j][i] for j in range(len(X)) if y[j] == c]
                unique_values = set(feature_values)

                for v in unique_values:
                    self.probabilities[c][i][v] = feature_values.count(v) / self.class_counts[c]

    def predict(self, X):
        predictions = []

        for sample in X:
            class_probs = {}

            for c in self.class_counts:
                class_probs[c] = self.class_counts[c] / self.total_samples

                for i in range(len(sample)):
                    value = sample[i]
                    if value in self.probabilities[c][i]:
                        class_probs[c] *= self.probabilities[c][i][value]
                    else:
                        class_probs[c] *= 1e-6  # Smoothing to avoid zero probability

            best_class = max(class_probs, key=class_probs.get)
            predictions.append(best_class)

        return predictions


# Sample training data (age, income level, education, buy decision)
X_train = [
    ['young', 'low', 'highschool'],
    ['young', 'low', 'college'],
    ['middle-aged', 'medium', 'college'],
    ['middle-aged', 'high', 'graduate'],
    ['old', 'high', 'graduate'],
    ['old', 'medium', 'highschool'],
    ['young', 'medium', 'college'],
    ['young', 'low', 'highschool'],
    ['middle-aged', 'high', 'graduate'],
    ['old', 'medium', 'highschool']
]

y_train = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no']

# Create and train the Naive Bayes model
nb = NaiveBayes()
nb.train(X_train, y_train)

# Asking the user for input
def ask_user_for_prediction():
    print("Let's predict if the person will buy the product.")
    age_group = input("Enter age group (young, middle-aged, old): ").strip().lower()
    income_level = input("Enter income level (low, medium, high): ").strip().lower()
    education = input("Enter education level (highschool, college, graduate): ").strip().lower()

    sample = [[age_group, income_level, education]]
    prediction = nb.predict(sample)
    print(f"[Prediction] Will the person buy the product? : {prediction[0]}")


# Ask the user if they want to make another prediction
def ask_another_prediction():
    while True:
        ask_user_for_prediction()
        another = input("Do you want to predict another? (yes/no): ").strip().lower()
        if another != 'yes':
            print("Thank you for using the prediction system!")
            break

# Start the prediction process
ask_another_prediction()
