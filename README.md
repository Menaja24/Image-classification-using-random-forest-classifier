# Image-classification-using-random-forest-classifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import io, color

# Step 1: Load and preprocess your image dataset
image_paths = [...]  # List of image file paths
labels = [...]  # Corresponding class labels

# Step 2: Feature Extraction (HOG feature extraction in this example)
def extract_features(image_path):
    image = io.imread(image_path)  # Load image
    gray_image = color.rgb2gray(image)  # Convert to grayscale
    features, _ = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)  # Compute HOG features
    return features

# Extract features from all images
X = np.array([extract_features(path) for path in image_paths])
y = np.array(labels)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = rf_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
