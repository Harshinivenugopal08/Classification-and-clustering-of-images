import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize 
from sklearn.metrics import precision_score, recall_score, f1_score


# Set the path to the directory containing the folders
base_path = r"C:\Users\HARSHINI\Downloads\World Archives_Images\Converted_Images"

# Load the VGG16 model + higher level layers
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features from an image
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess the image
    features = model.predict(image)  # Extract features
    return features.flatten()  # Flatten the features

# List to hold all features and labels
features_list = []
labels_list = []

# Iterate through each folder in the base path
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if os.path.isdir(folder_path):
        # Assuming the folder name is the label
        label = folder_name
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            features = extract_features(image_path)
            features_list.append(features)
            labels_list.append(label)

# Convert features list to numpy array
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Encode labels to integers
le = LabelEncoder()
labels_encoded = le.fit_transform(labels_array)
labels_categorical = np.eye(len(le.classes_))[labels_encoded]  # One-hot encoding

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_array, labels_categorical, test_size=0.2, random_state=42)

# Build a simple neural network for classification
classifier = Sequential()
classifier.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(len(le.classes_), activation='softmax'))  # Output layer

# Compile the model
classifier.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = classifier.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = classifier.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# After making predictions
y_pred = classifier.predict(X_test)

# Get predicted class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true class labels from one-hot encoded y_test
y_true = np.argmax(y_test, axis=1)



# Print shapes and unique values for debugging
print("y_true shape:", y_true.shape)
print("y_pred_classes shape:", y_pred_classes.shape)
print("Unique true classes:", np.unique(y_true))
print("Unique predicted classes:", np.unique(y_pred_classes))
print("Number of classes in le.classes_:", len(le.classes_))
print("Classes from LabelEncoder:", le.classes_)
print("Number of classes in le.classes_:", len(le.classes_))
print("True labels:", y_true)
print("Predicted labels:", y_pred_classes)
print("Unique predicted classes:", np.unique(y_pred_classes))
print("Unique true classes:", np.unique(y_true))

# Generate and print the classification report
unique_classes = np.unique(y_true)  # Get unique classes from true labels
target_names = le.inverse_transform(unique_classes)  # Get target names for the unique classes

# Generate the classification report
report = classification_report(y_true, y_pred_classes, target_names=target_names, labels=unique_classes, zero_division=1)
print("Classification Report:")
print(report)



