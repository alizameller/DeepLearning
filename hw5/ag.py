from datasets import load_dataset
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

dataset = load_dataset("ag_news")

# Extract sentences and labels from the dataset
sentences = dataset["train"]["text"]
labels = dataset["train"]["label"]

# Split the data into training and validation sets
sentences_train, sentences_val, labels_train, labels_val = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Load the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode the sentences into embeddings using the SentenceTransformer model
train_embeddings = model.encode(sentences_train, show_progress_bar=True)
val_embeddings = model.encode(sentences_val, show_progress_bar=True)

labels_train = np.array(labels_train)
labels_val = np.array(labels_val)

num_classes = 4

# Create embeddings of test set
sentences_test = dataset["test"]["text"]
labels_test = dataset["test"]["label"]
labels_test = np.array(labels_test)

test_embeddings = model.encode(sentences_test, show_progress_bar=True)

# Define a simple feedforward neural network for classification
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(train_embeddings.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')  
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_embeddings, labels_train, validation_data=(val_embeddings, labels_val), epochs=10, batch_size=64)

# Make predictions on the test set
predictions = model.predict(test_embeddings)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy and generate a classification report
accuracy = accuracy_score(labels_test, predicted_labels)
classification_rep = classification_report(labels_test, predicted_labels)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)
