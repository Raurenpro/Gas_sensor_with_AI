import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def normalize_data(data):
    # Assuming 4096 is the maximum value the sensors can return
    return data / 4096.0

# Load data from CSV file
csv_filename = "test_data.csv"
test_data = pd.read_csv(csv_filename)

# Separate features and labels
X_test = test_data.drop("labels", axis=1)
y_test = test_data["labels"]

# Perform one-hot coding for labels
mlb = MultiLabelBinarizer()
y_encoded = pd.DataFrame(mlb.fit_transform(
    y_test.apply(eval)), columns=mlb.classes_)

# Normalize test data
X_test_normalized = normalize_data(X_test)

# Load TensorFlow Lite model from file
interpreter = tf.lite.Interpreter(model_path='gas_model.tflite')
interpreter.allocate_tensors()

# Get the indices of the input and output tensors
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

predictions = []

for data_point in X_test_normalized.values:
    # Format the data for the model
    input_data = np.expand_dims(data_point, axis=0).astype(np.float32)

    # Load input data into model
    interpreter.set_tensor(input_index, input_data)

    # Run the model
    interpreter.invoke()

    # Get the results
    output_data = interpreter.get_tensor(output_index)
    predictions.append(output_data)

# Convert predictions to predicted classes
predicted_classes = np.argmax(np.vstack(predictions), axis=1)

# Convert encoded labels to actual classes
actual_classes = np.argmax(y_encoded.values, axis=1)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(actual_classes, predicted_classes)

# Calculate model accuracy
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Show confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=y_encoded.columns)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Matrice de Confusion')
plt.show()