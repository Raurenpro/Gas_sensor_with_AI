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

# Séparez les fonctionnalités et les étiquettes
X_test = test_data.drop("labels", axis=1)
y_test = test_data["labels"]

# Perform one-hot coding for labels
mlb = MultiLabelBinarizer()
y_encoded = pd.DataFrame(mlb.fit_transform(
    y_test.apply(eval)), columns=mlb.classes_)

# Normalisez les données de test
X_test_normalized = normalize_data(X_test)

# Charger le modèle TensorFlow Lite à partir du fichier
interpreter = tf.lite.Interpreter(model_path='gas_model.tflite')
interpreter.allocate_tensors()

# Obtenez les indices des tenseurs d'entrée et de sortie
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

predictions = []

for data_point in X_test_normalized.values:
    # Mettez en forme les données pour le modèle
    input_data = np.expand_dims(data_point, axis=0).astype(np.float32)

    # Chargez les données d'entrée dans le modèle
    interpreter.set_tensor(input_index, input_data)

    # Exécutez le modèle
    interpreter.invoke()

    # Obtenez les résultats
    output_data = interpreter.get_tensor(output_index)
    predictions.append(output_data)

# Convertir les prédictions en classes prédites
predicted_classes = np.argmax(np.vstack(predictions), axis=1)

# Convertir les étiquettes encodées en classes réelles
actual_classes = np.argmax(y_encoded.values, axis=1)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(actual_classes, predicted_classes)

# Calculer la précision du modèle
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=y_encoded.columns)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Matrice de Confusion')
plt.show()