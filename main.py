import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from matplotlib import pyplot as plt


def normalize_data(data):
    # Assuming 4096 is the maximum value the sensors can return
    return data / 4095.0

# Function: Convert some hex value into an array for C programming


def hex_to_c_array(hex_data, var_name):

    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Add array length at top of file
    c_str += '\nunsigned int ' + var_name + \
        '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data):

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str


# Load data from CSV file
csv_filename = "test_data.csv"
data = pd.read_csv(csv_filename)

# Separate features and labels
X = data.drop("labels", axis=1)
y = data["labels"]

# Perform one-hot coding for labels
mlb = MultiLabelBinarizer()
y_encoded = pd.DataFrame(mlb.fit_transform(
    y.apply(eval)), columns=mlb.classes_)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Data normalization
X_train_normalized = normalize_data(X_train)
X_test_normalized = normalize_data(X_test)

model = models.Sequential()
model.add(layers.Dense(8, input_shape=(4,), activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# To save workout history
history = model.fit(
    X_train_normalized, y_train,
    epochs=300, batch_size=1000,
    validation_data=(X_test_normalized, y_test),
)

# Model evaluation on test set
loss, accuracy = model.evaluate(X_test_normalized, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


# Convert model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to disk
with open('gas_model' + '.tflite', 'wb') as f:
    f.write(tflite_model)


# Write TFLite model to a C source (or header) file
with open('gas_model' + '.h', 'w') as file:
    file.write(hex_to_c_array(tflite_model, 'gas_model'))


# Plot accuracy on training and validation set across epochs
plt.figure(figsize=(12, 6))

# Subgraph for Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.text(len(history.history['accuracy']) - 1, history.history['accuracy'][-1],
         f'Final Accuracy: {history.history["accuracy"][-1] * 100:.2f}%', ha='right', va='bottom')

# Subchart for loss
plt.subplot(1, 2, 2)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

# Show figure
plt.tight_layout()  # To avoid overlaps
plt.show()
