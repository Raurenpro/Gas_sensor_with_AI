# Gas Sensor with AI

## Overview

This project utilizes an ESP32 microcontroller along with various gas sensors (DFRobot's MQ4, MQ5, MQ3B, and Flying-fish's MQ135) to detect and classify gases using a trained machine learning model. The model is implemented in TensorFlow Lite and runs on the ESP32, allowing real-time gas detection.

## Project Structure

- **dataTest.py**: Python script to generate synthetic test data for training the machine learning model. The generated data is stored in a CSV file named `test_data.csv`.

- **gas_model.h**: Contains the trained TensorFlow Lite machine learning model.

- **Gas_sensor_with_ai.ino**: Arduino sketch for ESP32, responsible for reading sensor values, normalizing them, running inference using the TensorFlow Lite model, and displaying the results.

- **main.py**: Python script for training the machine learning model. It loads data from `test_data.csv`, normalizes it, creates a neural network, compiles the model, trains it, converts it to TensorFlow Lite format, and saves both the model and a C header file. **(Accuracy is around 88%)**

- **testModel.py**: Python script to test the trained model using the test data in `test_data.csv`. It loads the TensorFlow Lite model, performs inference on the test data, calculates accuracy, and displays a confusion matrix.

## Prerequisites

Before running the project, ensure the following:

1. Create a virtual environment using `venv` (use appropriate commands for Linux/Windows).
   
    ```bash
    # Linux
    python3 -m venv venv

    # Windows
    python -m venv venv
    ```

2. Activate the virtual environment and install required libraries.

    ```bash
    # Linux
    source venv/bin/activate

    # Windows
    .\venv\Scripts\activate

    # Install required libraries
    pip install -r requirements.txt
    ```

3. Connect DFRobot's MQ4, MQ5, MQ3B, and Flying-fish's MQ135 sensors to specified pins on the ESP32.

4. Use Arduino IDE for uploading the project to ESP32.

5. Download the [TensorFlowLite_ESP32](https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32) library from Arduino IDE or the provided link.

## Usage

1. Upload the `Gas_sensor_with_ai.ino` sketch to the ESP32 using Arduino IDE.

2. Ensure the sensors are connected correctly and the ESP32 is powered.

3. Open the Serial Monitor in Arduino IDE to view gas detection results.

## Additional Notes

- The `Gas_sensor_with_ai.ino` sketch assumes specific pin configurations for each gas sensor. Make sure to update the pin definitions if you modify the hardware setup.

- Training and converting the machine learning model is performed using the `main.py` script. Adjust the script as needed, considering changes in the data or model architecture.

- The `testModel.py` script validates the accuracy of the trained model using the test data. Make sure the model file (`gas_model.tflite`) is present before running the script.
