#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "gas_model.h"

// Pins for sensors (do not use pin 12)
#define MQ4_PIN 14
#define MQ5_PIN 27
#define MQ3B_PIN 26
#define MQ135_PIN 25

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow
// arrays. You'll need to adjust this by combiling, running, and looking
// for errors.
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(gas_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while (1)
      ;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1)
      ;
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.begin(115200);
}

void loop() {
  // Read sensor values
  int mq4Value = analogRead(MQ4_PIN);
  int mq5Value = analogRead(MQ5_PIN);
  int mq3bValue = analogRead(MQ3B_PIN);
  int mq135Value = analogRead(MQ135_PIN);

  // Normalize values ​​between 0 and 1 (like in the training Python script)
  float normalized_mq4 = mq4Value / 4095.0;
  float normalized_mq5 = mq5Value / 4095.0;
  float normalized_mq3b = mq3bValue / 4095.0;
  float normalized_mq135 = mq135Value / 4095.0;

  // Copy values to input buffer
  model_input->data.f[0] = normalized_mq4;
  model_input->data.f[1] = normalized_mq5;
  model_input->data.f[2] = normalized_mq3b;
  model_input->data.f[3] = normalized_mq135;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
  }

  // Read predicted values from output buffer
  float outputAlcohol = model_output->data.f[0];
  float outputClean = model_output->data.f[1];
  float outputPropane = model_output->data.f[2];
  float outputMethane = model_output->data.f[3];
  float outputToxicGas = model_output->data.f[4];



  // Show the results
  Serial.print("MQ-4: ");
  Serial.print(mq4Value);

  Serial.print(", MQ-5: ");
  Serial.print(mq5Value);

  Serial.print(", MQ-3B: ");
  Serial.print(mq3bValue);

  Serial.print(", MQ-135: ");
  Serial.println(mq135Value);

  Serial.println();

  Serial.println("Gas Detected:");

  if (outputMethane > 0.5) {
    Serial.println("- Methane (CH4, C2H5OH, C3H8)");
  }

  if (outputPropane > 0.5) {
    Serial.println("- Propane (C3H8, CH4, C2H5OH)");
  }

  if (outputAlcohol > 0.5) {
    Serial.println("- Alcohol (C2H5OH)");
  }

  if (outputToxicGas > 0.5) {
    Serial.println("- Toxic Gas (H2, NH3, C6H5CH3)");
  }

  if (outputClean > 0.5 && outputMethane < 0.5 && outputPropane < 0.5 && outputAlcohol < 0.5 && outputToxicGas < 0.5) {
    Serial.println("- Clean");
  }

  Serial.println();
  Serial.println();
  Serial.println();

  delay(500);  // Delay for readability
}
