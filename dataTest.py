import random
import csv

# Set the number of test samples
num_samples = 100000

# Function to generate a random value for a sensor
def generate_sensor_value():
    x = random.randint(0, 4095)
    # if (x <= 3000):
    #     x = x//2
    return x

# Function to generate test data with labels
def generate_test_data():
    data = []
    for _ in range(num_samples):
        mq4_value = generate_sensor_value()
        mq5_value = generate_sensor_value()
        mq3b_value = generate_sensor_value()
        mq135_value = generate_sensor_value()

        labels = []

        if mq4_value > 3000:
            labels.append("Propane")

        if mq5_value > 3000:
            labels.append("Methane")

        if mq3b_value > 3000:
            labels.append("Alcohol")

        if mq135_value > 3000:
            labels.append("Toxic Gas")

        if not labels:
            labels.append("Clean")

        data.append({
            "MQ4": mq4_value,
            "MQ5": mq5_value,
            "MQ3B": mq3b_value,
            "MQ135": mq135_value,
            "labels": labels
        })

    return data

# Generate test data
test_data = generate_test_data()

# Set CSV file name
csv_filename = "test_data.csv"

# Write data to CSV file
with open(csv_filename, mode='w', newline='') as csv_file:
    fieldnames = ["MQ4", "MQ5", "MQ3B", "MQ135", "labels"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write CSV file header
    writer.writeheader()

    # Write data to CSV file
    for data_point in test_data:
        writer.writerow(data_point)