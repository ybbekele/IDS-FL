import argparse
import os
from PIL import Image
from PIL import ExifTags
from sklearn.preprocessing import StandardScaler
from flwr.client import ClientApp, NumPyClient
import tensorflow as tf
from flwr_datasets import FederatedDataset
import numpy as np
import random
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from Fault_Injector import inject_fault
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
from sklearn import preprocessing
import sys
import json
import struct
import seaborn as sns
import matplotlib.pyplot as plt
import utils_cav
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse arguments
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=[0, 1, 2, 3],
    default=0,
    help="Partition of the dataset (0, 1, 2 or 3). "
    "The dataset is divided into 3 partitions created artificially.",
)
parser.add_argument(
    "--client-id",
    type=int,
    required=True,
    help="Unique client ID for distinguishing between different clients.",
)
args, _ = parser.parse_known_args()

def count_parameters(model):
    weights_count = 0
    biases_count = 0
    activations_count = 0
    
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            weights_count += np.prod(weights[0].shape)
            biases_count += np.prod(weights[1].shape)
        if hasattr(layer, 'activation'):
            activations_count += 1
    
    return weights_count, biases_count, activations_count
    
(x_train, y_train), (x_test, y_test) = utils_cav.load_mnist()
x_train = preprocessing.scale(x_train)
x_train = preprocessing.normalize(x_train)
x_test = preprocessing.scale(x_test)
x_test = preprocessing.normalize(x_test)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# Load model and data (MobileNetV2, CIFAR-10)
model = keras.Sequential([
    keras.layers.Conv2D(96,(4,4),input_shape=(x_train.shape[1],x_train.shape[2],1),activation='relu',padding='same'),
    keras.layers.Conv2D(64,(3,3),activation="relu",padding='same'),
    keras.layers.Conv2D(32,(2,2),activation="relu",padding='same'),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation="relu"),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(32,activation="relu"),
    keras.layers.Dense(2,activation="softmax"),
    ])

model.compile("adam", "sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
model.summary()

weights_count, biases_count, activations_count = count_parameters(model)
print(f"Functional Model - Weights: {weights_count}, Biases: {biases_count}, Activations: {activations_count}")

# Function to convert a float to its binary representation
def float_to_binary(value):
    """Convert float to binary representation."""
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return f'{d:064b}'

# Function to print the size in bits of each weight and bias in the model
def print_weights_and_biases_size(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            weights = layer.kernel.numpy().flatten()
            for i, weight in enumerate(weights):
                binary_rep = float_to_binary(weight)
                print(f"Layer: {layer.name}, Weight {i}: Size (in bits) = {len(binary_rep)}")
        if hasattr(layer, 'bias'):
            biases = layer.bias.numpy().flatten()
            for i, bias in enumerate(biases):
                binary_rep = float_to_binary(bias)
                print(f"Layer: {layer.name}, Bias {i}: Size (in bits) = {len(binary_rep)}")

# Print the size in bits of each weight and bias in the model
#print_weights_and_biases_size(model)
# Train the model and save the original model
#model.fit(train_images, train_labels, epochs=5, validation_split=0.1)
#original_model_path = './original_model'
#os.makedirs(original_model_path, exist_ok=True)
#model.save(os.path.join(original_model_path, 'original_model.h5'))

# Evaluate the original model before fault injection
loss_before, accuracy_before = model.evaluate(x_test, y_test)
print(f'Initial Accuracy: {accuracy_before:.4f}, Initial Loss: {loss_before:.4f}')
# Initialize lists to store results
accuracies_before = [accuracy_before]
losses_before = [loss_before]
precisions_before = []
recalls_before = []
f1_scores_before = []

accuracies_after = []
losses_after = []
precisions_after = []
recalls_after = []
f1_scores_after = []
accuracy_degradation = []
loss_increase = []

sdc_counts = []
total_sdc = 0
client_metrics = {
    "round": [],
    "loss_before": [],
    "accuracy_before": [],
    }
# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print(f'Batch size is:{config["batch_size"]}')  # Prints `32`
        print(f'Current Round is: {config["current_round"]}')  # Prints `1`/`2`/`...`
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        #original_model_path = './original_model'
        #os.makedirs(original_model_path, exist_ok=True)
        #model.save(os.path.join(original_model_path, 'original_model.h5'))
        # Evaluate the original model before fault injection
        loss_before, accuracy_before = model.evaluate(x_test, y_test)
        print(f'Accuracy before fault injection: {accuracy_before:.4f}, Loss before fault injection: {loss_before:.4f}')
        accuracies_before = [accuracy_before]
        losses_before = [loss_before]
        # Store metrics before fault injection
        server_round = config.get("current_round", 0)  # Use 0 as default if not provided
        client_metrics["round"].append(config["current_round"])
        client_metrics["loss_before"].append(loss_before)
        client_metrics["accuracy_before"].append(accuracy_before)
        # Calculate precision, recall, and F1-score before fault injection
        y_pred_before = model.predict(x_test)
        y_pred_classes_before = np.argmax(y_pred_before, axis=1)
        #precision_before = precision_score(y_test, y_pred_classes_before, average='macro')
        #recall_before = recall_score(y_test, y_pred_classes_before, average='macro')
        #f1_before = f1_score(y_test, y_pred_classes_before, average='macro')

        #precisions_before.append(precision_before)
        #recalls_before.append(recall_before)
        #f1_scores_before.append(f1_before)
        # Prompt the user for fault injection options
        #fault_type = input("Enter fault type (weight or bias): ").strip().lower()
        #fault_model = input("Enter fault model (1: single, 2: double, 3: byte): ").strip().lower()
        #fault_type = "weight"
        #fault_model = "1"
        # Load the original model for each round of fault injections
        #original_model = models.load_model(os.path.join(original_model_path, 'original_model.h5'))
        # Re-inject faults into the model 
        #inject_fault(model, fault_type, fault_model, [])
       # print(f'Fault injected and parameters changed succesfully')
        #loss_after, accuracy_after = model.evaluate(x_test, y_test)
        #print(f'Accuracy after FI local evaluation: {accuracy_after:.4f}, Loss after FI local evaluation: {loss_after:.4f}')
        #accuracies_after.append(accuracy_after)
        #losses_after.append(loss_after)

        # Store metrics after fault injection
        #client_metrics["loss_after"].append(loss_after)
        #client_metrics["accuracy_after"].append(accuracy_after)

        #y_pred_after = model.predict(x_test)
        #y_pred_classes_after = np.argmax(y_pred_after, axis=1)
        #accuracy_degradation.append(abs(accuracy_before - accuracy_after))
        #loss_increase.append(abs(loss_before - loss_after))
        #precision_after = precision_score(y_test, y_pred_classes_after, average='macro')
        #recall_after = recall_score(y_test, y_pred_classes_after, average='macro')
        #f1_after = f1_score(y_test, y_pred_classes_after, average='macro')
        #precisions_after.append(precision_after)
        #recalls_after.append(recall_after)
        #f1_scores_after.append(f1_after)

        # Calculate silent data corruption (SDC)
        #sdc_local = 0
       # for i in range(len(y_test)):
        #    if y_pred_classes_before[i] == y_test[i] and y_pred_classes_after[i] != y_test[i]:
        #        sdc_local += 1
        #sdc_counts.append(sdc_local)
        #total_sdc += sdc_local
        #print(f'SDC count is: {total_sdc}')
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        num_examples = len(x_test)
        return loss, num_examples, {"accuracy": accuracy, "loss": loss}

# Save client metrics to a file after all rounds are completed
def save_client_metrics(client_id):
    filename = f'client_metrics_{client_id}.json'
    with open(filename, 'w') as f:
        json.dump(client_metrics, f)


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    ) 

# Save client metrics when done
save_client_metrics(args.client_id)