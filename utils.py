import os
import gzip
import json
import re
import matplotlib.pyplot as plt
import pandas as pd

from ProcessText import ProcessData, PerformStemming, PerformLemmatization


# Unloading a JSON at each experiment run adds a lot of time overhead.
# This adds an extra step and increases storage usage, but helps speed up the training.

# Retrieve processed (lemmatizing is last step) dataset, run processing if it does not yet exist
def get_data_as_dataframe():

    # If already processed
    if os.path.exists("datasetProcessing\\processed_data.csv"):
        df = pd.read_csv("datasetProcessing\\processed_data.csv")
        return df

    else:
        # lemmatization was performed, but not yet converted and saved as csv
        if os.path.exists("datasetProcessing\\lemmatized.json.gz"):
            with gzip.open("datasetProcessing\\lemmatized.json.gz", 'r') as fin:
                data = json.loads(fin.read().decode('utf-8'))
                df = convert_to_dataframe(data)
                df.to_csv("datasetProcessing\\processed_data.csv", index=False)
            return df

        else:
            ProcessData()
            PerformStemming()
            PerformLemmatization()

            with gzip.open("datasetProcessing\\lemmatized.json.gz", 'r') as fin:
                data = json.loads(fin.read().decode('utf-8'))
                df = convert_to_dataframe(data)
                df.to_csv("datasetProcessing\\processed_data.csv", index=False)
            return df


# Helper function for get_data_as_dataframe
# Convert to dataframe for easier manipulation
# Also convert numerical review 'stars' to labels
def convert_to_dataframe(data):

    category = []
    text = []
    label = []

    for key in data.keys():
        for review in data[key]:
            category.append(key)
            text.append(stringify_and_remove_specials(review[0]))
            if review[1] <= 2.0:
                label.append(0)
            elif review[1] == 5.0:
                label.append(2)
            else:
                label.append(1)

    df = pd.DataFrame({"Category": category, "Text": text, "Label": label})
    return df


# Helper function for convert to dataframe
def stringify_and_remove_specials(text):

    # Declare empty string for text
    s = ''
    for i in range(len(text)):

        # if not a special character
        if not re.match('[^a-zA-Z0-9|.|,|!]+', text[i]):

            # Start of text
            if i == 0:
                s += text[i]
            else:
                # is punctuation
                if text[i] in ['.', ',', '!']:
                    s += text[i]
                else:
                    s += (" " + text[i])
    return s

# graphs for training process
def get_training_graph(history, model_name):
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    x = [item for item in range(len(accuracy)+1) if item != 0]

    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(12, 8))
    plt.plot(x, accuracy, 'b-*', label="Training Accuracy")
    plt.plot(x, val_accuracy, 'r-*', label="Validation Accuracy")
    plt.legend()
    plt.title("{} Training Accuracy".format(model_name))
    plt.xticks(rotation=45)
    plt.xlabel("Training Epoch")
    plt.ylabel("Percentage")
    plt.grid(linestyle='dashed')
    plt.show()

    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(12, 8))
    plt.plot(x, loss, 'g-*', label="Training loss")
    plt.plot(x, val_loss, '-*', color='orange', label="Validation loss")
    plt.legend()
    plt.title("LSTM-RNN Training Loss")
    plt.xticks(rotation=45)
    plt.xlabel("Training Epochs")
    plt.ylabel("Percentage")
    plt.grid(linestyle='dashed')
    plt.show()
