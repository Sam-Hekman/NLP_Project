import numpy as np
import tensorflow as tf
print(tf.version.VERSION)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

from utils import get_data_as_dataframe, get_training_graph
from RNN_model import RNN

if __name__ == "__main__":

    # Declare variables for train-test
    seed = 30
    test_size = 0.3

    # Buffer and batch sizes
    batch_size = 64
    buffer_size = 10000

    # Size of encoder vocabulary and subsequent dimension of NN input
    vocab_size = 6000

    # Hyperparameters for LSTM-RNN
    epochs = 2 # 30
    validation_steps = 50
    learning_rate = 1e-4

    # Get data, runs pre-processing if not already completed
    df = get_data_as_dataframe()

    # Train test split, keeps categories to ensure even stratification between them
    X_train, X_test, y_train, y_test = train_test_split(df.drop("Label", axis=1), df["Label"],
                                                        random_state=seed, test_size=test_size)

    # Convert texts to tensors, remove categories
    X_train = tf.convert_to_tensor(np.array(X_train['Text']).astype('str'))
    X_test = tf.convert_to_tensor(np.array(X_test['Text']).astype('str'))

    # Encode labels to one hot, convert to tensors
    y_train = tf.convert_to_tensor(to_categorical(y_train))
    y_test_for_pred = tf.convert_to_tensor(y_test)
    y_test = tf.convert_to_tensor(to_categorical(y_test))

    # Zip training text and labels to dataset object, shuffle buffer and get batch
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Do the same for the test set
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build encoder
    # Makes vectorized form of text inputs
    encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    # Get vocabulary length for input dimension for first embedding layer of RNN
    vocab_len = len(encoder.get_vocabulary())

    # Build RNN model
    RNN_model = RNN(vocab_len, learning_rate, encoder)

    training_history = RNN_model.train(train_dataset, test_dataset, epochs, validation_steps)

    y_pred = RNN_model.model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    print(classification_report(y_test_for_pred, y_pred))

    get_training_graph(training_history, "LSTM-RNN")


    # np.save('training_history_final.npy', training_history.history)
    # RNN_model.model.save("RNN_model_final")