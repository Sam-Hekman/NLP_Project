import tensorflow as tf


class RNN:
    def __init__(self, input_dim, learning_rate, encoder):
        # Hyperparameters
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Main model
        self.model = self.build_RNN_model(encoder)
        self.model.summary()

    def build_RNN_model(self, encoder):
        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=64, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), ),
            tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3)
        ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(self.learning_rate), metrics=['accuracy'])

        return model

    def train(self, train_dataset, test_dataset, epochs=20, validation_steps=20):
        training_history = self.model.fit(train_dataset, epochs=epochs,
                                          validation_data=test_dataset, validation_steps=validation_steps)
        return training_history
