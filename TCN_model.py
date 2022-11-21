import tensorflow as tf
from tcn import TCN, tcn_full_summary
from keras import callbacks

class TCN_model:
    def __init__(self, input_dim, learning_rate, encoder):
        # Hyperparameters
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        # Main model
        self.model = self.build_TCN_model(encoder)
        self.model.summary()


    def build_TCN_model(self, encoder):
        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=64, mask_zero=True),

            TCN(64, dilations=[1, 2, 4], return_sequences=True, activation='relu', name='tcn1'),
            TCN(32, dilations=[1, 2, 4], return_sequences=True, activation='relu', name='tcn2'),

            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True), ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8), ),
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
