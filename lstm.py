import numpy as np
import tensorflow as tf

from data_gen import FUNCS


if __name__ == "__main__":
    func = FUNCS["bitcount"]
    x_train = np.random.randn(50000, 20)
    y_train = np.vstack(map(func, x_train, [True] * len(x_train)))
    x_test = np.random.randn(10000, 20)
    y_test = np.vstack(map(func, x_test, [True] * len(x_test)))

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(20,)),
            tf.keras.layers.Reshape((1, -1)),
            tf.keras.layers.LSTM(
                units=100,
            ),
            tf.keras.layers.Dense(3),
        ]
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=100)

    print("++++++++++++++++++++++++++++++++++")
    model.evaluate(x_test, y_test, verbose=2)
    print("++++++++++++++++++++++++++++++++++")

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    print(probability_model(x_test[:10]))
    print(y_test[:10])
