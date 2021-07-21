import tensorflow as tf


def build_model(class_num: int,
                image_size: int,
                learning_rate: float,
                activation_func: str,
                dense_units: list,
                conv2d_units: list,
                conv2d_kernels: list,
                pool_sizes: list):

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size)
    )

    model.add(
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    )

    for i, (units, kernel_size, pool_size) in enumerate(zip(conv2d_units, conv2d_kernels, pool_sizes)):
        if i == 0:
            model.add(
                tf.keras.layers.Conv2D(units,
                                       (kernel_size, kernel_size),
                                       activation=activation_func,
                                       input_shape=(image_size, image_size))
            )

        model.add(
            tf.keras.layers.Conv2D(units, (kernel_size, kernel_size), activation=activation_func)
        )

        model.add(
            tf.keras.layers.MaxPooling2D((pool_size, pool_size))
        )

    model.add(
        tf.keras.layers.Flatten()
    )

    for units in dense_units:
        model.add(
            tf.keras.layers.Dense(units, activation=activation_func),
        )

    model.add(
        tf.keras.layers.Dense(class_num, activation='softmax')
    )

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )
    return model
