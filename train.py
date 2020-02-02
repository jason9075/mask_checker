import pathlib

import numpy as np
import tensorflow as tf

TRAIN_DATA_PATH = 'dataset/mask_dataset/train/'
TEST_DATA_PATH = 'dataset/mask_dataset/test/'
OUTPUT_MODEL_FOLDER = 'exported/'
CLASS_NAMES = np.array([])
AUTOTUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 5
IMG_SHAPE = (112, 112, 3)
BATCH_SIZE = 40


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    one_hot = tf.cast(parts[-2] == CLASS_NAMES, tf.float32)
    return one_hot


def pre_process_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)
    return img


def process_path(file_path):
    label = get_label(file_path)
    img = pre_process_img(file_path)
    return img, label


def show_label(label_list):
    for idx, label in enumerate(label_list):
        print(f'{idx} : {label}')


def main():
    global CLASS_NAMES

    train_data_dir = pathlib.Path(TRAIN_DATA_PATH)
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    print(f'train img count:{train_image_count}')
    test_data_dir = pathlib.Path(TEST_DATA_PATH)
    test_image_count = len(list(test_data_dir.glob('*/*.jpg')))
    print(f'train img count:{test_image_count}')
    CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name not in [".keep", ".DS_Store"]])
    show_label(CLASS_NAMES)

    train_data_set = tf.data.Dataset.list_files(str(train_data_dir / '*/*.jpg'))
    train_data_set = train_data_set.map(process_path, num_parallel_calls=AUTOTUNE)
    train_data_set = train_data_set.shuffle(buffer_size=2000)
    train_data_set = train_data_set.repeat()
    train_data_set = train_data_set.batch(BATCH_SIZE)
    train_data_set = train_data_set.prefetch(buffer_size=AUTOTUNE)

    test_data_set = tf.data.Dataset.list_files(str(test_data_dir / '*/*.jpg'))
    test_data_set = test_data_set.map(process_path, num_parallel_calls=AUTOTUNE)
    test_data_set = test_data_set.batch(BATCH_SIZE)

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='sigmoid'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    save_cb = tf.keras.callbacks.ModelCheckpoint(OUTPUT_MODEL_FOLDER,
                                                 monitor='val_loss',
                                                 save_weights_only=False,
                                                 save_best_only=True,
                                                 mode='auto',
                                                 verbose=1)
    model.fit(train_data_set,
              epochs=EPOCHS,
              steps_per_epoch=train_image_count // BATCH_SIZE,
              validation_data=test_data_set,
              callbacks=[save_cb])


if __name__ == '__main__':
    main()
