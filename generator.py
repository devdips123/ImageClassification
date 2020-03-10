#! /usr/bin/env python3

import tensorflow as tf
import os
import numpy as np

keras = tf.keras

IMAGE_SIZE = (228, 228, 3)
TEST_IMAGE_PATH = ""
label_columns = []


def image_generator(testing_df):
    indices = testing_df.index
    for i in indices:
        f = testing_df.iloc[i]['filename']
        label = testing_df.iloc[i][label_columns].values.astype(np.float32)
        filename = os.path.join(TEST_IMAGE_PATH, f)
        img = keras.preprocessing.image.load_img(filename, target_size=IMAGE_SIZE)
        img = keras.preprocessing.image.img_to_array(img)
        img = keras.applications.resnet.preprocess_input(img)
        img = img / 255
        img = np.expand_dims(img, axis=0)

        yield img, label


class MyGenerator:
    index = 0

    def __init__(self, data, directory="", category_mapping="", batch_size=5):
        self.data = data
        self.batch_size = batch_size
        self.directory = directory
        self.category_mapping = category_mapping

        length = len(data)
        iterations = length // batch_size

        self.length = length
        self.iterations = iterations

        if isinstance(data, str):
            print("Data is of type string")
        elif isinstance(data, list):
            print("Data is of type list")
        else:
            print("[WARN] Data is not iterable!!")

    # print("Initializing generator")
    # return self.generator()

    def __repr__(self):
        return str(self.data)

    def __len__(self):
        return len(self.iterations)

    def generator(self):
        for i in range(self.iterations):
            start = self.index
            end = self.index + self.batch_size
            self.index = end
            yield [self.data[i] for i in range(start, end)], i


data = [a for a in range(17)]

gen = MyGenerator(data)
# print(len(gen))
# print(gen)
mygen = gen.generator()
print(gen.iterations)
print(gen.batch_size)
for i in mygen:
    print(i)
