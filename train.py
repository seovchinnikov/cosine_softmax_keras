from __future__ import print_function
from __future__ import division
import os
import math

import os

import keras as K
from keras.engine import Model
from keras.layers import GlobalAveragePooling2D, Activation
from keras.applications import MobileNetV2, keras_modules_injection
from keras.applications.mobilenet import preprocess_input
from imgaug import augmenters as iaa

import mobilenet_custom
from cosine_softmax import CosineSoftmax
from generator import Generator

IN_DIR = 'F:\\mars_dataset\\'
MODEL_RESTORE = None
MODEL_OUT_DIR = 'F:\\reid_model\\'
TARGET_SIZE = (224, 112)
BATCH_SIZE = 128

augm = iaa.Sequential([
    iaa.GaussianBlur((0, 1.0), name="GaussianBlur"),
    iaa.SomeOf((0, 2), [
        iaa.Dropout((0, 0.1), name="Dropout"),
        iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255), name="LittleNoise")
    ]),
    # iaa.Affine(rotate=(-15, 15)),  # rotate by -45 to +45 degrees, name="Affine"),
    iaa.SomeOf((0, 2), [
        iaa.Add((-30, 30), per_channel=0.2),  # change brightness of images (by -10 to 10 of original value)
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.ContrastNormalization((0.8, 1.2), per_channel=0.2),  # improve or worsen the contrast
    ])
])


def create_model_mobilenet(model_to_restore, in_shape, out_shape):
    @keras_modules_injection
    def MobileNetV2(*args, **kwargs):
        return mobilenet_custom.MobileNetV2(*args, **kwargs)

    base_model = MobileNetV2(alpha=1.0, input_shape=in_shape, include_top=False,
                             weights='imagenet', pooling=None)
    base_model_out = base_model.get_layer('out_relu').output
    x = GlobalAveragePooling2D()(base_model_out)
    x = K.layers.Dense(128, activation='relu', use_bias=True, name='encoding')(x)

    preds = CosineSoftmax(output_dim=out_shape)(x)
    model = Model(inputs=base_model.inputs, outputs=preds)
    # for layer in base_model.layers:
    #         layer.trainable = False

    if model_to_restore is not None:
        model.load_weights(model_to_restore)

    return model


if __name__ == '__main__':
    generator = Generator(IN_DIR, BATCH_SIZE, TARGET_SIZE[1], TARGET_SIZE[0],
                          val_to_train=0.15, preprocessor=preprocess_input, augmenter=augm.augment_images)
    model = create_model_mobilenet(MODEL_RESTORE, (TARGET_SIZE[0], TARGET_SIZE[1], 3), generator.cats_num)

    if MODEL_RESTORE is not None:
        optimizer = K.optimizers.RMSprop(lr=0.0001, decay=0.03)
    else:
        optimizer = K.optimizers.RMSprop(lr=0.001, decay=0.03)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    out = open(os.path.join(os.path.normpath(MODEL_OUT_DIR), "model_v2.json"), "w")
    out.write(model.to_json())
    out.close()

    model.fit_generator(
        generator.make_train_generator(),
        epochs=100,
        verbose=1,
        validation_data=generator.make_val_generator(),
        callbacks=[
            K.callbacks.ModelCheckpoint(
                os.path.join(MODEL_OUT_DIR, "weights_.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5"),
                save_best_only=True)
        ],
        class_weight=generator.obtain_class_weights())
