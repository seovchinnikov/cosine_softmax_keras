from __future__ import print_function
from __future__ import division
import os
import math

import os
import tensorflow as tf
import keras as K
from keras.backend import image_data_format
from keras.engine import Model
from keras.layers import GlobalAveragePooling2D, Activation, Dropout, Conv2D, Reshape
from keras.applications import keras_modules_injection
from keras.applications.mobilenet import preprocess_input
from imgaug import augmenters as iaa
import imgaug as ia
import mobilenet_custom
from cosine_softmax import CosineSoftmax
from generator import Generator

from keras.backend.tensorflow_backend import set_session

from eval_utils import cmc_callback

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

IN_DIR = '/home/farmer/reid/data_combined/'
MODEL_RESTORE = None  # '/hdd/reid/models/weights_.04-6.67-0.26.hdf5'
MODEL_OUT_DIR = '/hdd/reid/models/'
TARGET_SIZE = (224, 112)
BATCH_SIZE = 128
ENCODER_SHAPE = 128
DROPOUT_LAST_LAYER = 0.25
DROPOUT_MIDDLE_LAYER = 0.75

sometimes03 = lambda aug: iaa.Sometimes(0.3, aug)
sometimes05 = lambda aug: iaa.Sometimes(0.5, aug)

augm_hard = iaa.Sequential([
    sometimes03(iaa.GaussianBlur((0, 1.0), name="GaussianBlur")),
    iaa.SomeOf((0, 2), [
        iaa.Dropout((0, 0.1), name="Dropout"),
        iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255), name="LittleNoise")
    ]),
    sometimes05(iaa.Affine(rotate=(-20, 20), scale=(0.8, 1.2), translate_percent=(0, 0.1))),
    sometimes05(iaa.CropAndPad(
        percent=(-0.15, 0.3),
        pad_mode=ia.ALL,
        pad_cval=(0, 255)
    )),
    iaa.SomeOf((0, 2), [
        iaa.Add((-20, 20), per_channel=0.2),  # change brightness of images (by -10 to 10 of original value)
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.ContrastNormalization((0.8, 1.2), per_channel=0.2),  # improve or worsen the contrast
    ])
])

augm_middle = iaa.Sequential([
    sometimes03(iaa.GaussianBlur((0, 1.0), name="GaussianBlur")),
    iaa.SomeOf((0, 1), [
        iaa.Dropout((0, 0.1), name="Dropout"),
        iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255), name="LittleNoise")
    ]),
    sometimes05(iaa.Affine(rotate=(-10, 10), scale=(0.9, 1.1), translate_percent=(0, 0.07))),
    sometimes05(iaa.CropAndPad(
        percent=(-0.07, 0.07),
        pad_mode=ia.ALL,
        pad_cval=(0, 255)
    )),
    iaa.SomeOf((0, 2), [
        iaa.Add((-15, 15), per_channel=0.2),  # change brightness of images (by -10 to 10 of original value)
        iaa.AddToHueAndSaturation((-15, 15)),  # change hue and saturation
        iaa.Multiply((0.9, 1.1), per_channel=0.2),
        iaa.ContrastNormalization((0.9, 1.1), per_channel=0.2),  # improve or worsen the contrast
    ])
])

augm_easy = iaa.Sequential([
    sometimes03(iaa.Affine(rotate=(-5, 5), scale=(0.97, 1.03), translate_percent=(0, 0.03))),
    sometimes03(iaa.CropAndPad(
        percent=(-0.03, 0.03),
        pad_mode=ia.ALL,
        pad_cval=(0, 255)
    )),
    iaa.SomeOf((0, 1), [
        iaa.Add((-8, 8), per_channel=0.2),  # change brightness of images (by -10 to 10 of original value)
        iaa.AddToHueAndSaturation((-8, 8)),  # change hue and saturation
        iaa.Multiply((0.97, 1.03), per_channel=0.2),
        iaa.ContrastNormalization((0.97, 1.03), per_channel=0.2),  # improve or worsen the contrast
    ])
])


def mobilenet_out_prepare(layer, name, dropout=0.):
    base_model_out = layer.output
    dim_out = base_model_out.shape[3]
    if image_data_format() == 'channels_first':
        shape = (int(dim_out), 1, 1)
    else:
        shape = (1, 1, int(dim_out))

    x = GlobalAveragePooling2D()(base_model_out)
    x = Reshape(shape, name='reshape_' + name)(x)
    x = Dropout(dropout, name='dropout_' + name)(x)
    return x


def create_model_mobilenet(model_to_restore, in_shape, out_shape):
    @keras_modules_injection
    def MobileNetV2(*args, **kwargs):
        return mobilenet_custom.MobileNetV2(*args, **kwargs)

    base_model = MobileNetV2(alpha=1.3, input_shape=in_shape, include_top=False,
                             weights='imagenet', pooling=None)

    #out_5 = mobilenet_out_prepare(base_model.get_layer('block_5_add'), '5', dropout=DROPOUT_MIDDLE_LAYER)
    #out_9 = mobilenet_out_prepare(base_model.get_layer('block_9_add'), '9', dropout=DROPOUT_MIDDLE_LAYER)
    out_12 = mobilenet_out_prepare(base_model.get_layer('block_12_add'), '12', dropout=DROPOUT_MIDDLE_LAYER)
    out_14 = mobilenet_out_prepare(base_model.get_layer('block_14_add'), '14', dropout=DROPOUT_MIDDLE_LAYER)
    out_15 = mobilenet_out_prepare(base_model.get_layer('block_15_add'), '15', dropout=DROPOUT_MIDDLE_LAYER)
    out_last = mobilenet_out_prepare(base_model.get_layer('out_relu'), 'last', dropout=DROPOUT_LAST_LAYER)
    out_last = K.layers.concatenate([out_12, out_14, out_15, out_last], axis=-1)

    x = Conv2D(512, (1, 1),
               padding='valid', name='conv_preds0_prelast', activation=None)(out_last)
    x = K.layers.BatchNormalization(epsilon=1e-3,
                                    momentum=0.999,
                                    name='Conv_bn_prelast')(x)
    x = K.layers.ReLU(6., name='pre_preencoding')(x)
    ####

    x = Conv2D(ENCODER_SHAPE, (1, 1),
               padding='valid', name='conv_preds0_last', activation=None)(x)
    # x = K.layers.ReLU(6., name='pre_encoding')(x)
    x = Reshape((ENCODER_SHAPE,), name='encoding')(x)
    preds = CosineSoftmax(output_dim=out_shape)(x)
    model = Model(inputs=base_model.inputs, outputs=preds)
    # for layer in base_model.layers:
    #         layer.trainable = False

    if model_to_restore is not None:
        model.load_weights(model_to_restore)

    return model


if __name__ == '__main__':
    generator = Generator(IN_DIR, BATCH_SIZE, TARGET_SIZE[1], TARGET_SIZE[0],
                          val_to_train=0.15, preprocessor=preprocess_input,
                          augmenter=augm_hard.augment_images)  # augm.augment_images)
    model = create_model_mobilenet(MODEL_RESTORE, (TARGET_SIZE[0], TARGET_SIZE[1], 3), generator.cats_num)

    if MODEL_RESTORE is not None:
        optimizer = K.optimizers.RMSprop(lr=0.0001, decay=0.03)
    else:
        optimizer = K.optimizers.Adam(lr=0.0001)
        #optimizer = K.optimizers.RMSprop(lr=0.002, decay=0.01)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    out = open(os.path.join(os.path.normpath(MODEL_OUT_DIR), "model_13_v2.json"), "w")
    out.write(model.to_json())
    out.close()

    val_gen = generator.make_val_generator()
    model.fit_generator(
        generator.make_train_generator(),
        epochs=200,
        verbose=2,
        validation_data=val_gen,
        callbacks=[
            K.callbacks.ModelCheckpoint(
                os.path.join(MODEL_OUT_DIR, "weights_13_.{epoch:02d}-{val_loss:.2f}-{acc:.2f}.hdf5"),
                save_best_only=False),
            cmc_callback(val_gen)
        ],
        use_multiprocessing=True, workers=3,
        class_weight=None)
