from __future__ import print_function
from __future__ import division

import keras as K

from keras.applications.mobilenet import preprocess_input

from cosine_softmax import CosineSoftmax
from generator import Generator

from utils import compute_cmc

IN_DIR = '/home/farmer/reid/data/'
MODEL_RESTORE_WEIGHTS = '/hdd/reid/models/weights_13_.11-4.89-0.82.hdf5'
MODEL_RESTORE_JSON = '/hdd/reid/models/model_13_v2.json'
TARGET_SIZE = (224, 112)
BATCH_SIZE = 256

if __name__ == '__main__':
    generator = Generator(IN_DIR, BATCH_SIZE, TARGET_SIZE[1], TARGET_SIZE[0],
                          val_to_train=0.15, preprocessor=preprocess_input,
                          augmenter=None)  # augm.augment_images)
    json_string = open(MODEL_RESTORE_JSON).read()
    model = K.models.model_from_json(json_string, custom_objects={
        'CosineSoftmax': CosineSoftmax})
    model.load_weights(MODEL_RESTORE_WEIGHTS)
    model.summary()

    validator = generator.make_val_generator()
    print('CMC_i: ', compute_cmc(model, validator, k=5))
