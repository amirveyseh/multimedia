import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D

import os, numpy as np

_BATCH_SIZE = 32
_NUM_CLASS = 2
_EPOCH = 10
_SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
_MODEL_NAME = 'keras_relevance_trained_model.h5'

class relevance_learner:
    def __init__(self, train_set, label_set, feature_size):
        self.train_set = np.asarray(train_set)
        self.label_set = np.asarray(label_set)
        self.feature_size = feature_size

    def train(self, save=False):
        self.model = Sequential()
        self.model.add(Conv1D(10, 10, activation='relu', input_shape=(self.feature_size, 1)))
        self.model.add(MaxPooling1D(pool_size=5))
        self.model.add(Conv1D(10, 10, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())
        self.model.add((Dense(_NUM_CLASS, activation='sigmoid')))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        self.label_set = keras.utils.to_categorical(self.label_set, _NUM_CLASS)
        self.train_set = np.expand_dims(self.train_set, axis=2)
        self.model.fit(self.train_set, self.label_set,
                  batch_size=_BATCH_SIZE,
                  epochs=_EPOCH,
                  shuffle=True)

        if(save):
            if not os.path.isdir(_SAVE_DIR):
                os.makedirs(_SAVE_DIR)
            model_path = os.path.join(_SAVE_DIR, _MODEL_NAME)
            self.model.save(model_path)
            print('Saved trained model at %s ' % model_path)

        return self.model

    @staticmethod
    def predict(model, test_set):
        test_set = np.expand_dims(test_set, axis=2)
        return model.predict_classes(test_set)

    @staticmethod
    def evaluate(model, test_set, label_set):
        test_set = np.expand_dims(test_set, axis=2)
        label_set = keras.utils.to_categorical(label_set, _NUM_CLASS)
        scores = model.evaluate(test_set, label_set, verbose=1)
        print ''
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    @staticmethod
    def predict_by_mappings(model, test_set, mappings):
        labels = relevance_learner.predict(model, test_set)
        predictions = {}
        for i, label in enumerate(labels):
            query = mappings[i]['query']
            if query not in predictions.keys():
                predictions[query] = {}
            predictions[query][mappings[i]['img']] = label
        return predictions