import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from tensorflow_core.python.keras.models import load_model

from utils import Timer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean


class WaterLevelModel:

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, activation, input_timesteps, units, dropout_rate, num_x_signals,num_y_signals, optimizer):
        timer = Timer()
        timer.start()

        self.model.add(Dense(activation=activation, units=units))
        self.model.add(GRU(activation=activation,
                           return_sequences=True,
                           units=units,
                           input_shape=(input_timesteps, num_x_signals)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(units=units))
        self.model.add(Dense(num_y_signals,
                             activation='sigmoid'))

        self.model.compile(optimizer=optimizer)

        print('[Model] Model Compiled')

        timer.stop()
