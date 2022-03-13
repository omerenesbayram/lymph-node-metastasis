import math
from datetime import datetime

import numpy as np
from tensorflow.keras.utils import Sequence

import config
from data_generator.helper import dicom


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, gen_type, name_list_path, batch_size, segment=0., is_whole=True, 
                 size = 100, n_channels=1, shuffle=True):
        'Initialization'
        self.dim = (size,size,size)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.size = size
        self.segment = segment
        self.is_whole = is_whole
        self.gen_type = gen_type
        self.name_list_path = name_list_path
        self.time = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
       
   
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.list_IDs.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(self.list_IDs.shape[0] / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        list_IDs_temp = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
      
        X, y = self.__data_generation(list_IDs_temp, index)

        return X,y

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
 
    def __data_generation(self, list_IDs_temp, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size,2))
        
        # Generate data
        
        for i, ID in enumerate(list_IDs_temp):
          ID = ID.lower()
          try:
            if i % 2:
              y[i] = np.array([0,1])
              X[i,], _ = dicom.read_patient(ID, self.size, self.segment, self.is_whole)
            else:
              y[i] = np.array([1,0])
              X[i,] = np.random.randn(self.size, self.size, self.size, 1)
          except Exception as e:
              file_name = config.LOG_PATH + f"/log_{self.time}.txt"
              with open(file_name, 'a+') as f:
                  f.write(ID + "\n\n\n")
                  f.write(str(e))
                  f.write("\n---------------------------\n\n\n")
        
        return X,y
