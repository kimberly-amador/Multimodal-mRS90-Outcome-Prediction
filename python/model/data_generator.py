import numpy as np
from tensorflow.keras.utils import Sequence


# -----------------------------------
#      DATA GENERATOR - 4D CTP
# -----------------------------------

class DataGenerator_CTP(Sequence):
    """
    To utilize this data generator:
        1. Images should be already preprocessed, saved as a numpy array of name 'PatientID/preprocessed.npz'
        2. Npz files must contain a variable named 'img' containing the CTP data array
        3. 'img' should be of shape (height, width, n_slices, n_timepoints)

    Output:
        - List with three elements: CTP images, categorical features, continuous/numerical features
        - A scalar value representing the binary mRS90 score
    """
    def __init__(self, list_IDs, metadata_pd, outcome_scores, imagePath, dim=(512, 512, 16), batch_size=1, timepoints=32, n_classes=2, features=None, shuffle=True, **kwargs):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.metadata_pd = metadata_pd
        self.outcome_scores = outcome_scores
        self.timepoints = timepoints
        self.n_classes = n_classes
        self.features = features
        self.imagePath = imagePath
        self.shuffle = shuffle
        self.on_epoch_end()

    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # Generate indexes of the batch
        list_IDs_temp = [self.list_IDs[k] for k in indexes]  # Find list of IDs
        X, y = self.__data_generation(list_IDs_temp)  # Generate data
        return X, y

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates data containing batch_size samples
    def __data_generation(self, list_IDs_temp):
        # Variable initialization
        X = np.empty((self.batch_size, *self.dim, self.timepoints), dtype='float32')
        y = np.empty(self.batch_size, dtype='float32')
        C_continuous = np.empty((self.batch_size, len(self.features[0])), dtype='float32')
        C_categorical = np.empty((self.batch_size, len(self.features[1])), dtype='int32')

        # Generate data according to patient IDs
        for k, ID in enumerate(list_IDs_temp):
            X[k, ] = np.load(self.imagePath + f'{ID}/preprocessed.npz')['img']
            y[k, ] = self.outcome_scores.loc[ID].to_numpy()[0]  # y is a scalar value
            C_continuous[k, ] = self.metadata_pd[self.features[0]].loc[ID].to_numpy()  # error will raise when having one variable. remove to_numpy() in those situations
            C_categorical[k, ] = self.metadata_pd[self.features[1]].loc[ID].to_numpy()

        # Convert np arrays to lists (to match the model input format)
        X = np.split(X, self.timepoints, axis=4)
        C_continuous = np.split(C_continuous, len(self.features[0]), axis=1)
        C_categorical = np.split(C_categorical, len(self.features[1]), axis=1)
        return [X, C_categorical, C_continuous], y
