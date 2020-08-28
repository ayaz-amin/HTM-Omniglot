import numpy as np
from scipy.spatial.distance import cdist
from htm_utils import SpatialPooler


class OmniglotHTM(object):
    '''
    HTM applied to one shot learning

    Parameters
    ----------
    input_size: int
        The size of the input data
    columns: int
        The number of mini-columns in the spatial pooler
    '''
    def __init__(self, input_size, columns):
        '''
        Attributes
        ----------
        active_columns: int
            The number of active columns in the spatial pooler (i.e. the SDR)
        input_size: int
            The size of the input data
        columns: int
            The number of mini-columns in the spatial pooler
        memory: [np.array]
            Memory storing learnt SDRs
        pooler: SpatialPooler
            The spatial pooler
        '''
        active_columns = int(columns * 0.02)
        self.memory = []
        self.pooler = SpatialPooler(input_size, columns, active_columns)

    def reset_memory(self):
        '''Resets memory'''
        self.memory = []
        
    def learn(self, dataset):
        '''Learning algorithm for one timestep'''
        self.reset_memory()
        for input_data in dataset:
            self.pooler.run(input_data)
            self.memory.append(self.pooler.active)
    
    def compare_dist(self, input_a, input_b):
        '''
        Computing the distance between two inputs

        Parameters
        ----------
        input_a: np.array
            Input data #1 (SDR)
        input_b: np.array
            Input data #2 (SDR)

        Returns
        -------
        distance: float
            The distance between two points
        '''
        input_a = input_a.reshape((-1, 1))
        input_b = input_b.reshape((-1, 1))
        distance = cdist(input_a, input_b)
        mean_a = np.mean(distance.min(axis=1))
        mean_b = np.mean(distance.min(axis=0))
        return max(mean_a, mean_b)

    def predict(self, input_data, learn=False):
        '''
        Inference procedure

        Parameters
        ----------
        input_data: np.array
            Input image
        
        Returns
        -------
        predicted_index: int
            The predicted class of the image
        '''
        clf_dists = []

        self.pooler.run(input_data)
        sdr = self.pooler.active

        for stored_objects in self.memory:
            distance = self.compare_dist(sdr, stored_objects)
            clf_dists.append(distance)

        clf_dists = np.array(clf_dists)
        predicted_index = np.argmin(clf_dists)
        if learn:
            self.memory[predicted_index] = sdr
        return predicted_index