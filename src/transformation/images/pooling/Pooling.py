from abc import ABC, abstractmethod
from transformation.images.ImageTransformation import ImageTransformation
from transformation.images.filter.Filter import Filter
import numpy as np


class Pooling(ImageTransformation, ABC):

    def __init__(self, filterLength):
        super().__init__()
        self.filter = Filter(filterLength)

    @abstractmethod
    def pool(self, matrix):
        pass

    def transform(self, image):
        outputchannels=[]
        for channel in range(len(image[0][0])):
            outputchannel=[]
            for i in range(0, len(image[0]), self.filter.getLength()):
                outputrow=[]
                for j in range(0, len(image), self.filter.getLength()):
                    outputrow.append(self.pool(image[i:i+self.filter.getLength(), j:j+self.filter.getLength(), channel]))
                outputchannel.append(outputrow)
            outputchannels.append(outputchannel)
        return np.dstack(outputchannels)




