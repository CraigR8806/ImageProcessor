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
        outimage=[]
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]
        for row in range(0, height, self.filter.getSize()):
            outrow=[]
            for column in range(0, width, self.filter.getSize()):
                outpixel=[]
                for channel in range(channels):
                    outpixel.append(self.pool(image[row:row+self.filter.getSize(), column:column+self.filter.getSize(), channel]))
                outrow.append(outpixel)
            outimage.append(outrow)
        return outimage




