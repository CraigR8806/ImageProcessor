from abc import ABC, abstractmethod
from transformation.images.ImageTransformation import ImageTransformation
from parallellinear.datatypes.Matrix import Matrix
import numpy as np


class Convolution(ImageTransformation, ABC):
    
    def __init__(self, kernels, kernelSize):
        super().__init__()
        self.kernels = kernels
        self.kernelSize = kernelSize
    
    def getKernels(self):
        return self.kernels
    
    def convolve(self, image):
        outimage=[]
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]
        for row in range(height):
            if row+self.kernelSize > height:
                    continue
            outrow=[]
            for column in range(width):
                if column+self.kernelSize > width:
                    continue
                outrow.append(self._convoleOperation(row, column, channels, image))
            outimage.append(outrow)
        return np.array(outimage)


    def _convoleOperation(self, row, column, channels, image):
        outpixel = []
        for channel in range(channels):
            subset = list(image[row:row+self.kernelSize, column:column+self.kernelSize, channel].flat)
            for kernel in self.kernels:
                outpixel.append(kernel.getMatrix().elementWiseMultiply(Matrix.fromFlatListGivenRowNumber(self.kernelSize, subset), in_place=False).sum())
        return outpixel