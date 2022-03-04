from abc import ABC, abstractmethod
from transformation.images.ImageTransformation import ImageTransformation
from parallellinear.datatypes.Matrix import Matrix
import numpy as np


class Convolution(ImageTransformation, ABC):
    
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels
    
    def getKernels(self):
        return self.kernels
    
    def convolve(self, image):
        outchannels=[]
        numberOfChannels=len(image[0][0])
        for layer in self.kernels:
            outchannel=[]
            for i in range(len(image[0])):
                if i+layer.getLength() > len(image[0]):
                        continue
                outrow=[]
                for j in range(len(image)):
                    if j+layer.getLength() > len(image):
                        continue
                    outpixel=[]
                    for k in range(len(image[0][0])):
                        subset = list(image[i:i+layer.getLength(), j:j+layer.getLength(), k].flat)
                        outpixel.append(layer.getMatrix().elementWiseMultiply(Matrix.fromFlatListGivenRowNumber(layer.getLength(), subset), in_place=False).sum())
                    outrow.append(outpixel)
                outchannel.append(outrow)
            max=np.max(outchannel)
            min=np.min(outchannel)
            outchannel=(((outchannel - min)/(max - min))*255).astype(np.int32)
            outchannels.append(outchannel)
        return outchannels