from abc import ABC, abstractmethod
from transformation.images.ImageTransformation import ImageTransformation


class Convolution(ImageTransformation, ABC):
    
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels


    

    def getKernels(self):
        return self.kernels
    