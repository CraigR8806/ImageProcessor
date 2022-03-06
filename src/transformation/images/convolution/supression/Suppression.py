from abc import ABC, abstractmethod
from transformation.images.convolution.Convolution import Convolution
from transformation.images.filter.Filter import Filter

class Suppression(Convolution, ABC):

    def __init__(self, filterSize):
        super().__init__([Filter(filterSize)])
        
    def transform(self, image):
        pass

    @abstractmethod
    def suppress(self):
        pass

    