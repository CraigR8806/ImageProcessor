from abc import ABC, abstractmethod
from transformation.images.convolution.Convolution import Convolution


class Blur(Convolution, ABC):

    def __init__(self, kernel):
        super().__init__([kernel], kernel.getSize())

    @abstractmethod
    def transform(self, image):
        return self.convolve(image)

