from transformation.images.convolution.distortion.blur.Blur import Blur
from transformation.images.convolution.Kernel import Kernel
from parallellinear.datatypes.Matrix import Matrix
import numpy as np



class GaussianBlur(Blur):

    def __init__(self, kernelSize, sigma, normalize=False, minNormalValue=0, maxNormalValue=255):
        self.minNormalValue = minNormalValue
        self.maxNormalValue = maxNormalValue
        self.normalRange = self.maxNormalValue - self.minNormalValue
        self.normalize = normalize
        center = kernelSize // 2
        x, y = np.mgrid[0 - center : kernelSize - center, 0 - center : kernelSize - center]
        buffer = 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        matrix=[]
        for i in range(kernelSize):
            matrix.extend(buffer[i])
        super().__init__(Kernel(kernelSize, Matrix.fromFlatListGivenRowNumber(kernelSize, matrix)))


    def transform(self, image):
        return super().transform(image)

    def _adjustOutgoingValue(self, value):
        if self.normalize:
            return np.uint8(((value-self.minNormalValue)/self.normalRange)*(self.normalRange + self.minNormalValue))
        else:
            return value