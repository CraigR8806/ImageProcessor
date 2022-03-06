from transformation.images.convolution.Convolution import Convolution
from transformation.images.convolution.edgedetection.VerticalSobel import VerticalSobel
from transformation.images.convolution.edgedetection.HorizontalSobel import HorizontalSobel
from parallellinear.datatypes.Matrix import Matrix
import numpy as np


class SobelProcessing(Convolution):


    def __init__(self, kernelSize):
        super().__init__([HorizontalSobel(kernelSize).getKernel(), VerticalSobel(kernelSize).getKernel()])

    def transform(self, image):
        return self.convolve(image)

    def _convoleOperation(self, row, column, channels, image):
        outpixel = []
        for channel in range(channels):
            sobelValues = []
            subset = list(image[row:row+self.filterSize, column:column+self.filterSize, channel].flat)
            for kernel in self.kernels:
                sobelValues.append(kernel.getMatrix().elementWiseMultiply(Matrix.fromFlatListGivenRowNumber(self.filterSize, subset), in_place=False).sum())
            outpixel.append(np.sqrt(np.square(sobelValues[0]) + np.square(sobelValues[1])))
        return outpixel
