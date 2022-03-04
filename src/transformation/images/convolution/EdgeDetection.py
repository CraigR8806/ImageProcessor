from transformation.images.convolution.Convolution import Convolution
from transformation.images.convolution.Kernel import Kernel
from parallellinear.datatypes.Matrix import Matrix


class EdgeDetection(Convolution):

    def __init__(self, kernelLength):
        template=[0 for i in range(kernelLength)]
        template[0] = 1
        template[-1] = -1
        vertmatrix=[]
        for i in range(kernelLength):
            vertmatrix.extend(template)
        vertKernel = Kernel(kernelLength, Matrix.fromFlatListGivenRowNumber(kernelLength, vertmatrix))
        horizKernel = vertKernel.transpose(in_place=False)
        super().__init__([vertKernel, horizKernel])


    def transform(self, image):
        return self.convolve(image)
