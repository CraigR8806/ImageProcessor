from transformation.images.convolution.Convolution import Convolution
from transformation.images.convolution.Kernel import Kernel
from parallellinear.datatypes.Matrix import Matrix
import imageio
import numpy as np



class StandardConvolution(Convolution):


    def __init__(self, kernels):
        super().__init__(kernels)


    @classmethod
    def randomKernel(cls, length, channels, calcManager=None, random_low=0, random_high=1):
        kernels=[ Kernel.random(length, calcManager=calcManager, random_low=random_low, random_high=random_high) for i in range(channels)]
        return cls(kernels)


    def transform(self, image):
        outchannels=[]
        numberOfChannels=len(image[0][0])
        if numberOfChannels != len(self.kernels):
            raise ValueError("Image provided has an incorrect number of color channels for the loaded Kernal for this transformation...")
        channel=0
        for layer in self.kernels:
            outchannel=[]
            for i in range(len(image[0])):
                if i+layer.getLength() > len(image[0]):
                        continue
                outrow=[]
                for j in range(len(image)):
                    if j+layer.getLength() > len(image):
                        continue
                    subset = list(image[i:i+layer.getLength(), j:j+layer.getLength(), channel].flat)
                    outrow.append(layer.getMatrix().elementWiseMultiply(Matrix.fromFlatListGivenRowNumber(layer.getLength(), subset), in_place=False).sum())
                outchannel.append(outrow)

            outchannels.append(outchannel)
            channel+=1
        outimage=np.dstack(outchannels)
        max=np.max(outimage)
        outimage=((outimage/max)*255).astype(np.int32)
        
        return outimage
        


        