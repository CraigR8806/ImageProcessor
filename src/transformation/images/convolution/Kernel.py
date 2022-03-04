from parallellinear.datatypes.Matrix import Matrix
from transformation.images.filter.Filter import Filter

class Kernel(Filter):



    def __init__(self, length, matrix):
        super().__init__(length)
        self.matrix = matrix


    @classmethod
    def random(cls, length, calcManager=None, random_low=0, random_high=1):
        matrix=Matrix.random(length, length, calcManager=calcManager, random_low=random_low, random_high=random_high)
        return cls(length, matrix)

    def getMatrix(self):
        return self.matrix

    def transpose(self, in_place=True):
        if in_place:
            self.matrix.transpose()
        else:
            return Kernel(self.length, self.matrix.transpose(in_place=False))


    