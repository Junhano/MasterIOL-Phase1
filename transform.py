import torchvision.transforms.functional as TF


class SpecificErase:


    def __init__(self, i, j, h, w, v = 0):
        self.i = i
        self.j = j
        self.h = h
        self.w = w
        self.v = v

    def __call__(self, image):
        return TF.erase(image, self.i, self.j, self.h, self.w, self.v)