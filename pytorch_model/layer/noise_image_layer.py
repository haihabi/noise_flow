import normflowpy as nfp


class ImageFlowStep(nfp.ConditionalBaseFlowLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x, cond):
        clean_image = cond[0]
        return x - clean_image, 0

    def backward(self, z, cond):
        clean_image = cond[0]
        return z + clean_image, 0
