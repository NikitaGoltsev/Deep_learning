import numpy as np

class Activation_Functions():

    def __init__(self) -> None:
        
        return None

    # Here I need to relise new functions of actiwations( all new functions for me)

    ReLU = (lambda self, x: max(0.0, x))

    def SoftMax(self, x) -> None:

        return (np.exp(x)/np.exp(x).sum())
