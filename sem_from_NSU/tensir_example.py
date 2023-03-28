import numpy as np
#from typing import Touple
import torch
import torchvision # tensor_to_PIL(obj.circle)
from torchvision import transforms
from PIL import Image
# I need to work with tensor and img with torch fitches

def normalize_image(image): # !!! ???
    min_intensity = torch.min(image)
    image = image - min_intensity
    max_intensity = torch.max(image)
    return image / max_intensity

def tensor_to_PIL(tensor): # !!! ???
    return transforms.ToPILImage()(normalize_image(tensor).squeeze(0))

class Tensor_Figurse():

    def __init__(self) -> None:
        
        #I'm going to init something if will need it

        return None
    
    def circle(self, r:int = 5, width:int = 5) -> torch.Tensor: # Create and retrunrn circle
        # Cirle with r =
        diametr = 2*r
        canvas = torch.zeros((diametr, diametr))# Torch tensor with in 2d space
        # canvas - got only zero in every elem
        # 
        for i in range(diametr):
            for j in range(diametr):
                left_side = (i - r) ** 2 + (j - r) ** 2
                if(left_side >= r**2 and left_side >= (r - width)**2):
                    canvas[i,j] = 1
        
        return canvas

    def square(self, x:int = 10, width:int = 5) -> torch.Tensor:
        
        canvas = torch.ones((x,x))
        inner = torch.zeros((x - 2*width ), (x - 2*width ))
        # And now I am cut inner from canvas ) (I able to explain that on this example)
        canvas[width:x-width, width:x-width] = inner 

        return canvas

    def block(self, x:int = 10, y:int = 10, width:int = 5) -> torch.Tensor:

        canvas = torch.ones((y, x))
        inner = torch.zeros((y - 2 * width, y - 2 * width))
        canvas[width:y-width, width:x-width] = inner 

        return canvas
    
