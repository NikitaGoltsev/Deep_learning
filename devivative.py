import numpy as np
from sympy import *# That's lib created to work with devivvaitive

class Dev_Functions_Set():
    # I need to define most of functions here
    def __init__(self) -> None:
        
        # You need to define x and any functions to calculete dvv from x
        x = Symbol('x')
        f = (x ** 2)/ (4 * x)
        print(f.diff(x)) # dvv

        return None

obj1 = Dev_Functions_Set()