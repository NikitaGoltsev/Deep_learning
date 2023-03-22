import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import rc


class Leniar_reg():
    # I need to build it by the house dateset from kage
    def __init__(self) -> None:
         
        self.df_train = pd.read_csv('house_prices_train.csv')
        self.df_train['SalePrice'].describe()
        
        return None
        

obj1 = Leniar_reg()
   