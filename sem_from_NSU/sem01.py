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
        #print(self.df_train['SalePrice'].describe()) # some info
        print(self.df_train.head())

        self.__check_count_to_c__()


        return None
        
    def __check_count_to_c__(self):

        var = 'GrLivArea' # The 
        data = pd.concat([self.df_train['SalePrice'], self.df_train[var]], axis=1)
        data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), s=32)
        plt.show() # We can see linear 

        return None
    
    
obj1 = Leniar_reg()
