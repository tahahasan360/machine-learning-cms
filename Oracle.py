from abc import ABC, abstractmethod 
import numpy as np
import pickle
import sys
  
class Oracle(ABC): 
  
    @abstractmethod
    def predict_heavy_hitter(self, token): 
        pass

    @abstractmethod
    def get_memory_usage(self): 
        pass
