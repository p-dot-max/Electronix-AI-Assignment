
import torch
import numpy as np
import random

# Manula seeding the model
class seeding:
    def __init__(self, seed=42):
        self.seed = seed

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)




