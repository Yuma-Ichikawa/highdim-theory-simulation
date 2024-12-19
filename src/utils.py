import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import seaborn as sns
import pandas as pd

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def H(x):
    return 0.5 * (1 - erf(x / torch.sqrt(torch.tensor(2.0, device=device))))
