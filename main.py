from pipeline import *
from util import generate_train_configs

all_train_configs = generate_train_configs()

multi_CrossFoldPipeline(all_train_configs)