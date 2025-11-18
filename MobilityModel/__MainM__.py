# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import numpy as np
from ClassM import ModelM
from tqdm import tqdm
import pandas as pd
import sys

# ----------------------------------------------------------------- #
# Initialize Class
# ----------------------------------------------------------------- #

params_input = {'savename': 'High',
                'division': 100 # 5000 - 1000 - 500 - 100
                }
ClassM = ModelM(params_input)
ClassM.read_data()
#ClassM.mobility_matrix()
seed = int(sys.argv[1])
for mc in [seed]:
    ClassM.create_people_DF()
    ClassM.position_people()
    ClassM.create_extra_people_DF(5)
    ClassM.position_extraPeople()
    # ClassM.count_people()
    ClassM.save(mc) 
