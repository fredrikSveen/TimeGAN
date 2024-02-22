## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import datetime
import time
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan, timegan_from_pretrained
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
# 4. Analysis
from utils import list_to_df

try:
  model_to_load = sys.argv[1]
except IndexError:
  print("Missing argument: Specify the path for the pretrained model to use")
  sys.exit(1)
except Exception as e:
  print(f'Something is wrong. The exception is {e}')
  sys.exit(1)


# get the start time
st = time.time()


## Data loading
data_name = 'sensor'
seq_len = 24

if data_name in ['stock', 'energy', 'sensor']:
  ori_data = real_data_loading(data_name, seq_len)
elif data_name == 'sine':
  # Set number of samples and its dimensions
  no, dim = 10000, 5
  ori_data = sine_data_generation(no, seq_len, dim)
    
print(data_name + ' dataset is ready.')

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 10000
parameters['batch_size'] = 128

generated_data_pretrained = timegan_from_pretrained(model_to_load, ori_data, parameters, reproduce=False)

# Save generated data to csv
x = datetime.datetime.now()

timestamp = x.strftime("%d_%m_%y__%Hh%M")
generated_df = list_to_df(generated_data_pretrained)
filepath = f'generated_data/gen_data_pretrained{timestamp}'
generated_df.to_csv(f'{filepath}.csv')
generated_df.to_csv(f'{filepath}_indexless.csv', index=False)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')