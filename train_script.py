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
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
# 4. Analysis
from utils import list_to_df

# get the start time
st = time.time()

## Data loading
data_name = 'sensor'
try:
  seq_len = int(sys.argv[1])
  print(f'Sequence length: {seq_len}')
except IndexError:
  seq_len = 24
  print(f'The sequence length was not specified. The standard length of {seq_len} was used')

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

# Run TimeGAN
generated_data = timegan(ori_data, parameters, reproduce=True)   
print('Finish Synthetic Data Generation')

# Save generated data to csv
x = datetime.datetime.now()

timestamp = x.strftime("%d_%m_%y__%Hh%M")
generated_df = list_to_df(generated_data)
filepath = f'generated_data/gen_{data_name}_norm{timestamp}'
generated_df.to_csv(f'{filepath}.csv')
generated_df.to_csv(f'{filepath}_indexless.csv', index=False)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
h = round(elapsed_time//(60*60), 0)
m = round((elapsed_time - h*(60*60))//60, 0)
s = round(elapsed_time - h*(60*60) - m*60,1)
print(f'Execution time: {h}h{m}m{s}s')