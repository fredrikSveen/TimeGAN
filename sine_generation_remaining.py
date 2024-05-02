## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import datetime
import time
import json
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation, sine_data_loading


# get the start time
st = time.time()

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['batch_size'] = 128

## Data loading
data_name = 'sine'
n_samples = 1000

# Generating missing 1d sine with 1000 iterations
dim = 1
seq_len = 25
n_iterations = 1000
ori_data = sine_data_loading(n_samples, dim, seq_len)
            
print(data_name + ' dataset is ready.')
print(f'Using {dim} dimensions and {n_iterations} iterations')

# Run TimeGAN
parameters['iterations'] = n_iterations
generated_data = timegan(ori_data, parameters, reproduce=False)   
print('Finish Synthetic Data Generation')

# Save generated data to csv
x = datetime.datetime.now()

timestamp = x.strftime("%d%m%y_%Hh%M")
# generated_df = list_to_df(generated_data)
filepath = f'synthetic_sines/syn_sine_{n_iterations}_{dim}_{seq_len}_{timestamp}.json'
with open(filepath, 'w') as file:
    json.dump(generated_data.tolist(), file)


# Generating missing 9d sine with 50 000 iterations

dim = 9
seq_len = 200
n_iterations = 50000
ori_data = sine_data_loading(n_samples, dim, seq_len)
            
print(data_name + ' dataset is ready.')
print(f'Using {dim} dimensions and {n_iterations} iterations')

# Run TimeGAN
parameters['iterations'] = n_iterations
generated_data = timegan(ori_data, parameters, reproduce=False)   
print('Finish Synthetic Data Generation')

# Save generated data to csv
x = datetime.datetime.now()

timestamp = x.strftime("%d%m%y_%Hh%M")
# generated_df = list_to_df(generated_data)
filepath = f'synthetic_sines/syn_sine_{n_iterations}_{dim}_{seq_len}_{timestamp}.json'
with open(filepath, 'w') as file:
    json.dump(generated_data.tolist(), file)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
h = round(elapsed_time//(60*60), 0)
m = round((elapsed_time - h*(60*60))//60, 0)
s = round(elapsed_time - h*(60*60) - m*60,1)
print(f'Execution time: {h}h{m}m{s}s')