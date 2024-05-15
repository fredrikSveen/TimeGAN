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
from data_loading import complex_sine_loading


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
seq_len = 100
n_samples = 1000
n_iterations_list = [100000]
dim = 6


for n_iterations in n_iterations_list:

    ori_data = complex_sine_loading(n_samples, dim, seq_len)
        
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
    filepath = f'complex_synthetic_sines/comp_syn_sine_{n_iterations}_{dim}_{seq_len}_{timestamp}.json'
    with open(filepath, 'w') as file:
        json.dump(generated_data.tolist(), file)

# generated_df.to_csv(f'{filepath}.csv')
# generated_df.to_csv(f'{filepath}_indexless.csv', index=False)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
h = round(elapsed_time//(60*60), 0)
m = round((elapsed_time - h*(60*60))//60, 0)
s = round(elapsed_time - h*(60*60) - m*60,1)
print(f'Execution time: {h}h{m}m{s}s')