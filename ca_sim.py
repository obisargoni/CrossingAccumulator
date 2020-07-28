# crossing accumulator simualtion

import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt

from mesa.batchrunner import BatchRunner
 
from importlib import reload
import CrossingAccumulator
CrossingAccumulator = reload(CrossingAccumulator)
from CrossingAccumulator import CrossingModel

road_length = 50
road_width = 10

ped_start_location = 0
ped_walking_speed = 3

fixed_params = {"road_width": road_width,
               	"road_length": road_length,
               	"ped_origin": ped_start_location,
               	"ped_speed": ped_walking_speed,
               	"alpha" : 1.2,
               	"gamma" : 0.9,
               	"n_peds" : 1}

variable_params = {	"ped_destination": [road_length, road_length*0.5],
					"lam": range(1, 10, 1),
					"r": np.linspace(0,2,20),
					'a_rate': range(1,3),
					"vehicle_flow": range(0,5)}

variable_params = {	"ped_destination": [road_length],
					"lam": [1,10],
					"r": [0,1,2],
					'a_rate': [1,5],
					"vehicle_flow": [0,5]}

agent_reporters={"CrossingType":"chosenCAType"}

def model_crossing_choices(model):
    return model.crossing_choice

batch_run = BatchRunner(CrossingModel,
                        variable_params,
                        fixed_params,
                        iterations=50,
                        max_steps=100,
                        model_reporters = {"CrossingChoice":model_crossing_choices})
batch_run.run_all()

batch_data = batch_run.get_model_vars_dataframe()

# Column titles are wrong, need to rename
batch_data_columns = list(variable_params.keys()) + ['Run', 'CrossingChoice'] + list(fixed_params.keys())
batch_data.columns = batch_data_columns

batch_data['CrossingChoice'] = batch_data['CrossingChoice'].fillna("none")
batch_data['BatchRun'] = batch_data['Run'].map(lambda r: r // 50)
batch_data.drop('Run', axis=1, inplace=True)

param_columns = [c for c in batch_data.columns if c not in ['Run','CrossingChoice']]

# Group by param columns and calculate aggregate crossing proportions
def agg_crossing_choice(df, col = 'CrossingChoice'):
	counts = df[col].value_counts().to_dict()

	for k,v in counts.items():
		df[k] = v
	df.drop(col, axis=1, inplace=True)
	return df.drop_duplicates()

agg_batch_data = batch_data.groupby('BatchRun').apply(agg_crossing_choice)

