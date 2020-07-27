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

'''
model = CrossingModel(	ped_origin = ped_start_location, ped_destination = road_length, road_length = road_length, road_width = road_width, vehicle_flow = 1, 
						alpha = 1.2, gamma = 0.9, ped_speed = ped_walking_speed, lam_range = range(1,4), r_range = np.linspace(0.9,1.1,3, endpoint=True), 
						a_rate_range = [1], n_peds = 50)
while model.running:
	model.step()

# Get the data
crossing_types = model.datacollector.get_agent_vars_dataframe().unstack()

# Find out proportion using different crossing types
def filter_none_types(col):
	col = col.dropna()
	col.index = np.arange(len(col))
	return col
crossing_proportions = crossing_types.apply(filter_none_types).iloc[0]
print(crossing_proportions.value_counts(dropna=False))
'''