# crossing accumulator simualtion

import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from importlib import reload
 
import CrossingAccumulator
CrossingAccumulator = reload(CrossingAccumulator)
from CrossingAccumulator import CrossingModel

ped_acumulator_rate = 1

road_length = 50
road_width = 10

ped_start_location = 0
ped_walking_speed = 3


# Initialise a pedestrian agent
'''
p1 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = 1, r = 1, activation_threshold_factor = 1.1)
p2 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = 5, r = 1, activation_threshold_factor = 1.1)

p3 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length*0.5, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = 1, r = 1, activation_threshold_factor = 1.1)
p4 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length*0.5, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = 5, r = 1, activation_threshold_factor = 1.1)


# Generate a pedestrian population out of lambdas and activation_thresholds
lams1 = np.random.choice(range(1,4), 50)
lams2 = np.random.choice(range(4,7), 50)
activation_threshold_factor1 = np.random.choice(np.linspace(0.5,1.0,6, endpoint=True), 50)
activation_threshold_factor2 = np.random.choice(np.linspace(1.0,1.6,6, endpoint=True), 50)
rs = np.ones(50)

pop1 = zip(lams1, rs, activation_threshold_factor1)
pop2 = zip(lams1, rs, activation_threshold_factor2)
pop3 = zip(lams2, rs, activation_threshold_factor1)
pop4 = zip(lams2, rs, activation_threshold_factor2)

# Get chrossing choices of these different populations of peds
types1 = run_sim_get_ca_type_multiple(pop1)
types2 = run_sim_get_ca_type_multiple(pop2)
types3 = run_sim_get_ca_type_multiple(pop3)
types4 = run_sim_get_ca_type_multiple(pop4)

peds1 = run_sim_multiple(pop1)

dfTypes = pd.DataFrame({'pop1':types1, 'pop2':types2, 'pop3':types3, 'pop4':types4})
'''

model = CrossingModel(	ped_origin = ped_start_location, ped_destination = road_length, road_length = road_length, road_width = road_width, 
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