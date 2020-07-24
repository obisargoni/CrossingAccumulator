# crossing accumulator simualtion

import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from importlib import reload
 
import CrossingAccumulator
CrossingAccumulator = reload(CrossingAccumulator)
from CrossingAccumulator import CrossingAlternative, Ped

ped_acumulator_rate = 1

road_length = 50
road_width = 10

zebra_location = road_length * 0.75
zebra_wait_time = 0
zebra_type = 'zebra'

mid_block_wait_time = 3
mid_block_type = 'mid_block'

ped_start_location = 0
ped_walking_speed = 3

vehicle_flow = 1

# initialise two crossing alternatives, one a zebra crossing and one mid block crossing
zebra = CrossingAlternative(location = zebra_location, wait_time = zebra_wait_time, ctype = zebra_type, name = 'z1', vehicle_flow = vehicle_flow)
mid_block = CrossingAlternative(wait_time = mid_block_wait_time, ctype = mid_block_type, name = 'mid1', vehicle_flow = vehicle_flow)

# Initialise a pedestrian agent

# Crossing alternatives with salience factors
p1_crossing_altertives = [(mid_block, 1), (zebra, 1)]
p1 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = 1, r = 1, activation_threshold_factor = 1.1)
p2 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = 5, r = 1, activation_threshold_factor = 1.1)

p3 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length*0.5, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = 1, r = 1, activation_threshold_factor = 1.1)
p4 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length*0.5, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = 5, r = 1, activation_threshold_factor = 1.1)


def run_sim(lam, r, activation_threshold_factor):
	ped = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length, crossing_altertives = p1_crossing_altertives, road_length = road_length, road_width = road_width, lam = lam, r = r, activation_threshold_factor = activation_threshold_factor)
	while (ped.getLoc() < road_length) and (ped.getChosenCA() is None):

		# update ped's perceptions of crossing alternative utilities
		for i in range(ped_acumulator_rate):
			ped.accumulate_ca_activation()

		# move the ped along
		ped.walk()
	return ped

def run_sim_get_ca_type(lam, r, activation_threshold_factor):
	ped = run_sim(lam, r, activation_threshold_factor)

	# Record crossing type chosen
	chosen_ca = ped.getChosenCA()
	if chosen_ca is None:
		chosen_ca_type = 'none'
	else:
		chosen_ca_type = chosen_ca.getCrossingType()

	return chosen_ca_type

def run_sim_multiple(list_pedestrian_params):
	peds = np.array([])
	for params in list_pedestrian_params:
		ped = run_sim(*params)
		peds = np.append(peds, ped)
	return peds

def run_sim_get_ca_type_multiple(list_pedestrian_params):
	# Run simulation for ped 1
	chosen_ca_types = np.array([])

	for params in list_pedestrian_params:
		chosen_type = run_sim_get_ca_type(*params)
		chosen_ca_types = np.append(chosen_ca_types, chosen_type)
	return chosen_ca_types


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