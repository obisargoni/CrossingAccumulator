# crossing accumulator simualtion

import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from importlib import reload
 
import CrossingAccumulator
CrossingAccumulator = reload(CrossingAccumulator)
from CrossingAccumulator import CrossingAlternative, Ped

road_length = 50

zebra_location = road_length * 0.75
zebra_wait_time = 0
zebra_type = 'zebra'

mid_block_wait_time = 3
mid_block_type = 'mid_block'

ped_start_location = 0
ped_walking_speed = 3

# initialise two crossing alternatives, one a zebra crossing and one mid block crossing
zebra = CrossingAlternative(location = zebra_location, wait_time = zebra_wait_time, ctype = zebra_type, name = 'z1')
mid_block = CrossingAlternative(wait_time = mid_block_wait_time, ctype = mid_block_type, name = 'mid1')

# Initialise a pedestrian agent

# Crossing alternatives with salience factors
p1_crossing_altertives = [(mid_block, 1), (zebra, 1)]
p1 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length, crossing_altertives = p1_crossing_altertives, road_length = road_length, lam = 1)
p2 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length, crossing_altertives = p1_crossing_altertives, road_length = road_length, lam = 5)

p3 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length*0.5, crossing_altertives = p1_crossing_altertives, road_length = road_length, lam = 1)
p4 = Ped(location = ped_start_location, speed = ped_walking_speed, destination = road_length*0.5, crossing_altertives = p1_crossing_altertives, road_length = road_length, lam = 5)


# Run simulation for ped 1
def run_sim(ped):
	while ped.getLoc() < road_length:

		# update ped's perceptions of crossing alternative utilities
		ped.update_utility_accumulator()

		# move the ped along
		ped.walk()

run_sim(p1)
run_sim(p2)
run_sim(p3)
run_sim(p4)

# Then plots results
p1_acc_utilities = p1.getAccumulatedUtilityHistory()
p2_acc_utilities = p2.getAccumulatedUtilityHistory()
p3_acc_utilities = p3.getAccumulatedUtilityHistory()
p4_acc_utilities = p4.getAccumulatedUtilityHistory()
