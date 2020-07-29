import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt

from mesa.batchrunner import BatchRunner
 
from importlib import reload
import CrossingAccumulator
CrossingAccumulator = reload(CrossingAccumulator)
from CrossingAccumulator import CrossingModel, CrossingAlternative, Ped

road_length = 50
road_width = 10
zebra_location = road_length * 0.75
zebra_type = 'zebra'
mid_block_type = 'unmarked'

ped_start_location = 0
ped_walking_speed = 3
lam = 10


##################################
#
#
# Salience and sampling probability
#
#
##################################
zebra = CrossingAlternative(0, None, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = 0)
unmarked = CrossingAlternative(1, None, ctype = mid_block_type, name = 'mid1', vehicle_flow = 0)
crossing_altertives = [(unmarked, 1), (zebra, 1)]

ped = Ped(0, None, location = 0, speed = ped_walking_speed, destination = road_length/5.0, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.9, lam = 1, aw = 0, a_rate = 1)

cols = ['unmarked','zebra', 'loc']
saliences = np.array([np.append(ped.ca_saliences(), ped._loc)])
probs = np.array([np.append(scipy.special.softmax(lam * ped.ca_saliences()),ped._loc)])
for i in range(1, 50):
	ped._loc += 1
	salience_i = np.append(ped.ca_saliences(), ped._loc)
	probs_i = np.append(scipy.special.softmax(lam * salience_i[:-1]), ped._loc)

	saliences = np.append(saliences, [salience_i], axis=0)
	probs = np.append(probs, [probs_i], axis=0)


# Now plot
df_salience = pd.DataFrame(columns = cols, data = saliences)
df_probs = pd.DataFrame(columns = cols, data = probs)

def plot_two_series(df, series_a, series_b, label_a, label_b, title):

	fig = plt.figure(figsize=(12,5))

	fig.gca().set_title(title)

	ax1 = df[series_a].plot(color='blue', label=series_a)
	ax2 = df[series_b].plot(color='red', secondary_y=False, label=series_b)

	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()

	fig.legend(h1+h2, l1+l2, loc=2)
	return fig

fig_salience = plot_two_series(df_salience, 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Crossing Saliences')
fig_probs = plot_two_series(df_probs, 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Sampling Probability')

fig_salience.show()
fig_probs.show()


#####################################
#
#
# Costs and utility
#
#
#####################################

def get_utility_of_crossing_alterantives(ped, crossing_altertives):
	utilities = np.zeros(len(crossing_altertives))

	for i, (ca, sf) in enumerate(crossing_altertives):
		ui = ped.ca_utility(ca)
		utilities[i] = ui

	return utilities

ped0 = Ped(0, None, location = 0, speed = ped_walking_speed, destination = road_length/5.0, 
			crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.9, lam = 1, aw = 0, a_rate = 1)

ped1 = Ped(0, None, location = 0, speed = ped_walking_speed, destination = road_length/5.0, 
			crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.9, lam = 1, aw = 1, a_rate = 1)

pedhalf = Ped(0, None, location = 0, speed = ped_walking_speed, destination = road_length/5.0, 
			crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.9, lam = 1, aw = 0.5, a_rate = 1)

zebra = CrossingAlternative(0, None, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = 0)
unmarked = CrossingAlternative(1, None, ctype = mid_block_type, name = 'mid1', vehicle_flow = 0)

crossing_altertives = [(unmarked, 1), (zebra, 1)]

utility_cols = ['unmarked_u','zebra_u', 'loc']

# Get utility for 0 vehicle flow and compare to costs for 5 vehicle flow
u0 = np.append(get_utility_of_crossing_alterantives(ped0, crossing_altertives), ped0._loc)
utility_r0_v0 = np.array([u0])
for i in range(1,50):
	ped0._loc += 1
	ui = np.append(get_utility_of_crossing_alterantives(ped0, crossing_altertives), ped0._loc)
	utility_r0_v0 = np.append(utility_r0_v0, [ui], axis=0)


# Increase vehcile flow
zebra._vehicle_flow = 2
unmarked._vehicle_flow = 2
ped0._loc = 0

u0 = np.append(get_utility_of_crossing_alterantives(ped0, crossing_altertives), ped0._loc)
utility_r0_v2 = np.array([u0])
for i in range(1,50):
	ped0._loc += 1
	ui = np.append(get_utility_of_crossing_alterantives(ped0, crossing_altertives), ped0._loc)
	utility_r0_v2 = np.append(utility_r0_v2, [ui], axis=0)


# Use aw=1 ped
u0 = np.append(get_utility_of_crossing_alterantives(ped1, crossing_altertives), ped1._loc)
utility_r1_v2 = np.array([u0])
for i in range(1,50):
	ped1._loc += 1
	ui = np.append(get_utility_of_crossing_alterantives(ped1, crossing_altertives), ped1._loc)
	utility_r1_v2 = np.append(utility_r1_v2, [ui], axis=0)

u0 = np.append(get_utility_of_crossing_alterantives(pedhalf, crossing_altertives), pedhalf._loc)
utility_rh_v2 = np.array([u0])
for i in range(1,50):
	pedhalf._loc += 1
	ui = np.append(get_utility_of_crossing_alterantives(pedhalf, crossing_altertives), pedhalf._loc)
	utility_rh_v2 = np.append(utility_rh_v2, [ui], axis=0)

df_u_r0_v0 = pd.DataFrame(columns = utility_cols, data = utility_r0_v0)
df_u_r0_v2 = pd.DataFrame(columns = utility_cols, data = utility_r0_v2)
df_u_r1_v2 = pd.DataFrame(columns = utility_cols, data = utility_r1_v2)
df_u_rh_v2 = pd.DataFrame(columns = utility_cols, data = utility_rh_v2)


# Plot the different utility curves
fig_u_r0_v0 = plot_two_series(df_u_r0_v0, 'unmarked_u', 'zebra_u', 'unmarked crossing', 'zebra crossing', 'Crossing Utilities')
fig_u_r0_v2 = plot_two_series(df_u_r0_v2, 'unmarked_u', 'zebra_u', 'unmarked crossing', 'zebra crossing', 'Crossing Utilities')
fig_u_r1_v2 = plot_two_series(df_u_r1_v2, 'unmarked_u', 'zebra_u', 'unmarked crossing', 'zebra crossing', 'Crossing Utilities')
fig_u_rh_v2 = plot_two_series(df_u_rh_v2, 'unmarked_u', 'zebra_u', 'unmarked crossing', 'zebra crossing', 'Crossing Utilities')