import itertools
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
gamma = 0.1
lam = 0.1



##################################
#
#
# Functions
#
#
##################################

def ped_salience_distance_and_factors(ped, n_steps, softmax = False):
	cols = ['unmarked','zebra', 'loc']

	if softmax==False:
		salience_distances = np.array([np.append(ped.ca_salience_distances(), ped._loc)])
		s_factors = np.array([np.append(ped.ca_salience_factors(),ped._loc)])
	else:
		salience_distances = np.array([np.append(ped.ca_salience_distances_softmax(), ped._loc)])
		s_factors = np.array([np.append(ped.ca_salience_factors_softmax(),ped._loc)])

	for i in range(1, n_steps):
		ped._loc += 1

		if softmax == False:
			salience_i = np.append(ped.ca_salience_distances(), ped._loc)
			sf_i = np.append(ped.ca_salience_factors(), ped._loc)
		else:
			salience_i = np.append(ped.ca_salience_distances_softmax(), ped._loc)
			sf_i = np.append(ped.ca_salience_factors_softmax(), ped._loc)			

		salience_distances = np.append(salience_distances, [salience_i], axis=0)
		s_factors = np.append(s_factors, [sf_i], axis=0)
	
	df_sd = pd.DataFrame(columns = cols, data = salience_distances)
	df_sf = pd.DataFrame(columns = cols, data = s_factors)
	
	return {'salience_distances':df_sd, 'salience_factors':df_sf}

def get_utility_costs_of_crossing_alterantives(ped):
	utilities = ped.ca_utilities()
	wt_costs = ped.ca_ww_times()
	ve_costs = ped.ca_vehicle_exposures()

	return np.concatenate((utilities, wt_costs, ve_costs))


def plot_two_series(df, series_a, series_b, label_a, label_b, title, title_suffix = '', dict_markers = None):

	fig = plt.figure(figsize=(12,5))

	fig.gca().set_title(title+title_suffix)

	ax = df[series_a].plot(color='blue', label=series_a)
	ax = df[series_b].plot(color='red', secondary_y=False, label=series_b)

	h1, l1 = ax.get_legend_handles_labels()
	#h2, l2 = ax2.get_legend_handles_labels()

	# Add marker for positions of zebra crossing and destination
	for k,v in dict_markers.items():
		plt.axvline(v, linestyle = '--', linewidth = 0.5)
		plt.annotate(k, (v, ax.get_ylim()[1]))

	fig.legend(h1,l1,loc=2)
	return fig

def plot_utilities_and_costs(df, cols, title, title_suffix, dict_markers = None):
	fig = plt.figure(figsize=(12,5))

	fig.gca().set_title(title+title_suffix)

	colours = ['black', 'blue', 'red']

	# Plot pairs of utilities and costs (for each crossing alternative)
	for i in range(0, len(cols), 2):
		c = colours[i//2]
		ax = df[cols[i]].plot(color=c, linestyle = '-', label=cols[i])
		ax = df[cols[i+1]].plot(color=c, linestyle = '--', secondary_y=False, label=cols[i+1])

	h1, l1 = ax.get_legend_handles_labels()
	#h2, l2 = ax2.get_legend_handles_labels()

	# Add marker for positions of zebra crossing and destination
	for k,v in dict_markers.items():
		plt.axvline(v, linestyle = '--', linewidth = 0.5, color = 'grey')
		plt.annotate(k, (v, ax.get_ylim()[1]))

	fig.legend(h1,l1,loc=2)
	return fig

##################################
#
#
# Salience distance and salience factors
#
#
##################################
zebra = CrossingAlternative(0, None, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = 0)
unmarked = CrossingAlternative(1, None, ctype = mid_block_type, name = 'mid1', vehicle_flow = 0)
crossing_altertives = [unmarked, zebra]

lam = 0.1
dest = road_length/5.0
dict_markers = {'destination':dest, 'zebra':zebra_location}

ped = Ped(0, None, location = 0, speed = ped_walking_speed, destination = dest, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.1, lam = lam, aw = 0.5, a_rate = 1)

dict_data = ped_salience_distance_and_factors(ped, 50)

title_suffix = " lam:{}".format(lam)
fig_salience = plot_two_series(dict_data['salience_distances'], 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Salience Distances',title_suffix = title_suffix, dict_markers = dict_markers)
fig_probs = plot_two_series(dict_data['salience_factors'], 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Salience Factor',title_suffix = title_suffix, dict_markers = dict_markers)

fig_salience.show()
fig_probs.show()


# try with different lambda value

lam = 1

ped = Ped(0, None, location = 0, speed = ped_walking_speed, destination = dest, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.1, lam = lam, aw = 0.5, a_rate = 1)

dict_data = ped_salience_distance_and_factors(ped, 50)

title_suffix = " lam:{}".format(lam)
fig_salience = plot_two_series(dict_data['salience_distances'], 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Salience Distances', title_suffix=title_suffix, dict_markers = dict_markers)
fig_probs = plot_two_series(dict_data['salience_factors'], 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Salience Factor', title_suffix=title_suffix, dict_markers = dict_markers)

fig_salience.show()
fig_probs.show()


# Try with different salience factor calculation (softmax)
lam = 1

ped = Ped(0, None, location = 0, speed = ped_walking_speed, destination = dest, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.1, lam = lam, aw = 0.5, a_rate = 1)

dict_data = ped_salience_distance_and_factors(ped, 50, softmax = True)

title_suffix = " softmax, lam:{}".format(lam)
fig_salience = plot_two_series(dict_data['salience_distances'], 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Salience Distances', title_suffix=title_suffix, dict_markers = dict_markers)
fig_probs = plot_two_series(dict_data['salience_factors'], 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Salience Factor', title_suffix=title_suffix, dict_markers = dict_markers)

fig_salience.show()
fig_probs.show()



#####################################
#
#
# Costs and utility
#
#
#####################################
ped = Ped(0, None, location = 0, speed = ped_walking_speed, destination = dest, 
			crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = gamma, lam = lam, aw = 0.5, a_rate = 1)

ped0 = Ped(0, None, location = 0, speed = ped_walking_speed, destination = road_length/5.0, 
			crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = gamma, lam = lam, aw = 0, a_rate = 1)

ped1 = Ped(0, None, location = 0, speed = ped_walking_speed, destination = road_length/5.0, 
			crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = gamma, lam = lam, aw = 1, a_rate = 1)

pedhalf = Ped(0, None, location = 0, speed = ped_walking_speed, destination = road_length/5.0, 
			crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = gamma, lam = lam, aw = 0.5, a_rate = 1)

zebra = CrossingAlternative(0, None, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = 0)
unmarked = CrossingAlternative(1, None, ctype = mid_block_type, name = 'mid1', vehicle_flow = 0)

crossing_altertives = [unmarked, zebra]

utility_costs_cols = ['unmarked_u','zebra_u', 'unmarked_wt', 'zebra_wt', 'unmarked_ve', 'zebra_ve', 'loc']

# Get utility for 0 vehicle flow and compare to costs for 5 vehicle flow
u0 = np.append(get_utility_costs_of_crossing_alterantives(ped0), ped0._loc)
utility_r0_v0 = np.array([u0])
for i in range(1,50):
	ped0._loc += 1
	ped0.accumulate_ca_activation()
	ui = np.append(get_utility_costs_of_crossing_alterantives(ped0), ped0._loc)
	utility_r0_v0 = np.append(utility_r0_v0, [ui], axis=0)

df_a_r0_v0 = pd.DataFrame(columns = ['unmarked','zebra'], data = ped0.getActivationHistory())

# Increase vehcile flow
zebra._vehicle_flow = 2
unmarked._vehicle_flow = 2
ped0._loc = 0
ped0._crossing_alternatives = crossing_altertives
ped0._ca_activation_history = np.array([[0] * len(ped0._crossing_alternatives)])

u0 = np.append(get_utility_costs_of_crossing_alterantives(ped0), ped0._loc)
utility_r0_v2 = np.array([u0])
for i in range(1,50):
	ped0._loc += 1
	ped0.accumulate_ca_activation()
	ui = np.append(get_utility_costs_of_crossing_alterantives(ped0), ped0._loc)
	utility_r0_v2 = np.append(utility_r0_v2, [ui], axis=0)

df_a_r0_v2 = pd.DataFrame(columns = ['unmarked','zebra'], data = ped0.getActivationHistory())

# Use aw=1 ped
ped1._crossing_alternatives = crossing_altertives
u0 = np.append(get_utility_costs_of_crossing_alterantives(ped1), ped1._loc)
utility_r1_v2 = np.array([u0])
for i in range(1,50):
	ped1._loc += 1
	ped1.accumulate_ca_activation()
	ui = np.append(get_utility_costs_of_crossing_alterantives(ped1), ped1._loc)
	utility_r1_v2 = np.append(utility_r1_v2, [ui], axis=0)

# use half ped
pedhalf._crossing_alternatives = crossing_altertives
u0 = np.append(get_utility_costs_of_crossing_alterantives(pedhalf), pedhalf._loc)
utility_rh_v2 = np.array([u0])
for i in range(1,50):
	pedhalf._loc += 1
	pedhalf.accumulate_ca_activation()
	ui = np.append(get_utility_costs_of_crossing_alterantives(pedhalf), pedhalf._loc)
	utility_rh_v2 = np.append(utility_rh_v2, [ui], axis=0)

df_u_r0_v0 = pd.DataFrame(columns = utility_costs_cols, data = utility_r0_v0)
df_u_r0_v2 = pd.DataFrame(columns = utility_costs_cols, data = utility_r0_v2)
df_u_r1_v2 = pd.DataFrame(columns = utility_costs_cols, data = utility_r1_v2)
df_u_rh_v2 = pd.DataFrame(columns = utility_costs_cols, data = utility_rh_v2)

# Plot the different utility curves
fig_u_r0_v0 = plot_utilities_and_costs(df_u_r0_v0, utility_costs_cols[:-1], 'Utility + Costs', " aw:{}, v:{}".format(0,0), dict_markers =dict_markers)
fig_u_r0_v2 = plot_utilities_and_costs(df_u_r0_v2, utility_costs_cols[:-1], 'Utility + Costs', " aw:{}, v:{}".format(0,2), dict_markers =dict_markers)
fig_u_r1_v2 = plot_utilities_and_costs(df_u_r1_v2, utility_costs_cols[:-1], 'Utility + Costs', " aw:{}, v:{}".format(1,2), dict_markers =dict_markers)
fig_u_rh_v2 = plot_utilities_and_costs(df_u_rh_v2, utility_costs_cols[:-1], 'Utility + Costs', " aw:{}, v:{}".format(0.5,2), dict_markers =dict_markers)

fig_u_r0_v0.show()
fig_u_r0_v2.show()
fig_u_r1_v2.show()
fig_u_rh_v2.show()


# Plot the activation histories for each ped
df_a_r1_v2 = pd.DataFrame(columns = ['unmarked','zebra'], data = ped1.getActivationHistory())
df_a_rh_v2 = pd.DataFrame(columns = ['unmarked','zebra'], data = pedhalf.getActivationHistory())


fig_a_r0_v0 = plot_two_series(df_a_r0_v0, 'unmarked', 'zebra', 'unmarked', 'zebra', 'Activations', title_suffix = " aw:{}, v:{}".format(0,0), dict_markers = dict_markers)
fig_a_r0_v2 = plot_two_series(df_a_r0_v2, 'unmarked', 'zebra', 'unmarked', 'zebra', 'Activations', title_suffix = " aw:{}, v:{}".format(0,2), dict_markers = dict_markers)
fig_a_r1_v2 = plot_two_series(df_a_r1_v2, 'unmarked', 'zebra', 'unmarked', 'zebra', 'Activations', title_suffix = " aw:{}, v:{}".format(1,2), dict_markers = dict_markers)
fig_a_rh_v2 = plot_two_series(df_a_rh_v2, 'unmarked', 'zebra', 'unmarked', 'zebra', 'Activations', title_suffix = " aw:{}, v:{}".format(0.5,2), dict_markers = dict_markers)

fig_a_r0_v0.show()
fig_a_r0_v2.show()
fig_a_r1_v2.show()
fig_a_rh_v2.show()