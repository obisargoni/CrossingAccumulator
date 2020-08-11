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

##################################
#
#
# Functions
#
#
##################################
def get_utility_costs_of_crossing_alterantives(ped, model):
	utilities = ped.ca_utilities(model = model)
	attributes = ped.cas_attributes_sampling()
	wt_costs = attributes[:,0]
	ve_costs = attributes[:,1]

	return np.concatenate((utilities, wt_costs, ve_costs))

def ped_salience_distance_and_factors(ped, n_steps, salience_type = 'ca'):
	cols = ['unmarked','zebra', 'loc']

	if salience_type == 'ca':
		salience_distances = np.array([np.append(ped.ca_salience_distances_to_ca(), ped._loc)])
	else:
		salience_distances = np.array([np.append(ped.ca_salience_distances_to_dest(), ped._loc)])
	
	s_factors = np.array([np.append(ped.ca_salience_factors_softmax(salience_type),ped._loc)])

	for i in range(1, n_steps):
		ped._loc += 1

		if salience_type == 'ca':
			salience_i = np.append(ped.ca_salience_distances_to_ca(), ped._loc)
		else:
			salience_i = np.append(ped.ca_salience_distances_to_dest(), ped._loc)

		sf_i = np.append(ped.ca_salience_factors_softmax(salience_type),ped._loc)

		salience_distances = np.append(salience_distances, [salience_i], axis=0)
		s_factors = np.append(s_factors, [sf_i], axis=0)
	
	df_sd = pd.DataFrame(columns = cols, data = salience_distances)
	df_sf = pd.DataFrame(columns = cols, data = s_factors)
	
	return {'salience_distances':df_sd, 'salience_factors':df_sf}

def plot_utilities_and_costs(df, cols, labels, title, title_suffix, vehicle_flow_col = None, ylab = None, xlab = None, dict_markers = None):
	fig = plt.figure(figsize=(12,5))

	fig.gca().set_title(title+title_suffix)

	linestyles = ['-', '--', ':']
	colours = ['blue','red']

	# Plot pairs of utilities and costs (for each crossing alternative)
	for i in range(0, len(cols), 2):
		l = linestyles[i//2]
		ax = df[cols[i]].plot(color=colours[0], linestyle = l, label=labels[i])
		ax = df[cols[i+1]].plot(color=colours[1], linestyle = l, secondary_y=False, label=labels[i+1])

	if vehicle_flow_col is not None:
		ax = df[vehicle_flow_col].plot(color='black', linestyle = 'dotted', label=vehicle_flow_col)

	if ylab is not None:
		ax.set_ylabel(ylab)
	if xlab is not None:
		ax.set_xlabel(xlab)

	h1, l1 = ax.get_legend_handles_labels()

	# Add marker for positions of zebra crossing and destination
	for k,v in dict_markers.items():
		plt.axvline(v, linestyle = '--', linewidth = 0.5, color = 'grey')
		plt.annotate(k, (v - 7, ax.get_ylim()[0] + 0.05), xycoords = 'data')

	fig.legend(h1,l1,loc=4)
	return fig


def plot_dists_and_probs(df, cols, labels, title, title_suffix, ylab1 = None, ylab2 = None, xlab = None, dict_markers = None):
	fig = plt.figure(figsize=(12,5))

	fig.gca().set_title(title+title_suffix)

	colours = ['blue','red']

	# Plot pairs of utilities and costs (for each crossing alternative)
	for i in range(0, len(cols), 2):
		c = colours[i//2]
		ax1 = df[cols[i]].plot(color=c, linestyle = '--', label=labels[i])
		ax2 = df[cols[i+1]].plot(color=c, linestyle = '-', secondary_y=True, label=labels[i+1])

	if ylab1 is not None:
		ax1.set_ylabel(ylab1)
	if ylab2 is not None:
		ax2.set_ylabel(ylab2)
	if xlab is not None:
		ax1.set_xlabel(xlab)
		ax2.set_xlabel(xlab)

	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()

	# Add marker for positions of zebra crossing and destination
	for k,v in dict_markers.items():
		plt.axvline(v, linestyle = '--', linewidth = 0.5, color = 'grey')
		plt.annotate(k, (v - 7, ax2.get_ylim()[0] + 0.02), xycoords = 'data')

	fig.legend(h1+h2,l1+l2,loc=4)
	return fig

def plot_two_series(df, cols, labels, title, title_suffix, error_cols = None, x=None, vehicle_flow_col = None, dict_markers = None, xlab = None, ylab = None):
	if x is not None:
		df.set_index(x, inplace=True)
	
	fig = plt.figure(figsize=(12,5))

	fig.gca().set_title(title+title_suffix)

	if error_cols is None:
		ax = df[cols[0]].plot(color='blue', label=labels[0])
		ax = df[cols[1]].plot(color='red', secondary_y=False, label=labels[1])
	else:
		ax = df[cols[0]].plot(color='blue', label=labels[0])
		ax = ax.fill_between(df.index, df[cols[0]] - df[error_cols[0]], df[cols[0]] + df[error_cols[0]], color = 'blue', alpha = 0.5)
		ax = df[cols[1]].plot(color='red', secondary_y=False, label=labels[1])
		ax = ax.fill_between(df.index, df[cols[1]] - df[error_cols[1]], df[cols[1]] + df[error_cols[1]], color = 'red', alpha = 0.5)

	if vehicle_flow_col is not None:
		ax = df[vehicle_flow_col].plot(color='black', linestyle = 'dotted', label=vehicle_flow_col)

	if ylab is not None:
		ax.set_ylabel(ylab)
	if xlab is not None:
		ax.set_xlabel(xlab)

	h1, l1 = ax.get_legend_handles_labels()

	# Add marker for positions of zebra crossing and destination
	for k,v in dict_markers.items():
		plt.axvline(v, linestyle = '--', linewidth = 0.5)
		if k == 'Choice\nMade':
			plt.annotate(k, (v-4, ax.get_ylim()[0]+0.1))
		else:
			plt.annotate(k, (v+0.5, ax.get_ylim()[0]+0.1))


	fig.legend(h1,l1,loc=4)
	return fig

road_length = 50
road_width = 10
zebra_location = road_length * 0.75
zebra_type = 'zebra'
mid_block_type = 'unmarked'

ped_start_location = 0
ped_walking_speed = 3
gamma = 0.9
epsilon = 2
lam = 1
a_rate = 1
dest = road_length/3
dict_markers = {'Unsignalised\nCrossing':zebra_location}

zebra = CrossingAlternative(0, None, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = 0)
unmarked = CrossingAlternative(1, None, ctype = mid_block_type, name = 'mid1', vehicle_flow = 0)
crossing_altertives = [unmarked, zebra]

ped = Ped(0, None, location = 0, speed = ped_walking_speed, destination = dest, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, epsilon = 1.2, gamma = 0.9, lam = lam, alpha = 0.5, a_rate = 1)

dict_data = ped_salience_distance_and_factors(ped, 50, salience_type = 'ca')

# Join the data together to that distances and saliences can be plotted on the same figure
df_data = pd.merge(dict_data['salience_distances'], dict_data['salience_factors'], on = 'loc', suffixes = ('_dist', '_prob'))

cols = ['unmarked_dist','unmarked_prob', 'zebra_dist','zebra_prob']
labels = ['Unmarked $d_j$','Unmarked probability', 'Unsignalised $d_j$','Unsignalised probability']

fig_probs_sals = plot_dists_and_probs(df_data, cols, labels, "Sampling Probabilities", "", ylab1 = "$d_j$", ylab2 = "p", xlab = "$x_{ped}$", dict_markers = dict_markers)
fig_probs_sals.show()
fig_probs_sals.savefig(".\\img\\distances_probabilities_l1.png")


#################################
#
# Get CA attributes and utilities and plot these
#
#################################

# update vehicle flow of crossing alternatives
# Increase vehcile flow
zebra._vehicle_flow = 0.5
unmarked._vehicle_flow = 0.5

pedhalf = Ped(0, None, location = 0, speed = ped_walking_speed, destination = dest, 
			crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, epsilon = epsilon, gamma = gamma, lam = lam, alpha = 0.5, a_rate = 1)

utility_costs_cols = ['unmarked_u','zebra_u', 'unmarked_wt', 'zebra_wt', 'unmarked_ve', 'zebra_ve', 'loc']
utility_costs_labels = ['Unmarked Utility','Unsignalised Utility', 'Unmarked Time Attr.', 'Unsignalised Time Attr.', 'Unmarked Exposure Attr.', 'Unsignalised Exposure Attr.', 'loc']

# Get utility for 0 vehicle flow and compare to costs for 5 vehicle flow
model = 'sampling'
u0 = np.append(get_utility_costs_of_crossing_alterantives(pedhalf, model), pedhalf._loc)
utility_r0_v0 = np.array([u0])
for i in range(1,50):
	pedhalf._loc += 1
	pedhalf.accumulate_ca_activation()
	ui = np.append(get_utility_costs_of_crossing_alterantives(pedhalf, model), pedhalf._loc)
	utility_r0_v0 = np.append(utility_r0_v0, [ui], axis=0)

df_u_r0_v2 = pd.DataFrame(columns = utility_costs_cols, data = utility_r0_v0)
dict_markers['Destination'] = dest
fig_u_r0_v2 = plot_utilities_and_costs(df_u_r0_v2, utility_costs_cols[:-1], utility_costs_labels[:-1], 'Attributes and Utilities', "\n $\\alpha$ = {}".format(0.5), xlab = "$x_{ped}$", dict_markers =dict_markers)
fig_u_r0_v2.show()
fig_u_r0_v2.savefig(".\\img\\attrs_utilities_a0.5_v0.5.png")



################################
#
# Attributes and Utilities with varying traffic flow
#
################################
gap_size = 5
v_vary_low = [0.5]*road_length
for i in range(-gap_size, gap_size):
	ind = int(dest + i)
	v_vary_low[ind]=0

lam = 1
alpha = 0.5
vf = v_vary_low
suffs = " lam:{}, alpha:{}".format(lam, alpha)
model_vlow = CrossingModel(	ped_origin = ped_start_location, ped_destination = dest, road_length = road_length, road_width = road_width, vehicle_flow = vf, 
						epsilon = epsilon, gamma = gamma, ped_speed = ped_walking_speed, lam = lam, alpha = alpha, a_rate = a_rate)
while model_vlow.running:
	model_vlow.step()

# Get the utilities and attributes from the model_vlow
ped_cost_cols = ['unmarked_wt','unmarked_ve', 'zebra_wt', 'zebra_ve']
ped_utility_cols = ['unmarked_u','zebra_u']

df_attrs = pd.DataFrame(columns = ped_cost_cols, data = model_vlow.ped._ca_costs_history[1:])
df_utilities = pd.DataFrame(columns = ped_utility_cols, data = model_vlow.ped._ca_utility_history[1:])

df_u_a = pd.merge(df_utilities, df_attrs, left_index = True, right_index = True)
df_u_a['Vehicle Flow'] = vf

fig_u_a= plot_utilities_and_costs(df_u_a, utility_costs_cols[:-1], utility_costs_labels[:-1], 'Attributes and Utilities with Varied Vehicle Flow', "\n $\\alpha$ = {}".format(0.5), vehicle_flow_col = 'Vehicle Flow', xlab = "$x_{ped}$", dict_markers =dict_markers)
fig_u_a.show()
fig_u_a.savefig(".\\img\\attrs_utilities_a0.5_v_vary.png")



##############################
#
# Plot activation history and crossing choice
#
##############################
activation_cols = ['unmarked_a','zebra_a']
activation_labels = ['Unmaked Activation', 'Unsignalised Activation']
df_activations = pd.DataFrame(columns = activation_cols, data = model_vlow.ped.getActivationHistory()[1:])
df_activations['Vehicle Flow'] = v_vary_low

# Add in rolling errors
# Doesn't make sense to calculate sd of trend data
'''
df_activations['unmarked_sd'] = df_activations['unmarked_a'].expanding(1).std()
df_activations['zebra_sd'] = df_activations['zebra_a'].expanding(1).std()
error_cols = ['unmarked_sd', 'zebra_sd']
'''

dict_markers['Choice\nMade'] = model_vlow.choice_step
f_act = plot_two_series(df_activations, activation_cols, activation_labels, 'Accumulated Activation with Varied Vehicle Flow', "\n $\\alpha$ = {}".format(0.5), vehicle_flow_col = 'Vehicle Flow', dict_markers = dict_markers, ylab = 'Activation', xlab = 'Tick')
f_act.show()
f_act.savefig(".\\img\\activation_a0.5_v_vary.png")



#################################
#
# Run model_vhigh with higher vehicle flow, shorter gap, show choice made is different
#
#################################
gap_size = 3
v_vary_high = [3]*road_length
for i in range(-gap_size, gap_size):
	ind = int(dest + i)
	v_vary_high[ind]=0

lam = 1
alpha = 0.5
vf = v_vary_high
suffs = " lam:{}, alpha:{}".format(lam, alpha)
model_vhigh = CrossingModel(	ped_origin = ped_start_location, ped_destination = dest, road_length = road_length, road_width = road_width, vehicle_flow = vf, 
						epsilon = epsilon, gamma = gamma, ped_speed = ped_walking_speed, lam = lam, alpha = alpha, a_rate = a_rate)
while model_vhigh.running:
	model_vhigh.step()

df_activations = pd.DataFrame(columns = activation_cols, data = model_vhigh.ped.getActivationHistory()[1:])
df_activations['Vehicle Flow'] = v_vary_high

dict_markers['Choice\nMade'] = model_vhigh.choice_step
f_act = plot_two_series(df_activations, activation_cols, activation_labels, 'Accumulated Activation with Varied Vehicle Flow', "\n $\\alpha$ = {}".format(0.5), vehicle_flow_col = 'Vehicle Flow', dict_markers = dict_markers, ylab = 'Activation', xlab = 'Tick')
f_act.show()
f_act.savefig(".\\img\\activation_a0.5_v_vary_high.png")