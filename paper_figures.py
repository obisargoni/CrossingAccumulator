import itertools
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mesa.batchrunner import BatchRunner
 
from importlib import reload
import CrossingAccumulator
CrossingAccumulator = reload(CrossingAccumulator)
from CrossingAccumulator import CrossingModel, CrossingAlternative, Ped, Road

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

	fig.suptitle(title+title_suffix)

	# create grid for different subplots
	spec = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], wspace=0.5, hspace=0.1, height_ratios=[3, 1])
	 
	# initializing x,y axis value
	x = np.arange(0, 10, 0.1)
	y = np.cos(x)
	 
	# ax0 will take 0th position in
	# geometry(Grid we created for subplots),
	# as we defined the position as "spec[0]"
	ax0 = fig.add_subplot(spec[0])
	ax1 = fig.add_subplot(spec[1])

	linestyles = ['-', '--', ':']
	colours = ['blue','red']

	# Plot pairs of utilities and costs (for each crossing alternative)
	for i in range(0, len(cols), 2):
		l = linestyles[i//2]
		ax0 = df[cols[i]].plot(ax=ax0, color=colours[0], linestyle = l, label=labels[i])
		ax0 = df[cols[i+1]].plot(ax=ax0, color=colours[1], linestyle = l, secondary_y=False, label=labels[i+1])

	if vehicle_flow_col is not None:
		ax1 = df[vehicle_flow_col].plot(ax=ax1, color='black', linestyle = 'dotted', label=vehicle_flow_col)

	if ylab is not None:
		ax0.set_ylabel(ylab)
	if xlab is not None:
		ax1.set_xlabel(xlab)

	ax1.set_ylabel(r"vehicles $s^{-1}$", fontsize=9)

	ax0.get_xaxis().set_visible(False)
	ax0.spines.right.set_visible(False)
	ax0.spines.top.set_visible(False)
	ax1.spines.right.set_visible(False)
	ax1.spines.top.set_visible(False)

	h1, l1 = ax0.get_legend_handles_labels()

	# Add marker for positions of zebra crossing and destination
	for k,v in dict_markers.items():
		plt.sca(ax0)
		plt.axvline(v, linestyle = '--', linewidth = 0.5, color = 'grey')

		plt.sca(ax1)
		plt.axvline(v, linestyle = '--', linewidth = 0.5, color = 'grey')
		plt.annotate(k, (v - 6, ax1.get_ylim()[0] + 0.15), xycoords = 'data')

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
		plt.annotate(k, (v - 5.3, ax2.get_ylim()[0] + 0.02), xycoords = 'data')

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
			plt.annotate(k, (v-3.5, ax.get_ylim()[0]+0.1))
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
lam = 0.5
a_rate = 1
dest = road_length/3
dict_markers = {'Dedicated\nCrossing':zebra_location}

v_low = [(i%5)//4 for i in range(51)]
v_high = [1 for i in range(51)]


zebra = CrossingAlternative(0, None, location = zebra_location, ctype = zebra_type, name = 'z1')
unmarked = CrossingAlternative(1, None, ctype = mid_block_type, name = 'mid1')
crossing_altertives = [unmarked, zebra]

i=0
road = Road(i, None, length=road_length, width=road_width, crossing_altertives=crossing_altertives, vehicle_addition_times=v_low)

ped = Ped(i+1, None, location = 0, speed = ped_walking_speed, destination = dest, road=road, epsilon = 1.2, gamma = 0.9, lam = lam, alpha = 0.5, a_rate = 1)

dict_data = ped_salience_distance_and_factors(ped, 50, salience_type = 'ca')

# Join the data together to that distances and saliences can be plotted on the same figure
df_data = pd.merge(dict_data['salience_distances'], dict_data['salience_factors'], on = 'loc', suffixes = ('_dist', '_prob'))

cols = ['unmarked_dist','unmarked_prob', 'zebra_dist','zebra_prob']
labels = ['Informal $d_j$','Informal probability', 'Dedicated $d_j$','Dedicated probability']

fig_probs_sals = plot_dists_and_probs(df_data, cols, labels, "Sampling Probabilities", "", ylab1 = "$d_j$", ylab2 = "p", xlab = "$P(t)$", dict_markers = dict_markers)
#fig_probs_sals.show()
fig_probs_sals.savefig(".\\img\\distances_probabilities_l1.png")


#################################
#
# Get CA attributes and utilities and plot these
#
#################################

#pedhalf = Ped(0, None, location = 0, speed = ped_walking_speed, destination = dest, road=road, epsilon = epsilon, gamma = gamma, lam = lam, alpha = 0.5, a_rate = 1)

utility_costs_cols = ['unmarked_u','zebra_u', 'unmarked_wt', 'zebra_wt', 'unmarked_ve', 'zebra_ve', 'loc']
utility_costs_labels = ['Informal Utility','Dedicated Utility', 'Informal Time Attr.', 'Dedicated Time Attr.', 'Informal Exposure Attr.', 'Dedicated Exposure Attr.', 'loc']
ped_cost_cols = ['unmarked_wt','unmarked_ve', 'zebra_wt', 'zebra_ve']
ped_utility_cols = ['unmarked_u','zebra_u']


alpha = 0.5
suffs = " lam:{}, alpha:{}".format(lam, alpha)
model_vlow = CrossingModel(	ped_origin = ped_start_location, ped_destination = dest, road_length = road_length, road_width = road_width, vehicle_addition_times = v_low, epsilon = epsilon, gamma = gamma, ped_speed = ped_walking_speed, lam = lam, alpha = alpha, a_rate = a_rate)
while model_vlow.running:
	model_vlow.step()

	vs = model_vlow.road._vs
	print("Time:{}".format(model_vlow.schedule.time))
	print("Vehicles pos:{}".format([v.x for v in vs]))

df_attrs = pd.DataFrame(columns = ped_cost_cols, data = model_vlow.ped._ca_costs_history[1:])
df_utilities = pd.DataFrame(columns = ped_utility_cols, data = model_vlow.ped._ca_utility_history[1:])

df_u_r0_v2 = pd.merge(df_utilities, df_attrs, left_index = True, right_index = True)
df_u_r0_v2['Vehicle Flow'] = model_vlow.road._vflows[:-1]

dict_markers['Destination'] = dest
fig_u_r0_v2 = 	plot_utilities_and_costs(df_u_r0_v2, utility_costs_cols[:-1], utility_costs_labels[:-1], 'Attributes and Utilities', "\n $\\alpha$ = {}".format(0.5), vehicle_flow_col = 'Vehicle Flow', xlab = "$P(t)$", dict_markers =dict_markers)

#fig_u_r0_v2.show()
fig_u_r0_v2.savefig(".\\img\\attrs_utilities_a0.5_vlow.png")


################################
#
# Attributes and Utilities with varying traffic flow
#
################################
gap_size = 5
v_vary_low = v_low
for i in range(-gap_size, gap_size):
	ind = int(dest + i)
	v_vary_low[ind]=0

lam = 1
alpha = 0.5
vf = v_vary_low
suffs = " lam:{}, alpha:{}".format(lam, alpha)
model_vlow = CrossingModel(	ped_origin = ped_start_location, ped_destination = dest, road_length = road_length, road_width = road_width, vehicle_addition_times = vf, epsilon = epsilon, gamma = gamma, ped_speed = ped_walking_speed, lam = lam, alpha = alpha, a_rate = a_rate)
while model_vlow.running:
	model_vlow.step()

	# print position of each vehicle
	vs = model_vlow.road._vs
	print("Time:{}".format(model_vlow.schedule.time))
	print("Vehicles pos:{}".format([v.x for v in vs]))

# Get the utilities and attributes from the model_vlow
ped_cost_cols = ['unmarked_wt','unmarked_ve', 'zebra_wt', 'zebra_ve']
ped_utility_cols = ['unmarked_u','zebra_u']

df_attrs = pd.DataFrame(columns = ped_cost_cols, data = model_vlow.ped._ca_costs_history[1:])
df_utilities = pd.DataFrame(columns = ped_utility_cols, data = model_vlow.ped._ca_utility_history[1:])

df_u_a = pd.merge(df_utilities, df_attrs, left_index = True, right_index = True)
df_u_a['Vehicle Flow'] = model_vlow.road._vflows[:-1]

fig_u_a= plot_utilities_and_costs(df_u_a, utility_costs_cols[:-1], utility_costs_labels[:-1], 'Attributes and Utilities with Varied Vehicle Flow', "\n $\\alpha$ = {}".format(0.5), vehicle_flow_col = 'Vehicle Flow', xlab = "$P(t)$", dict_markers =dict_markers)
#fig_u_a.show()
fig_u_a.savefig(".\\img\\attrs_utilities_a0.5_v_vary_low.png")



##############################
#
# Plot activation history and crossing choice
#
##############################
activation_cols = ['unmarked_a','zebra_a']
activation_labels = ['Informal Crossing', 'Dedicated Crossing']
df_activations = pd.DataFrame(columns = activation_cols, data = model_vlow.ped.getActivationHistory()[1:])
df_activations['Vehicle Flow'] = model_vlow.road._vflows[:-1]

# Add in rolling errors
# Doesn't make sense to calculate sd of trend data
'''
df_activations['unmarked_sd'] = df_activations['unmarked_a'].expanding(1).std()
df_activations['zebra_sd'] = df_activations['zebra_a'].expanding(1).std()
error_cols = ['unmarked_sd', 'zebra_sd']
'''

dict_markers['Choice\nMade'] = model_vlow.choice_step
f_act = plot_two_series(df_activations, activation_cols, activation_labels, 'Accumulated Activation with Varied Vehicle Flow', "\n $\\alpha$ = {}".format(0.5), vehicle_flow_col = 'Vehicle Flow', dict_markers = dict_markers, ylab = 'Activation', xlab = 'P(t)')
#f_act.show()
f_act.savefig(".\\img\\activation_a0.5_v_vary_low.png")



#################################
#
# Run model_vhigh with higher vehicle flow, shorter gap, show choice made is different
#
#################################
gap_size = 3
v_vary_high = v_high
for i in range(-gap_size, gap_size):
	ind = int(dest + i)
	v_vary_high[ind]=0

lam = 1
alpha = 0.5
vf = v_vary_high
suffs = " lam:{}, alpha:{}".format(lam, alpha)
model_vhigh = CrossingModel(	ped_origin = ped_start_location, ped_destination = dest, road_length = road_length, road_width = road_width, vehicle_addition_times = vf, epsilon = epsilon, gamma = gamma, ped_speed = ped_walking_speed, lam = lam, alpha = alpha, a_rate = a_rate)
while model_vhigh.running:
	model_vhigh.step()

df_activations = pd.DataFrame(columns = activation_cols, data = model_vhigh.ped.getActivationHistory()[1:])
df_activations['Vehicle Flow'] = model_vhigh.road._vflows[:-1]

dict_markers['Choice\nMade'] = model_vhigh.choice_step
f_act = plot_two_series(df_activations, activation_cols, activation_labels, 'Accumulated Activation with Varied Vehicle Flow', "\n $\\alpha$ = {}".format(0.5), vehicle_flow_col = 'Vehicle Flow', dict_markers = dict_markers, ylab = 'Activation', xlab = 'P(t)')
#f_act.show()
f_act.savefig(".\\img\\activation_a0.5_v_vary_high.png")