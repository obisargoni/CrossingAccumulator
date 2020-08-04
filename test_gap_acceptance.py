# Plot activation for crossing alternatives with time varying vehicle flow

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
# Functions (taken from plot_model_components.py)
#
#
#################################
def plot_two_series(df, series_a, series_b, title, title_suffix, vehicle_flow_col = None, dict_markers = None):

	fig = plt.figure(figsize=(12,5))

	fig.gca().set_title(title+title_suffix)

	ax = df[series_a].plot(color='blue', label=series_a)
	ax = df[series_b].plot(color='red', secondary_y=False, label=series_b)

	if vehicle_flow_col is not None:
		ax = df[vehicle_flow_col].plot(color='black', linestyle = 'dotted', label=vehicle_flow_col)

	h1, l1 = ax.get_legend_handles_labels()

	# Add marker for positions of zebra crossing and destination
	for k,v in dict_markers.items():
		plt.axvline(v, linestyle = '--', linewidth = 0.5)
		plt.annotate(k, (v, ax.get_ylim()[1]))

	fig.legend(h1,l1,loc=2)
	return fig

def plot_costs(df, cols, title, title_suffix, vehicle_flow_col = None, dict_markers = None):
	fig = plt.figure(figsize=(12,5))

	fig.gca().set_title(title+title_suffix)

	linestyles = ['-', '--']

	# Plot pairs of utilities and costs (for each crossing alternative)
	for i in range(0, len(cols), 2):
		l = linestyles[i//2]
		ax = df[cols[i]].plot(color='blue', linestyle = l, label=cols[i])
		ax = df[cols[i+1]].plot(color='red', linestyle = l, secondary_y=False, label=cols[i+1])

	if vehicle_flow_col is not None:
		ax = df[vehicle_flow_col].plot(color='black', linestyle = 'dotted', label=vehicle_flow_col)

	h1, l1 = ax.get_legend_handles_labels()
	#h2, l2 = ax2.get_legend_handles_labels()

	# Add marker for positions of zebra crossing and destination
	for k,v in dict_markers.items():
		plt.axvline(v, linestyle = '--', linewidth = 0.5, color = 'grey')
		plt.annotate(k, (v, ax.get_ylim()[1]))

	fig.legend(h1,l1,loc=2)
	return fig

def plot_model_reults(model, suffs, vflow):
	model_data = model.datacollector.get_agent_vars_dataframe()

	df_costs = pd.DataFrame(columns = ped_cost_cols, data = model.ped._ca_costs_history[1:])
	df_utilities = pd.DataFrame(columns = ped_utility_cols, data = model.ped._ca_utility_history[1:])
	df_activations = pd.DataFrame(columns = activation_cols, data = model.ped.getActivationHistory()[1:])

	df_costs['v_flow'] = vflow
	df_utilities['v_flow'] = vflow
	df_activations['v_flow'] = vflow

	# Make plots of costs, utilities, activations each with vehicle flow over the top
	dict_markers['choice']=model.choice_step
	f_costs = plot_costs(df_costs, ped_cost_cols, 'Costs', suffs, vehicle_flow_col = 'v_flow', dict_markers = dict_markers)
	f_u = plot_two_series(df_utilities, 'unmarked_u', 'zebra_u', 'utilities', suffs, vehicle_flow_col = 'v_flow', dict_markers = dict_markers)
	f_act = plot_two_series(df_activations, 'unmarked_a', 'zebra_a', 'utilities', suffs, vehicle_flow_col = 'v_flow', dict_markers = dict_markers)

	f_costs.show()
	f_u.show()
	f_act.show()

###############################
#
#
# Setup consts
#
#
###############################
road_length = 50
road_width = 10
zebra_location = road_length * 0.75
zebra_type = 'zebra'
mid_block_type = 'unmarked'

ped_start_location = 0
dest = road_length/5
ped_walking_speed = 3
gamma = 0.9

ped_cost_cols = ['unmarked_wt', 'zebra_wt', 'unmarked_ve','zebra_ve']
ped_utility_cols = ['unmarked_u','zebra_u']
activation_cols = ['unmarked_a','zebra_a']

dict_markers = {'dest':dest, 'zebra':zebra_location}

# Time varying vehicle flow, create gaps in traffic that occur when ped is walking past detination
v_const_high = [5]*road_length
v_const_low = [1]*road_length

gap_size = 5
v_vary_high = [5]*road_length
for i in range(-gap_size, gap_size):
	ind = int(dest + i)
	v_vary_high[ind] = 0

v_vary_low = [1]*road_length
for i in range(-gap_size, gap_size):
	ind = int(dest + i)
	v_vary_low[ind]=0




# Get the data
lam = 1
aw = 0.5
vf = v_const_high
suffs = " lam:{}, aw:{}".format(lam, aw)
model = CrossingModel(	ped_origin = ped_start_location, ped_destination = dest, road_length = road_length, road_width = road_width, vehicle_flow = vf, 
						alpha = 2, gamma = gamma, ped_speed = ped_walking_speed, lam = lam, aw = aw,	a_rate = 1)
while model.running:
	model.step()

plot_model_reults(model, suffs, vf)


lam = 1
aw = 0.5
vf = v_vary_high
suffs = " lam:{}, aw:{}".format(lam, aw)
model = CrossingModel(	ped_origin = ped_start_location, ped_destination = dest, road_length = road_length, road_width = road_width, vehicle_flow = vf, 
						alpha = 2, gamma = gamma, ped_speed = ped_walking_speed, lam = lam, aw = aw,	a_rate = 1)
while model.running:
	model.step()

plot_model_reults(model, suffs, vf)


lam = 1
aw = 0.5
vf = v_vary_low
suffs = " lam:{}, aw:{}".format(lam, aw)
model = CrossingModel(	ped_origin = ped_start_location, ped_destination = dest, road_length = road_length, road_width = road_width, vehicle_flow = vf, 
						alpha = 2, gamma = gamma, ped_speed = ped_walking_speed, lam = lam, aw = aw,	a_rate = 1)
while model.running:
	model.step()

plot_model_reults(model, suffs, vf)

