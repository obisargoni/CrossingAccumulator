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

road_length = 50
road_width = 10
zebra_location = road_length * 0.75
zebra_type = 'zebra'
mid_block_type = 'unmarked'

ped_start_location = 0
ped_walking_speed = 3
gamma = 0.9
lam = 1
dest = road_length/3
dict_markers = {'Unsignalised\nCrossing':zebra_location}

zebra = CrossingAlternative(0, None, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = 0)
unmarked = CrossingAlternative(1, None, ctype = mid_block_type, name = 'mid1', vehicle_flow = 0)
crossing_altertives = [unmarked, zebra]

ped = Ped(0, None, location = 0, speed = ped_walking_speed, destination = dest, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.9, lam = lam, aw = 0.5, a_rate = 1)

dict_data = ped_salience_distance_and_factors(ped, 50, salience_type = 'ca')

# Join the data together to that distances and saliences can be plotted on the same figure
df_data = pd.merge(dict_data['salience_distances'], dict_data['salience_factors'], on = 'loc', suffixes = ('_dist', '_prob'))

cols = ['unmarked_dist','unmarked_prob', 'zebra_dist','zebra_prob']
labels = ['Unmarked $d_j$','Unmarked probability', 'Unsignalised $d_j$','Unsignalised probability']

fig_probs_sals = plot_dists_and_probs(df_data, cols, labels, "Sampling Probabilities", "", ylab1 = "$d_j$", ylab2 = "p", xlab = "$x_{ped}$", dict_markers = dict_markers)
fig_probs_sals.show()
fig_probs_sals.savefig(".\\img\\distances_probabilities_l1.png")