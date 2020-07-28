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
lam = 1

zebra = CrossingAlternative(0, None, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = 0)
unmarked = CrossingAlternative(1, None, ctype = mid_block_type, name = 'mid1', vehicle_flow = 0)
crossing_altertives = [(unmarked, 1), (zebra, 1)]

ped = Ped(0, None, location = 0, speed = ped_walking_speed, destination = road_length, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = 1.2, gamma = 0.9, lam = 1, r = 0, a_rate = 1)

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
fig_probs = plot_two_series(df_probs, 'unmarked', 'zebra', 'unmarked crossing', 'zebra crossing', 'Crossing Probability')

fig_salience.show()
fig_probs.show()