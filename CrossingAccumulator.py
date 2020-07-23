# Testing out accumulator model of crossing option choice

import numpy as np
import scipy.special

class CrossingAlternative():

	_loc = None
	_wait_time = None
	_ctype = None
	_name = None

	_vehicle_flow = None

	def __init__(self, location = None, wait_time = None, ctype = None, name = None, vehicle_flow = None):
		self._loc = location
		self._wait_time = wait_time
		self._ctype = ctype
		self._name = name
		self._vehicle_flow = vehicle_flow

	def getLoc(self):
		return self._loc

	def getWaitTime(self):
		return self._wait_time

	def getName(self):
		return self._name

	def getCrossingType(self):
		return self._ctype

	def getVehicleFlow(self):
		return self._vehicle_flow



class Ped():

	_alpha = 0.9
	_beta = 1 - _alpha

	_loc = None
	_speed = None # ms-1
	_dest = None
	_crossing_alternatives = None
	_ped_salience_factors = None

	_road_length = None
	_road_width = None

	_lambda = None # Used to control degree of randomness of pedestrian decision
	_r = None # Controls sensitivity to traffic exposure

	_n_decision = None # Number of times pedestrian accumulates costs before making a decision
	_n_accumulate = None # Number of times ped has accumulated costs
	_chosen_ca = None
	_ca_activation_history = None

	def __init__(self, location, speed, destination, crossing_altertives, road_length, road_width, lam, r, n_decision):
		self._loc = location
		self._speed = speed
		self._dest = destination

		self._road_length = road_length
		self._road_width = road_width

		self._lambda = lam
		self._r = r

		self._n_decision = n_decision
		self._n_accumulate = 0

		self._crossing_alternatives = np.array([])
		self._ped_salience_factors = np.array([])

		for ca, sf in crossing_altertives:
			self.add_crossing_alternative(ca, salience_factor = sf)

		# At time step 0 accumulated utilities are 0
		self._ca_activation_history = np.array([[np.nan] * len(self._crossing_alternatives)])


	def add_crossing_alternative(self, ca, salience_factor = 1):
		self._crossing_alternatives = np.append(self._crossing_alternatives, ca)
		self._ped_salience_factors = np.append(self._ped_salience_factors, salience_factor)

	def caLoc(self, ca):
		ca_loc = ca.getLoc()

		# Mid block crossings not assigned a locations because they take place at ped's current location
		if ca_loc is None:
			ca_loc = self._loc

		return ca_loc

	def ca_costs(self, ca):
		'''Return the costs of the input crossing alternative for this pedestrian
		'''

		ca_loc = self.caLoc(ca)

		# cost simply defined as journey time to destination
		#print(self._loc, ca_loc, self._speed, ca.getWaitTime(), self._dest)
		cost = abs(self._loc - ca_loc)*self._speed + ca.getWaitTime() + abs(ca_loc - self._dest)
		#u = np.exp(-1*self._beta*cost)
		return cost

	def ca_saliences(self):
		'''Salience of crossing option determined by distance to crossing althernative plus distance from crossing alternative to destination
		'''
		ca_saliences = []
		for (i,ca) in enumerate(self._crossing_alternatives):
			s = (2*self._road_length - (abs(self._loc - self.caLoc(ca) + abs(self._dest - self.caLoc(ca))))) / self._road_length
			ca_saliences.append(s)
		return np.array(ca_saliences)

	def update_costs_accumulator(self):
		'''Sample crossing alternatives based on their costs. From the selected alternative update ped's perception of its costs.
		'''

		# Accumulate costs while decision threshold not met
		if self._n_accumulate < self._n_decision:
			# Sample crossing alternatives to select one to update perceived costs of
			probs = scipy.special.softmax(self._lambda * self.ca_saliences())

			# Choosing arg max not correct. Need to sample using these probabilities, or calculate expectation values/ use prob * costs
			ca = np.random.choice(self._crossing_alternatives, p = probs)
			i = np.where(self._crossing_alternatives == ca)[0][0]

			ui = self.ca_costs(self._crossing_alternatives[i])

			accumulated_costs = self._ca_activation_history[-1]

			# Check if value to update is nan (meaning not updated yet). If it is initialise as zero
			if np.isnan(accumulated_costs[i]):
				accumulated_costs[i] = 0.0

			accumulated_costs[i] = self._alpha * accumulated_costs[i] + self._beta * ui # alpha and beta control the balance of influence between new information and old information
			self._ca_activation_history = np.append(self._ca_activation_history, [accumulated_costs], axis = 0)

			self._n_accumulate += 1
		else:
			# Otherwise make choice
			self.choose_ca()

	def walk(self):
		self._loc += self._speed

	def choose_ca(self, history_index = -1):
		'''Chose a crossing alternative by comparing the accumulated costs. Default to the most recent set of accumulated costs
		'''

		accumulated_costs = self._ca_activation_history[history_index]

		# Choose option with lowest accumulated cost, ignoring nan entires as these represent options that haven't been considered
		try:
			cai = np.nanargmin(accumulated_costs)
		except ValueError:
			# If all nan make random choice
			cai = np.random.choice(range(len(accumulated_costs)))

		self._chosen_ca = self._crossing_alternatives[cai]



	def getLoc(self):
		return self._loc

	def getDestination(self):
		return self._dest

	def getSpeed(self):
		return self._speed

	def getActivationHistory(self):
		return self._ca_activation_history

	def getChosenCA(self):
		return self._chosen_ca