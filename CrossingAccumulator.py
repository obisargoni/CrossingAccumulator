# Testing out accumulator model of crossing option choice

import numpy as np
import scipy.special

class CrossingAlternative():

	_loc = None
	_wait_time = None
	_ctype = None
	_name = None

	def __init__(self, location = None, wait_time = None, ctype = None, name = None):
		self._loc = location
		self._wait_time = wait_time
		self._ctype = ctype
		self._name = name

	def getLoc(self):
		return self._loc

	def getWaitTime(self):
		return self._wait_time

	def getName(self):
		return self._name

	def getCrossingType(self):
		return self._ctype


class Ped():

	_alpha = 0.9
	_beta = 1 - _alpha

	_loc = None
	_speed = None # ms-1
	_dest = None
	_crossing_alternatives = None
	_ped_salience_factors = None

	_road_length = None

	_lambda = None # Used to control degree of randomness of pedestrian decision

	_accumulated_utility_history = None



	def __init__(self, location, speed, destination, crossing_altertives, road_length, lam):
		self._loc = location
		self._speed = speed
		self._dest = destination

		self._road_length = road_length

		self._lambda = lam

		self._crossing_alternatives = np.array([])
		self._ped_salience_factors = np.array([])

		for ca, sf in crossing_altertives:
			self.add_crossing_alternative(ca, salience_factor = sf)

		# At time step 0 accumulated utilities are 0
		self._accumulated_utility_history = np.array([np.zeros(len(self._crossing_alternatives))])


	def add_crossing_alternative(self, ca, salience_factor = 1):
		self._crossing_alternatives = np.append(self._crossing_alternatives, ca)
		self._ped_salience_factors = np.append(self._ped_salience_factors, salience_factor)

	def caLoc(self, ca):
		ca_loc = ca.getLoc()

		# Mid block crossings not assigned a locations because they take place at ped's current location
		if ca_loc is None:
			ca_loc = self._loc

		return ca_loc

	def ca_utility(self, ca):
		'''Return the utility of the input crossing alternative for this pedestrian
		'''

		ca_loc = self.caLoc(ca)

		# utility simply defined as journey time to destination
		#print(self._loc, ca_loc, self._speed, ca.getWaitTime(), self._dest)
		cost = abs(self._loc - ca_loc)*self._speed + ca.getWaitTime() + abs(ca_loc - self._dest)
		u = -1*cost
		return u

	def ca_saliences(self):
		'''Salience of crossing option determined by its proximity to pedestrian multiplied by pedestrian crossing salience factor.
		'''
		ca_saliences = []
		for (i,ca) in enumerate(self._crossing_alternatives):
			s = ((self._road_length - abs(self._loc - self.caLoc(ca))) / self._road_length) * self._ped_salience_factors[i]
			ca_saliences.append(s)
		return np.array(ca_saliences)

	def update_utility_accumulator(self):
		'''Sample crossing alternatives based on their utility. From the selected alternative update ped's perception of its utility.
		'''
		# Sample crossing alternatives to select one to update perceived utility of
		probs = scipy.special.softmax(self._lambda * self.ca_saliences())
		i = np.argmax(probs)

		ui = self.ca_utility(self._crossing_alternatives[i])

		accumulated_utility = self._accumulated_utility_history[-1]
		accumulated_utility[i] = self._alpha * accumulated_utility[i] + self._beta * ui
		self._accumulated_utility_history = np.append(self._accumulated_utility_history, [accumulated_utility], axis = 0)

	def walk(self, nseconds = None):
		self._loc += self._speed


	def getLoc(self):
		return self._loc

	def getDestination(self):
		return self._dest

	def getSpeed(self):
		return self._speed

	def getAccumulatedUtilityHistory(self):
		return self._accumulated_utility_history

