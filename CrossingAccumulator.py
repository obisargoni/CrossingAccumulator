# Testing out accumulator model of crossing option choice

import numpy as np
import scipy.special
import sys

from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

class CrossingAlternative(Agent):

    _loc = None
    _wait_time = None
    _ctype = None
    _name = None

    _vehicle_flow = None

    def __init__(self, unique_id, model, location = None, wait_time = None, ctype = None, name = None, vehicle_flow = None):
        super().__init__(unique_id, model)
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
        if self._ctype == 'unmarked':
            return self._vehicle_flow
        else:
            return 0

    def getVehicleFlow(self, t):
        if self._ctype == 'unmarked':
            if isinstance(self._vehicle_flow, int):
                return self._vehicle_flow
            else:
                return self._vehicle_flow[t]
        else:
            return 0



class Ped(Agent):

    _loc = None
    _speed = None # ms-1
    _dest = None
    _crossing_alternatives = None
    _ped_salience_factors = None

    _road_length = None
    _road_width = None

    _lambda = None # Used to control effect of salience distance on contribution of option utility to activation
    _aw = None # Controls sensitivity to traffic exposure
    _gamma = None # controls the rate at which historic activations decay

    _C = None # Contrast matrix used to compare crossing option costs to get activations
    _S = None # Matrix used to decay (and possibly compare, though not currently) already accumulated activations

    _alpha = None # Proportion of median activation that ca activation must be to be considered dominant
    _acumulator_rate = None
    _chosen_ca = None
    _ca_activation_history = None

    def __init__(self, unique_id, model, location, speed, destination, crossing_altertives, road_length, road_width, alpha, gamma, lam, aw, a_rate):
        super().__init__(unique_id, model)
        self._loc = location
        self._speed = speed
        self._dest = destination

        self._road_length = road_length
        self._road_width = road_width

        self._lambda = lam
        self._aw = aw # Patameter that controls weight walk time vs crossing exposure in ca utility calculation
        self._acumulator_rate = a_rate

        self._alpha = alpha
        self._gamma = gamma

        self._crossing_alternatives = np.array([])
        self._ped_salience_factors = np.array([])

        for ca in crossing_altertives:
            self.add_crossing_alternative(ca)

        # At time step 0 accumulated utilities are 0
        self._ca_activation_history = np.array([[0] * len(self._crossing_alternatives)])
        self._ca_costs_history = np.array([[0] * len(self._crossing_alternatives) * 2])
        self._ca_utility_history = np.array([[0] * len(self._crossing_alternatives)])

        # Set up matrices used for activation accumulation
        self.setup_C()
        self.setup_S()


    def setup_C(self):
        '''Set up matrix used to contrast utilities/costs of different crossing alternatives
        '''
        num_cas = len(self._crossing_alternatives)
        self._C = np.zeros((num_cas, num_cas))
        for i in range(num_cas):
            for j in range(num_cas):
                if i==j:
                    self._C[i,j] = 1
                else:
                    self._C[i,j] = -1/(num_cas-1)

    def setup_S(self):
        '''Setup matrix used to decay accumulated costs
        '''
        num_cas = len(self._crossing_alternatives)
        self._S = (self._gamma) * np.identity(num_cas)


    def add_crossing_alternative(self, ca, salience_factor = 1):
        self._crossing_alternatives = np.append(self._crossing_alternatives, ca)
        self._ped_salience_factors = np.append(self._ped_salience_factors, salience_factor)

    def caLoc(self, ca):
        ca_loc = ca.getLoc()

        # Mid block crossings not assigned a locations because they take place at ped's current location
        if ca_loc is None:
            ca_loc = self._loc

        return ca_loc


    def ca_vehicle_exposure(self, ca):
        '''Pedestrian vehicle exposure calcualted as the number of vehicles that will pass through crossing during time it takes ped to cross raised
        to the power of the pedestrian traffic sensitivity parameter.
        '''

        t_cross = self._road_width / self._speed

        # use peds location as a proxy for time
        t = int(self._loc) % self._road_length

        ve = (t_cross * ca.getVehicleFlow(t))

        return ve

    def ca_vehicle_exposures(self):
        '''Get varray of vehicle exposures for all crossing alternatives
        '''
        v_ve = np.vectorize(self.ca_vehicle_exposure)
        return v_ve(self._crossing_alternatives)


    def ca_walk_time(self, ca):
        ca_loc = self.caLoc(ca)

        # separate costsing into waiting and walking on road time (no traffic exposure) time
        ww_time = abs(self._loc - ca_loc)/self._speed + abs(ca_loc - self._dest)/self._speed

        return ww_time

    def ca_walk_times(self):
        '''Get array of walking times for each crossing alternative
        '''
        v_ww_times = np.vectorize(self.ca_walk_time)
        return v_ww_times(self._crossing_alternatives)

    def ca_costs(self):
        return np.concatenate((self.ca_walk_times(), self.ca_vehicle_exposures()))

    def ca_utility(self, ca):
        '''Use vehicle exposure as the measure of crossing utility
        '''

        # aw gives weight of walk time attribute, 1-aw gives weight of vehicle exposure attribute
        weights = np.array([self._aw, 1-self._aw])
        ca_utility_parts = -1*np.array([self.ca_walk_time(ca), self.ca_vehicle_exposure(ca)]) # Multiply by -1 to go from costs to utility

        return np.dot(weights, ca_utility_parts)

    def ca_utilities(self):
        '''Get array of utilities for all crossing alternatives
        '''
        v_u = np.vectorize(self.ca_utility)
        return v_u(self._crossing_alternatives)

    def ca_salience_distances(self):
        '''Salience of crossing option determined by distance to crossing althernative plus distance from crossing alternative to destination
        '''
        ca_salience_distances = []
        for i,ca in enumerate(self._crossing_alternatives):
            # Get distance from agent to destination
            d = abs(self._dest - self._loc)

            # Get distnaces to and from the ca
            d_to = self.caLoc(ca) - self._loc
            d_from = self._dest - self.caLoc(ca)

            # Salience distance is difference between direct distance and distance via crossing, scaled by road length
            d_s = (abs(d_to) + abs(d_from) - d) / self._road_length
            ca_salience_distances.append(d_s)
        return np.array(ca_salience_distances)

    def ca_salience_distances_softmax(self):
        '''Salience of crossing option determined by distance to crossing althernative plus distance from crossing alternative to destination
        '''
        ca_salience_distances = []
        for i,ca in enumerate(self._crossing_alternatives):

            # Get distance from agent to destination
            d = abs(self._dest - self._loc)

            # Get distnaces to and from the ca
            d_to = self.caLoc(ca) - self._loc
            d_from = self._dest - self.caLoc(ca)

            # Salience distance is difference between direct distance and distance via crossing, scaled by road length
            d_s = (2*self._road_length - (abs(d_to) + abs(d_from) - d)) / self._road_length
            ca_salience_distances.append(d_s)
        return np.array(ca_salience_distances)

    def ca_salience_factors(self):
        '''Get the factors to scale utilities by
        '''
        return np.exp(-self._lambda * self.ca_salience_distances())

    def ca_salience_factors_softmax(self):
        return scipy.special.softmax(self._lambda * self.ca_salience_distances_softmax())

    def accumulate_ca_activation(self):
        '''Sample crossing alternatives based on their costs. From the selected alternative update ped's perception of its costs.
        '''

        # Sample crossing alternatives according to their salience
        salience_factors = self.ca_salience_factors_softmax()

        # Get utilities of crossing alternatives
        u = self.ca_utilities()

        ca_activations = self._ca_activation_history[-1]

        # Decay accumulated activations
        ca_activations = np.matmul(self._S, ca_activations) + np.matmul(self._C, u)

        self._ca_activation_history = np.append(self._ca_activation_history, [ca_activations], axis = 0)

    def accumulate_ca_activation_sampling(self):
        '''Sample crossing alternatives based on their costs. From the selected alternative update ped's perception of its costs.
        '''

        # Sample crossing alternatives according to their salience
        salience_factors = self.ca_salience_factors_softmax()

        ca = np.random.choice(self._crossing_alternatives, p = salience_factors)
        i = np.where(self._crossing_alternatives == ca)[0][0]

        # Get utility of sampled alternative - use a measure of vehicle exposure as utility for activation
        ui = self.ca_utility(self._crossing_alternatives[i])

        # Calculate activation by doing max cost + ui (since ui = -costs this gives difference between maximum costs and cost of this ca)
        weights = np.array([self._aw, 1-self._aw])
        max_costs = np.dot(weights, np.array([(2*self._road_length / self._speed), 20])) # 20 represents maximum vehicle exposure for this road
        ca_act = max_costs + ui

        ca_activations = self._ca_activation_history[-1]

        # Check if value to update is nan (meaning not updated yet). If it is initialise as zero
        if np.isnan(ca_activations[i]):
            ca_activations[i] = 0.0

        # Decay accumulated activations
        ca_activations = ca_activations * self._gamma

        # Accumulate new activation for sampled ca
        ca_activations[i] += ca_act

        self._ca_activation_history = np.append(self._ca_activation_history, [ca_activations], axis = 0)


    def walk(self):
        self._loc += self._speed

        # Check whether a crossing alternative has emerged as the dominant alternative and choose it if it's nearby
        self.choose_ca()

        return

    def step(self):

        # Check if ped has reached end of the road or if it has chosen a crossing
        if (self.getLoc() < self._road_length) and (self._chosen_ca is None):

            # update ped's perceptions of crossing alternative utilities
            for i in range(self._acumulator_rate):
                self.accumulate_ca_activation()

                self._ca_costs_history = np.append(self._ca_costs_history, [self.ca_costs()], axis=0)
                self._ca_utility_history = np.append(self._ca_utility_history, [self.ca_utilities()], axis=0)

            # move the ped along
            self.walk()
        else:
            # When agent is done remove from schedule
            self.model.crossing_choice = self.chosenCAType
            self.model.schedule.remove(self)


    def choose_ca(self, history_index = -1):
        '''Chose a crossing alternative by comparing the accumulated costs. Default to the most recent set of accumulated costs
        '''

        # Get the indices of crossing alternatives whose activation is above the threshold value
        ca_activations = self._ca_activation_history[history_index]
        dom_threshold = np.nanmean(ca_activations) * self._alpha
        dominant_indices = np.where( ca_activations > dom_threshold)

        # Select the nearest of these
        min_dist = sys.float_info.max
        nearest_ca = None
        for i in dominant_indices[0]:
            dom_ca = self._crossing_alternatives[i]
        
            if (self.caLoc(dom_ca) < min_dist):
                min_dist = self.caLoc(dom_ca)
                nearest_ca = dom_ca


        # If nearest dominant ca identified set as chosen option
        if nearest_ca is not None:
            self._chosen_ca = nearest_ca
            self.model.choice_step = self.model.nsteps


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

    @property   
    def chosenCAType(self):
        if self.getChosenCA() is None:
            return None
        else:
            return self.getChosenCA().getCrossingType()


class CrossingModel(Model):
    def __init__(self, ped_origin, ped_destination, road_length, road_width, vehicle_flow, alpha, gamma, ped_speed, lam, aw, a_rate):
        self.schedule = RandomActivation(self)
        self.running = True
        self.nsteps = 0

        # Create two crossing alternatives, one a zebra crossing and one mid block crossing
        zebra_location = road_length * 0.75
        zebra_type = 'zebra'
        mid_block_type = 'unmarked'
        
        zebra = CrossingAlternative(0, self, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = vehicle_flow)
        unmarked = CrossingAlternative(1, self, ctype = mid_block_type, name = 'mid1', vehicle_flow = vehicle_flow)

        # Crossing alternatives with salience factors
        crossing_altertives = np.array([unmarked,zebra])

        i = 0
        self.ped = Ped(i, self, location = ped_origin, speed = ped_speed, destination = ped_destination, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = alpha, gamma = gamma, lam = lam, aw = aw, a_rate = a_rate)
        self.schedule.add(self.ped)

        self.datacollector = DataCollector(agent_reporters={"CrossingType": "chosenCAType"})

        self.crossing_choice = None
        self.choice_step = None

    def step(self):
        self.nsteps += 1
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.get_agent_count() == 0:
            self.running = False
