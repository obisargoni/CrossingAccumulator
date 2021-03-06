# Testing out accumulator model of crossing option choice

import numpy as np
np.random.seed(3)

import scipy.special
from scipy.stats import bernoulli
import sys

from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

class CrossingAlternative(Agent):

    _loc = None
    _ctype = None
    _name = None

    _vehicle_flow = None
    _av_flow = None

    def __init__(self, unique_id, model, location = None, wait_time = None, ctype = None, name = None, vehicle_flow = None):
        super().__init__(unique_id, model)
        self._loc = location
        self._wait_time = wait_time
        self._ctype = ctype
        self._name = name
        self._vehicle_flow = vehicle_flow

        self._av_flow = 2

    def getLoc(self):
        return self._loc

    def getName(self):
        return self._name

    def getCrossingType(self):
        return self._ctype

    def getVehicleFlow(self, t = None):
        if self._ctype == 'unmarked':
            if isinstance(self._vehicle_flow, (int, float)):
                return self._vehicle_flow
            else:
                return self._vehicle_flow[t]
        else:
            return 0

    def getAvgVehicleFlow(self):
        return self._av_flow



class Ped(Agent):

    _model_type = None

    _loc = None
    _speed = None # ms-1
    _dest = None
    _crossing_alternatives = None
    _ped_salience_factors = None

    _road_length = None
    _road_width = None

    _lambda = None # Used to control effect of salience distance on contribution of option utility to activation
    _alpha = None # Controls sensitivity to traffic exposure
    _gamma = None # controls the rate at which historic activations decay

    _C = None # Contrast matrix used to compare crossing option costs to get activations
    _S = None # Matrix used to decay (and possibly compare, though not currently) already accumulated activations

    _epsilon = None # Proportion of median activation that ca activation must be to be considered dominant
    _acumulator_rate = None
    _chosen_ca = None
    _ca_activation_history = None

    def __init__(self, unique_id, model, location, speed, destination, crossing_altertives, road_length, road_width, epsilon, gamma, lam, alpha, a_rate, model_type = 'sampling'):
        super().__init__(unique_id, model)

        self._model_type = model_type

        self._loc = location
        self._speed = 1
        self._dest = destination

        self._road_length = road_length
        self._road_width = road_width

        self._lambda = lam
        self._alpha = alpha # Patameter that controls weight walk time vs crossing exposure in ca utility calculation
        self._acumulator_rate = a_rate

        self._epsilon = epsilon
        self._gamma = gamma

        self._crossing_alternatives = np.array([])
        self._ped_salience_factors = np.array([])

        for ca in crossing_altertives:
            self.add_crossing_alternative(ca)

        # At time step 0 accumulated utilities are 0
        self._ca_activation_history = np.array([[0] * len(self._crossing_alternatives)])
        self._ca_costs_history = np.array([[0] * len(self._crossing_alternatives) * 2])
        self._ca_utility_history = np.array([[0] * len(self._crossing_alternatives)])
        self._loc_history = np.array([0])

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

    def ca_walk_time(self, ca):
        ca_loc = self.caLoc(ca)

        # separate costsing into waiting and walking on road time (no traffic exposure) time
        ww_time = abs(self._loc - ca_loc)/self._speed + abs(ca_loc - self._dest)/self._speed

        return ww_time

    def ca_detour_time(self, ca):
        '''The time difference between using the crossing alternative and walking directly to the destination
        via an unmarked crossing
        '''
        ca_loc = self.caLoc(ca)

        d_ca = abs(self._loc - ca_loc) + abs(ca_loc - self._dest)

        detour_dist = d_ca - abs(self._dest - self._loc)

        return detour_dist/self._speed


    def ca_vehicle_exposure_fd(self, ca):
        '''Get vehicle exposure fractional difference from characteristic vehicle, 
        chosen to be the time taken to walk the length of the road.
        '''
        t_cross = self._road_width / self._speed

        # use peds location as a proxy for time
        t = int(self._loc) % self._road_length

        ve = (t_cross * ca.getVehicleFlow(t))
        av_ve = (t_cross * ca.getAvgVehicleFlow())
        
        return (1 - ve/av_ve)


    def ca_walk_time_fd(self, ca):
        '''Get walk time fractional difference from characteristic walk time, 
        chosen to be the time taken to walk the length of the road.
        '''
        wt = self.ca_detour_time(ca)
        char_wt = self._road_length / self._speed

        return (1-wt/char_wt)


    def stochastic_weights(self):
        '''For use in decision field theory model. Probabilistically set weights for attributes to be either 
        0 or 1 so that at each time step crossing alternatives are compared on a single attribute only.
        '''

        # Draw 0 or 1 at random from bernouli distribution, with prob pof 1 given by weight parameter alpha
        weight_time = bernoulli.rvs(self._alpha)
        weight_ve = int(not weight_time)
        return np.array([weight_time, weight_ve])


    def cas_attributes_dft(self):
        cas_attr = []
        for ca in self._crossing_alternatives:
            ca_attr = -1*np.array([self.ca_walk_time(ca), self.ca_vehicle_exposure(ca)]) # Multiply by -1 since these are costs
            cas_attr.append(ca_attr)

        return np.array(cas_attr)

    def cas_attributes_sampling(self):
        cas_attr = []
        for ca in self._crossing_alternatives:
            ca_attr = np.array([self.ca_walk_time_fd(ca), self.ca_vehicle_exposure_fd(ca)])
            cas_attr.append(ca_attr)

        return np.array(cas_attr)


    def ca_costs(self, model = 'dft'):
        '''Reshapes the attributes matrix into a single line of costs for each attribute and crossing alternative.
        '''
        if model == 'dft':
            ca_costs = np.reshape(self.cas_attributes_dft(), (len(self._crossing_alternatives) * 2))
        else:
            ca_costs = np.reshape(self.cas_attributes_sampling(), (len(self._crossing_alternatives) * 2))
        return ca_costs


    def ca_utilities(self, model = 'dft'):
        '''Get array of utilities for all crossing alternatives
        '''

        if model == 'dft':
            weights = self.stochastic_weights()
            cas_attrs = self.cas_attributes_dft()
        else:
            weights = np.array([self._alpha, 1-self._alpha])
            cas_attrs = self.cas_attributes_sampling()

        utilities = np.matmul(cas_attrs, weights)
        return utilities

    def ca_salience_distances_to_ca(self):
        '''Salience of crossing option determined by difference between road length and distance to crossing alternative.
        '''
        ca_salience_distances = []
        for i,ca in enumerate(self._crossing_alternatives):

            # Get distnaces to and from the ca
            d_to = self.caLoc(ca) - self._loc

            # Salience distance is difference between direct distance and distance via crossing, scaled by road length
            d_s = (self._road_length - abs(d_to)) / self._road_length
            ca_salience_distances.append(d_s)
        return np.array(ca_salience_distances)

    def ca_salience_distances_to_dest(self):
        '''Salience of crossing option determined by difference between twice the road length and the distance to the destination
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


    def ca_salience_factors_softmax(self, salience_type = 'ca'):
        if salience_type == 'ca':
            return scipy.special.softmax(self._lambda * self.ca_salience_distances_to_ca())
        else:
            return scipy.special.softmax(self._lambda * self.ca_salience_distances_to_dest())

    def accumulate_ca_activation(self, model_type = 'sampling'):
        '''Sample crossing alternatives based on their costs. From the selected alternative update ped's perception of its costs.
        '''
        ca_activations = self._ca_activation_history[-1]

        # Get utilities of crossing alternatives
        u = self.ca_utilities(model = model_type)

        if model_type == 'dft':
            # Decay accumulated activations
            ca_activations = np.matmul(self._S, ca_activations) + np.sign(np.matmul(self._C, u))
        else:
            # Sample crossing alternatives according to their salience
            salience_factors = self.ca_salience_factors_softmax(salience_type = 'ca')
            i = np.random.choice(len(salience_factors), p= salience_factors)

            # Get matrix used to make activation only accumulate for the sampled crossing alternative
            _C = np.zeros((len(salience_factors), len(salience_factors)))
            _C[i,i] = 1

            ca_activations = np.matmul(self._S, ca_activations) + np.matmul(_C, u)

        self._ca_activation_history = np.append(self._ca_activation_history, [ca_activations], axis = 0)


    def walk(self):
        self._loc += self._speed

        # Check whether a crossing alternative has emerged as the dominant alternative and choose it if it's nearby
        if self._chosen_ca is None: # Put in place since no longer removing peds after choice is made
            self.choose_ca()

        return

    def step(self):

        # Check if ped has reached end of the road or if it has chosen a crossing
        if (self.getLoc() < self._road_length): #and (self._chosen_ca is None):

            # update ped's perceptions of crossing alternative utilities
            self.accumulate_ca_activation(model_type = self._model_type)
            self._ca_costs_history = np.append(self._ca_costs_history, [self.ca_costs(model = self._model_type)], axis=0)
            self._ca_utility_history = np.append(self._ca_utility_history, [self.ca_utilities(model = self._model_type)], axis=0)
            self._loc_history = np.append(self._loc_history, self._loc)

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
        sorted_activations = np.sort(ca_activations)


        # Check whether the largest activation is greater than second largest by threshold amount
        # If it is then select largest
        if (sorted_activations[-1] - sorted_activations[-2]) > self._epsilon:
            ca_i = np.argmax(ca_activations)
            self._chosen_ca = self._crossing_alternatives[ca_i]
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
    def __init__(self, ped_origin, ped_destination, road_length, road_width, vehicle_flow, epsilon, gamma, ped_speed, lam, alpha, a_rate):
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
        model_type = 'sampling'
        self.ped = Ped(i, self, location = ped_origin, speed = ped_speed, destination = ped_destination, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, epsilon = epsilon, gamma = gamma, lam = lam, alpha = alpha, a_rate = a_rate, model_type = model_type)
        self.schedule.add(self.ped)

        self.datacollector = DataCollector(agent_reporters={"CrossingType": "chosenCAType"})

        self.crossing_choice = None
        self.choice_step = None

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.get_agent_count() == 0:
            self.running = False
        self.nsteps += 1

