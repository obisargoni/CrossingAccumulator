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
        return self._vehicle_flow



class Ped(Agent):

    _gamma = None # controls the rate at which historic activations decay

    _loc = None
    _speed = None # ms-1
    _dest = None
    _crossing_alternatives = None
    _ped_salience_factors = None

    _ca_distance_threshold = 2 # Distance in meters ped must be to ca to choose it or beyond ca to exclude it from further consideration

    _road_length = None
    _road_width = None

    _lambda = None # Used to control degree of randomness of pedestrian decision
    _r = None # Controls sensitivity to traffic exposure

    _alpha = None # Proportion of median activation that ca activation must be to be considered dominant
    _acumulator_rate = None
    _chosen_ca = None
    _ca_activation_history = None

    def __init__(self, unique_id, model, location, speed, destination, crossing_altertives, road_length, road_width, alpha, gamma, lam, r, a_rate):
        super().__init__(unique_id, model)
        self._loc = location
        self._speed = speed
        self._dest = destination

        self._road_length = road_length
        self._road_width = road_width

        self._lambda = lam
        self._r = r
        self._acumulator_rate = a_rate

        self._alpha = alpha
        self._gamma = gamma

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

    def ca_utility(self, ca):
        '''Return the utility of the input crossing alternative for this pedestrian
        '''

        ca_loc = self.caLoc(ca)

        # separate costsing into waiting and walking on road time (no traffic exposure) time
        ww_time = abs(self._loc - ca_loc)*self._speed + ca.getWaitTime() + abs(ca_loc - self._dest)

        # and vehicle exposure when crossing the road
        ve = self.vehicleExposure(ca)

        cost = ww_time * ve

        return np.exp(-1*cost)

    def ca_saliences(self):
        '''Salience of crossing option determined by distance to crossing althernative plus distance from crossing alternative to destination
        '''
        ca_saliences = []
        for (i,ca) in enumerate(self._crossing_alternatives):

            # Get distance from agent to destination
            d = self._dest - self._loc

            # Get distnaces to and from the ca
            d_to = self.caLoc(ca) - self._loc
            d_from = self._dest - self.caLoc(ca)

            # Compare signs to determine whether ca lies in direction of destination or not. Use this to calculate salience distance
            if (np.sign(d) == np.sign(d_to)):
                d_s = d - (abs(d_to) + abs(d_from))
            else:
                d_s = d + (abs(d_to) + abs(d_from))

            # transform salience so that low distances are high salience (because they represent cas closer to ped agent)
            s = (2*self._road_length - d_s) / self._road_length
            ca_saliences.append(s)
        return np.array(ca_saliences)

    def accumulate_ca_activation(self):
        '''Sample crossing alternatives based on their costs. From the selected alternative update ped's perception of its costs.
        '''

        # Sample crossing alternatives according to their salience
        probs = scipy.special.softmax(self._lambda * self.ca_saliences())
        ca = np.random.choice(self._crossing_alternatives, p = probs)
        i = np.where(self._crossing_alternatives == ca)[0][0]

        # Get utility of sampled alternative
        ui = self.ca_utility(self._crossing_alternatives[i])

        ca_activations = self._ca_activation_history[-1]

        # Check if value to update is nan (meaning not updated yet). If it is initialise as zero
        if np.isnan(ca_activations[i]):
            ca_activations[i] = 0.0

        # Decay accumulated activations
        ca_activations = ca_activations * self._gamma

        # Accumulate new activation for sampled ca
        ca_activations[i] += ui

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


        # If nearest dominant ca identified, find distance to this crossing. If within threshold distance choosing this crossing option
        if nearest_ca is not None:
            dist_nearest_ca = abs(self._loc - self.caLoc(nearest_ca))
            if dist_nearest_ca < self._ca_distance_threshold:
                self._chosen_ca = nearest_ca


    def vehicleExposure(self, ca):
        '''Pedestrian vehicle exposure calcualted as the number of vehicles that will pass through crossing during time it takes ped to cross raised
        to the power of the pedestrian traffic sensitivity parameter.
        '''

        t_cross = self._road_width / self._speed

        ve = (t_cross * ca.getVehicleFlow()) ** self._r

        return ve



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

    def __init__(self, ped_origin, ped_destination, road_length, road_width, vehicle_flow, alpha, gamma, ped_speed, lam, r, a_rate, n_peds):
        self.n_peds = n_peds
        self.schedule = RandomActivation(self)
        self.running = True

        # Create two crossing alternatives, one a zebra crossing and one mid block crossing
        zebra_location = road_length * 0.75
        zebra_wait_time = 0
        zebra_type = 'zebra'
        mid_block_wait_time = 3
        mid_block_type = 'mid_block'
        
        zebra = CrossingAlternative(0, self, location = zebra_location, wait_time = zebra_wait_time, ctype = zebra_type, name = 'z1', vehicle_flow = vehicle_flow)
        mid_block = CrossingAlternative(1, self, wait_time = mid_block_wait_time, ctype = mid_block_type, name = 'mid1', vehicle_flow = vehicle_flow)

        # Crossing alternatives with salience factors
        crossing_altertives = [(mid_block, 1), (zebra, 1)]

        # Create population of pedestrian agents
        '''
        lams = np.random.choice(lam_range, n_peds)
        rs = np.random.choice(r_range, n_peds)
        a_rates = np.random.choice(a_rate_range, n_peds)

        population_params = zip(lams, rs, a_rates)
        '''

        i = 0
        ped = Ped(i, self, location = ped_origin, speed = ped_speed, destination = ped_destination, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, alpha = alpha, gamma = gamma, lam = lam, r = r, a_rate = a_rate)
        self.schedule.add(ped)

        self.datacollector = DataCollector(agent_reporters={"CrossingType": "chosenCAType"})

        self.crossing_choice = None

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.get_agent_count() == 0:
            self.running = False
