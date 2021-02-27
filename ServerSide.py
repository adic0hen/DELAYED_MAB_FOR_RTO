from abc import ABC, abstractmethod
import numpy as np
from ClientSide import RTT_RANGE

#Possible RTT_SEQUENCE to play with
RTT_SEQUENCE_1 = [500, 400, 300, 200, 100, 900] #sequence of rtt that the server follows
RTT_SEQUENCE_2 = [500, 400, 300, 200, 100, 900, 456, 127, 984, 652, 222, 123, 399, 700, 421, 2, 635] #sequence of rtt that the server follows
RTT_SEQUENCE_3 = [900, 100, 500, 200] #sequence of rtt that the server follows

class Server(ABC):
    @abstractmethod
    def get_rtt(self):
        pass

    def get_name(self):
        return self.name

    def packet_received(self):
        return True

    def reset(self):
        pass

"""
This server has drops, not in use for our simulations or analysis
"""
class UnreliableServer(Server):
    def __init__(self, receive_prob=0.9, rtt=0):
        self.rtt = rtt
        self.receive_prob = receive_prob
        self.name = "UnstableServer"
    def set_rtt(self, rtt):
        self.rtt = rtt
    def get_rtt(self):
        return self.rtt
    def packet_received(self):
        return np.random.binomial(1, self.receive_prob)

class StableServer(Server):
    def __init__(self, rtt=0):
        self.rtt = rtt
        self.name = "StableServer"

    def get_rtt(self):
        return self.rtt

class RandomServer(Server):
    def __init__(self, rtt_range=RTT_RANGE):
        self.name = "RandomServer"
        self.rtt_range = rtt_range

    def get_rtt(self):
        return np.random.randint(self.rtt_range)

"""
This server might drift to a nearby value and set it as its self.rtt with a given probability
"""
class DriftingServer(Server):
    def __init__(self, rtt=0, variance=4, switch_rtt_prob=0.01):
        self.rtt = rtt
        self.name = "DriftingServer"
        self.variance = variance
        self.switch_rtt_prob = switch_rtt_prob

    def get_rtt(self):
        rtt = int(abs(np.round(np.random.normal(self.rtt, self.variance))))
        should_switch_rtt = np.random.binomial(1, self.switch_rtt_prob)
        if(should_switch_rtt):
            self.rtt = rtt
        return rtt


"""
This server might completely change its self.rtt with a given probability
"""
class VolatileServer(Server):
    def __init__(self, rtt=0, variance=6, switch_rtt_prob=0.0003):
        self.rtt = rtt
        self.name = "VolatileServer"
        self.variance = variance
        self.switch_rtt_prob = switch_rtt_prob

    def get_rtt(self):
        rtt = int(abs(np.round(np.random.normal(self.rtt, self.variance))))
        should_jump_rtt = np.random.binomial(1, self.switch_rtt_prob)
        if(should_jump_rtt):
            self.rtt = np.random.randint(RTT_RANGE)
            print("rtt jump to {}".format(rtt))
        return rtt


"""
This server simulates a possible adversary that can influence the RTT
Pass it an rtt_sequence to follow
"""

class AdversarialServer(Server):
    def __init__(self, rtt_sequence, rounds):
        self.name = "AdversarialServer"
        self.rtt_sequence = rtt_sequence
        self.rounds = rounds
        self.times_called = 0
        self.curr_rtt = 0

    def get_rtt(self):
        if (int((self.times_called / self.rounds)*len(self.rtt_sequence)) - 1 >= self.curr_rtt):
            self.curr_rtt += 1
            print("Server's rtt is {}".format(self.rtt_sequence[self.curr_rtt]))
        self.times_called += 1
        return int(abs(np.round(np.random.normal(self.rtt_sequence[self.curr_rtt], 4))))

    def reset(self):
        self.curr_rtt = 0
        self.times_called = 0
