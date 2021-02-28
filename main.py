from heapq import heappop, heappush
import numpy as np
from ClientSide import RandomAgent, Transmitter, WindowedThompsonAgent, RTT_RANGE
from matplotlib import pyplot as plt
from matplotlib import collections as matcoll
from ServerSide import StableServer, AdversarialServer, DriftingServer, RandomServer, \
    RTT_SEQUENCE_2, RTT_SEQUENCE_3, VolatileServer

PERIOD_UPDATES = 100 #number of iterations after which data regarding the loss and plotting data is collected


class Results():
    def __init__(self):
        self.accumulated_average_loss = []
        self.chosen_arms_list = []
        self.actual_rtt = []
        self.total_regret = 0
        self.rto_above = 0
        self.rto_below = 0

    def update(self, periodic_regret, rto_above_cnt, rto_below_cnt, rto, rtt):
        self.accumulated_average_loss.append(periodic_regret / PERIOD_UPDATES)
        self.chosen_arms_list.append(rto)
        self.actual_rtt.append(rtt)
        self.total_regret += periodic_regret
        self.rto_above += rto_above_cnt
        self.rto_below += rto_below_cnt


class Feedback():
    def __init__(self, rto, rtt, arrival, packet_id):
        self.rto = rto
        self.rtt = rtt
        self.arrival = arrival
        self.packet_id = packet_id


    def __lt__(self, other):
        return self.arrival < other.arrival

def getArrivedFeedbacks(delayed_feedbacks, t):
    arrived_feedbacks = []
    while(len(delayed_feedbacks) > 0 and t >= delayed_feedbacks[0].arrival):
        arrived_feedbacks.append(heappop(delayed_feedbacks))
    return arrived_feedbacks


def runExperiment(rounds, transmitter=Transmitter(RandomAgent()), server=StableServer(), verbose=False):
    print("starting experiment, parameters:\n rounds = {}, transmitterAgent = {}, server = {}"
          .format(rounds, transmitter.get_agent_name(), server.get_name()))
    results = Results()
    periodic_regret = 0
    rto_above_cnt = 0
    rto_below_cnt = 0
    delayed_feedbacks = []
    for t in range(1, rounds + 1):
        rto = transmitter.send_packet(packet_id=t)
        rtt = server.get_rtt()
        delay = rtt
        #delay = 1 #Uncomment to see results when there is no delay for feedback
        heappush(delayed_feedbacks, Feedback(rto, rtt, arrival=t + delay, packet_id=t))
        if (t % PERIOD_UPDATES == 0):
            if(verbose):
                print("PERIOD_UPDATES reached, iteration {}".format(t))
            results.update(periodic_regret, rto_above_cnt, rto_below_cnt, rto, rtt)
            periodic_regret = 0
            rto_above_cnt = 0
            rto_below_cnt = 0

        transmitter.update(getArrivedFeedbacks(delayed_feedbacks, t))

        #Loss types
        periodic_regret += calculateLoss(rto, rtt)
        if (rto > rtt):
            rto_above_cnt += 1
        else:
            rto_below_cnt += 1

    return results


def calculateLoss(rto, rtt):
    return abs(rto - rtt)

def single_experiment():
    rounds = 1 * pow(10, 4)
    agent = WindowedThompsonAgent(window_size=100)
    #agent = DecayingThompsonAgent()
    server = DriftingServer(rtt=200, variance=9, switch_rtt_prob=0.01)
    #server =AlternatingServer(RTT_SEQUENCE_2, rounds)


    transmitter = Transmitter(agent)
    results = runExperiment(rounds, transmitter=transmitter, server=server)

    print("total_regret is {}".format(results.total_regret))
    print("rto above rtt count {}".format(results.rto_above))
    print("rto below rtt count (expired rto) {}".format(results.rto_below))
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Results for RTO Learning, Transmitter Agent - {}, Server - {}".format(transmitter.get_agent_name(), server.get_name()))
    ax1.set_title("Last 1000 requests Average loss")
    ax2.set_title("Chosen arms")
    ax1.plot(results.accumulated_average_loss)
    ax2.plot(results.actual_rtt, label="actual rtt")
    ax2.plot(results.chosen_arms_list, label="chosen rtt")
    plt.legend()
    plt.show()

def experiment_different_servers():
    rounds = 1 * pow(10, 4)
    thompson_agent = WindowedThompsonAgent(window_size=100)

    servers = [RandomServer(), DriftingServer(rtt=200, variance=6), VolatileServer(), AdversarialServer(RTT_SEQUENCE_3, rounds)]
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Results for RTO Learning VS different server types")
    for i, server in enumerate(servers):
        transmitter = Transmitter(thompson_agent)
        results = runExperiment(rounds, transmitter=transmitter, server=server)
        print("Regret was {:,}".format(results.total_regret))
        curr_axis = axs[i//2][i%2]
        curr_axis.set_title("{}, Regret - {:,}".format(server.get_name(), results.total_regret))
        xs = np.linspace(0, rounds, len(results.actual_rtt))
        curr_axis.plot(xs, results.actual_rtt, label="actual rtt")
        curr_axis.plot(xs, results.chosen_arms_list, label="chosen rto")
    plt.legend()
    plt.show()

def experiment_compare_window_size():
    rounds = 1 * pow(10, 4)
    window_sizes = [1, 50, 100, 300, 500, 1000]
    total_regret_dict = {}
    servers = [AdversarialServer(RTT_SEQUENCE_2, rounds), DriftingServer(rtt=200, variance=6)]
    #servers = [DriftingServer(rtt=200, variance=6)]
    fig, axs = plt.subplots(2)
    fig.suptitle("Regret by window size")
    upper_ylim = 0
    for i, server in enumerate(servers):
        for window_size in window_sizes:
            server.reset()
            results = runExperiment(rounds, transmitter=Transmitter(WindowedThompsonAgent(window_size=window_size)), server=server)
            total_regret_dict[window_size] = results.total_regret

        curr_axis = axs[i]
        linecoll = matcoll.LineCollection([[(key, 0), (key, total_regret_dict[key])] for key in total_regret_dict.keys()])
        curr_axis.set_title("{}".format(server.get_name()))
        curr_axis.scatter(total_regret_dict.keys(), total_regret_dict.values())
        curr_axis.add_collection(linecoll)
        curr_axis.set_xticks(list(total_regret_dict.keys()))
        if (upper_ylim < max(total_regret_dict.values())):
            upper_ylim = max(total_regret_dict.values())
        curr_axis.set_ylabel("Regret")
        curr_axis.set_xlabel("Window Size")
    for axis in axs:
        axis.set_ylim([0, upper_ylim])
    plt.show()

"""
Uncomment any of the following line to run an experiment
"""
if __name__ == '__main__':
    single_experiment()
    #experiment_different_servers()
    #experiment_compare_window_size()