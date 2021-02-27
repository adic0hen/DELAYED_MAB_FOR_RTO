from abc import ABC, abstractmethod
import numpy as np
RTT_RANGE = 1000
MAX_REWARDS = 1000
MIN_REWARDS = 1
PROMOTE_INDEX = 0
DEMOTE_INDEX = 1
ARMS_INITIAL_WEIGHT = 2.0
BASE_DEMOTE_WEIGHT = 1
BASE_PROMOTE_WEIGHT = 10
PROMOTION_RANGE = 5

#Basically just a shell for the agent
class Transmitter():
    def __init__(self, agent):
        self.agent = agent
    def get_agent_name(self):
        return self.agent.get_name()
    def update(self, feedbacks):
        return self.agent.update(feedbacks)
    def send_packet(self, packet_id):
        return self.agent.send_packet(packet_id)

#Abstract agent
class Agent(ABC):
    @abstractmethod
    def choose_rto(self):
        pass

    @abstractmethod
    def update(self, feedbacks):
        pass

    def send_packet(self, packet_id):
        return self.choose_rto()

    def get_name(self):
        return self.name

    def printArms(self):
        pass




"""
The main agent used to determine the RTO set to a sent packet
Uses a sliding window and determines its decision according to the information in the window
Promotes successful arms and demotes unsuccessful arms according to their distance from the right answer
Leveraging the fact that the arms are not IID, the success of one arm indicates the failure of others
Uses early demotion for packets which had their RTO expired to adapt quickly even when the feedback is not yet available
"""
class WindowedThompsonAgent(Agent):
    def __init__(self, window_size=150):
        super()
        self.arms = np.full((RTT_RANGE, 2), ARMS_INITIAL_WEIGHT)
        self.history = np.full((window_size, RTT_RANGE, 2), 0.0).tolist()
        self.name = "WindowedThompsonAgent"
        self.pending_packets = {}

    def send_packet(self, packet_id):
        rto = self.choose_rto()
        self.pending_packets[packet_id] = [rto, rto] #one is for countdown to detect expired RTO, the other is for reference on the original RTO
        return rto

    #Sampling values from all arms, then choosing the arm with the highest value sampled in this round
    def choose_rto(self):
        maxTheta = 0
        chosenArm = np.random.randint(len(self.arms))
        for i, betaParams in enumerate(self.arms):
            theta = np.random.beta(betaParams[0], betaParams[1])
            if (theta > maxTheta):
                maxTheta = theta
                chosenArm = i
        return chosenArm

    def update(self, feedbacks):
        self.receive_packets(feedbacks)
        self.update_expired_rto()
        for feedback in feedbacks:
            frame = self.getUpdateParametersOnFeedback(feedback.rto, feedback.rtt)
            self.updateWindow(frame)

    def receive_packets(self, feedbacks):
        received_packet_ids = [feedback.packet_id for feedback in feedbacks]
        for packet_id in received_packet_ids:
            if(packet_id in self.pending_packets.keys()):
                del self.pending_packets[packet_id]

    def update_expired_rto(self):
        to_del = []
        for packet_id in self.pending_packets.keys():
            if(self.isRTOExpired(packet_id)):
                frame = self.getUpdateParametersRTOExpired(self.pending_packets[packet_id][1]) #demoting rto and below
                self.updateWindow(frame)
                to_del.append(packet_id)
        for packet_id in to_del:
            del self.pending_packets[packet_id]

    def isRTOExpired(self, packet_id):
        self.pending_packets[packet_id][0] -= 1
        return self.pending_packets[packet_id][0] < 0

    """
    Adding values to the arms according to the new frame we built, the new frame is added to the window
    then reducing values according to the oldest frame in the window, which is also removed from the window
    """
    def updateWindow(self, frame):
        self.arms += frame
        self.history.append(frame)

        old_frame = self.history.pop(0)
        self.arms -= old_frame
        #Keeping the arms values at a given range to avoid over promoting/demoting
        self.arms = np.clip(self.arms, MIN_REWARDS, MAX_REWARDS)

    def printArms(self):
        print(self.arms)

    def getUpdateParametersOnFeedback(self, rto, rtt):
        frame = np.zeros((RTT_RANGE, 2))
        promote_environment = [max(rtt - PROMOTION_RANGE, 0), min(rtt + PROMOTION_RANGE, RTT_RANGE)]
        # arms far away from the promotion environment get LOWER promotion weight
        promoting_weights = [BASE_PROMOTE_WEIGHT/(1 + abs(i - rtt)) for i in range(promote_environment[0], promote_environment[1])]
        frame[promote_environment[0]: promote_environment[1], PROMOTE_INDEX] += promoting_weights
        #frame[promote_environment[0]: promote_environment[1], DEMOTE_INDEX] -= promoting_weights

        if(rtt <= rto):
            demoting_range = [min(rtt + PROMOTION_RANGE, RTT_RANGE), RTT_RANGE]
        else:
            demoting_range = [rto, max(rto, rtt - PROMOTION_RANGE)]
        # arms far away from the promition environment get HIGHER demotion weight
        demoting_weights = [BASE_DEMOTE_WEIGHT * (abs(rtt - i)/RTT_RANGE) for i in range(demoting_range[0], demoting_range[1])]
        frame[demoting_range[0]:demoting_range[1], DEMOTE_INDEX] += demoting_weights
        #frame[demoting_range[0]:demoting_range[1], PROMOTE_INDEX] -= demoting_weights
        return frame.tolist()

    def getUpdateParametersRTOExpired(self, rto):
        #should retransmit since rto expired, demoting arms at or below the chosen rto
        frame = np.zeros((RTT_RANGE, 2))
        demoting_weights = [BASE_DEMOTE_WEIGHT * sigmoid(abs(rto - i)) for i in range(rto + 1)] #arms far below the rto get higher demotion weight
        frame[:rto + 1, DEMOTE_INDEX] = demoting_weights
        return frame.tolist()

class RandomAgent(Agent):
    def __init__(self):
        super()
        self.name = "RandomAgent"
    def choose_rto(self):
        return np.random.randint(RTT_RANGE)

    def update(self, feedbacks):
        pass


"""
Handling delays with a decay parameter that multiplies and flattens the arms parameters over time
Not in use for our simulations
"""
class DecayingThompsonAgent(Agent):
    def __init__(self, decay_factor=0.55, variance=0.3, num_scatter=10):
        super()
        self.arms = np.full((RTT_RANGE, 2), 2)
        self.name = "ThompsonAgent"
        self.decay_factor = decay_factor
        self.variance = variance
        self.num_scatter = num_scatter
        np.random.seed(np.random.randint(1000))

    def choose_rto(self):
        self.arms = self.arms*self.decay_factor
        maxTheta = 0
        chosenArm = np.random.randint(len(self.arms))
        for i, betaParams in enumerate(self.arms):
            theta = np.random.beta(betaParams[0], betaParams[1])
            if (theta > maxTheta):
                maxTheta = theta
                chosenArm = i
        return chosenArm

    def update(self, feedbacks):
        for feedback in feedbacks:
            self.update_focal(feedback)

    def printArms(self):
        print(self.arms)

    #Opening a gaussian around the rtt and samlpling indexes around it to promote
    def update_focal(self, feedback):
        idxes = np.round(np.random.normal(feedback.rtt, self.variance, self.num_scatter))
        for idx in idxes:
            if(0 <= idx <= RTT_RANGE):
                self.arms[int(idx)][PROMOTE_INDEX] = min(MAX_REWARDS, self.arms[int(idx)][0] + 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))