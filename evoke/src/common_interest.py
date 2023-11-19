"""
Analyses of common interest
"""
import itertools as it
import numpy as np
from scipy.stats import kendalltau
from scipy.special import comb

from evoke.src import exceptions


class CommonInterest_1_pop:
    """
    Calculate quantities useful for the study of the degree of common interest
    between senders and receivers
    """

    def __init__(self, game):
        self.player = game.payoff_matrix
        try:
            self.state_chances = game.state_chances
            self.chance_node = True
        except AttributeError:
            self.chance_node = False

    def K(self, array):
        """
        Calculate K as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return intra_tau(self.state_chances, array)

    def sender_K(self):
        """
        Calculate K for the sender
        """
        return self.K(self.sender)

    def receiver_K(self):
        """
        Calculate K for the receiver
        """
        return self.K(self.receiver)

    def C_chance(self):
        """
        Calculate C as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return tau_per_rows(self.state_chances, self.sender, self.receiver)

    def C_nonchance(self):
        """
        Calculate the C for non-chance games (using the total KTD)
        """
        return total_tau(self.sender, self.receiver)


class CommonInterest_2_pops:
    """
    Calculate quantities useful for the study of the degree of common interest
    between senders and receivers
    """

    def __init__(self, game):
        self.sender = game.sender_payoff_matrix
        self.receiver = game.receiver_payoff_matrix
        try:
            self.state_chances = game.state_chances
            self.chance_node = True
        except AttributeError:
            self.chance_node = False

    def K(self, array):
        """
        Calculate K as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return intra_tau(self.state_chances, array)

    def sender_K(self):
        """
        Calculate K for the sender
        """
        return self.K(self.sender)

    def receiver_K(self):
        """
        Calculate K for the receiver
        """
        return self.K(self.receiver)

    def C_chance(self):
        """
        Calculate C as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return tau_per_rows(self.state_chances, self.sender, self.receiver)

    def C_nonchance(self):
        """
        Calculate the C for non-chance games (using the total KTD)
        """
        return total_tau(self.sender, self.receiver)


def intra_tau(unconds, array):
    """
    Calculate the average (weighted by <unconds> of the pairwise Kendall's tau
    distance between rows (states) of <array>
    """
    taus = np.array(
        [kendalltau(row1, row2)[0] for row1, row2 in it.combinations(array, 2)]
    )
    return unconds.dot(taus)


def intra_tau(unconds, array):
    """
    Calculate the average (weighted by <unconds> of the pairwise Kendall's tau
    distance between rows (states) of <array>
    """
    taus = np.array(
        [kendalltau(row1, row2)[0] for row1, row2 in it.combinations(array, 2)]
    )
    return unconds.dot(taus)


class CommonInterest_2_pops:
    """
    Calculate quantities useful for the study of the degree of common interest
    between senders and receivers
    """

    def __init__(self, game):
        self.sender = game.sender_payoff_matrix
        self.receiver = game.receiver_payoff_matrix
        try:
            self.state_chances = game.state_chances
            self.chance_node = True
        except AttributeError:
            self.chance_node = False

    def K(self, array):
        """
        Calculate K as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return intra_tau(self.state_chances, array)

    def sender_K(self):
        """
        Calculate K for the sender
        """
        return self.K(self.sender)

    def receiver_K(self):
        """
        Calculate K for the receiver
        """
        return self.K(self.receiver)

    def C_chance(self):
        """
        Calculate C as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return tau_per_rows(self.state_chances, self.sender, self.receiver)

    def C_nonchance(self):
        """
        Calculate the C for non-chance games (using the total KTD)
        """
        return total_tau(self.sender, self.receiver)


def C(vector1, vector2):
    """
    Calculate C for two vectors
    """
    max_value = comb(len(vector1.flatten()), 2)
    return 1 - tau(vector1, vector2) / max_value


def tau(vector1, vector2):
    """
    Calculate the Kendall tau statistic among two vectors
    """
    vector1 = vector1.flatten()  # in case they are not vectors
    vector2 = vector2.flatten()
    comparisons1 = np.array(
        [np.sign(elem1 - elem2) for (elem1, elem2) in it.combinations(vector1, 2)]
    )
    comparisons2 = np.array(
        [np.sign(elem1 - elem2) for (elem1, elem2) in it.combinations(vector2, 2)]
    )
    return np.sum(np.abs(comparisons1 - comparisons2) > 1)


def intra_tau(unconds, array):
    """
    Calculate the average (weighted by <unconds> of the pairwise Kendall's tau
    distance between rows (states) of <array>
    """
    taus = np.array(
        [kendalltau(row1, row2)[0] for row1, row2 in it.combinations(array, 2)]
    )
    return unconds.dot(taus)


def total_tau(array1, array2):
    """
    Calculate the KTD between the flattened <array1> and <array2>. Useful for
    NonChance games
    """
    return kendalltau(array1, array2)[0]


def tau_per_rows(unconds, array1, array2):
    """
    Calculate the average (weighted by <unconds> of the Kendall's tau distance
    between the corresponding rows (states) of <array1> and <array2>
    """
    taus = np.array([kendalltau(row1, row2)[0] for row1, row2 in zip(array1, array2)])
    return unconds.dot(taus)


class Nash:
    """
    Calculate Nash equilibria
    """

    def __init__(self, game):
        self.game = game

    def receivers_vs_sender(self, sender):
        receivers = np.identity(self.game.lrs)
        return [
            self.game.receiver_avg_payoff(receiver, sender) for receiver in receivers
        ]

    def senders_vs_receiver(self, receiver):
        senders = np.identity(self.game.lss)
        return [self.game.sender_avg_payoff(sender, receiver) for sender in senders]

    def is_Nash(self, sender, receiver):
        """
        Find out if sender and receiver are a Nash eqb
        """
        payoffsender = self.game.sender_avg_payoff(sender, receiver)
        payoffreceiver = self.game.receiver_avg_payoff(receiver, sender)
        senderisbest = (
            abs(payoffsender - max(self.senders_vs_receiver(receiver))) < 1e-2
        )
        receiverisbest = (
            abs(payoffreceiver - max(self.receivers_vs_sender(sender))) < 1e-2
        )
        return senderisbest and receiverisbest


# What follow are some helper functions to ascertain whether a population has
# reached a state in which no more interesting changes should be expected


def stability(array):
    """
    Compute a coarse grained measure of the stability of the array
    """
    trans_array = array.T
    stable = np.apply_along_axis(stable_vector, 1, trans_array)
    if np.all(stable):
        return "stable"
    nonstable = trans_array[np.logical_not(stable)]
    periodic = np.apply_along_axis(periodic_vector, 1, nonstable)
    if np.all(periodic):
        return "periodic"
    else:
        return "nonstable"


def stable_vector(vector):
    """
    Return true if the vector does not move
    """
    return np.allclose(0, max(vector) - min(vector))


def periodic_vector(vector):
    """
    We take the FFT of a vector, and eliminate all components but the two main
    ones (i.e., the static and biggest sine amplitude) and compare the
    reconstructed wave with the original. Return true if close enough
    """
    rfft = np.fft.rfft(vector)
    magnitudes = np.abs(np.real(rfft))
    choice = magnitudes > sorted(magnitudes)[-3]
    newrfft = np.choose(choice, (np.zeros_like(rfft), rfft))
    newvector = np.fft.irfft(newrfft)
    return np.allclose(vector, newvector, atol=1e-2)
