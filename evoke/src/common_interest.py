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
        """
        Initialize the class with the payoff matrix of the game

        Parameters
        ----------
        game : One of the Evoke one-population games.
            The game to be analyzed
        """
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

        Parameters
        ----------
        array : 2D numpy array
            The payoff array to be analyzed

        Returns
        -------
        float
            The K value for the array
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return intra_tau(self.state_chances, array)

    def sender_K(self):
        """
        Calculate K for the sender

        Returns
        -------
        float
            The K value for the sender
        """
        return self.K(self.sender)

    def receiver_K(self):
        """
        Calculate K for the receiver

        Returns
        -------
        float
            The K value for the receiver
        """
        return self.K(self.receiver)

    def C_chance(self):
        """
        Calculate C as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau

        Returns
        -------
        float
            The C value for the chance node game
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return tau_per_rows(self.state_chances, self.sender, self.receiver)

    def C_nonchance(self):
        """
        Calculate the C for non-chance games (using the total KTD)

        Returns
        -------
        float
            The C value for non-chance games

        """
        return total_tau(self.sender, self.receiver)


class CommonInterest_2_pops:
    """
    Calculate quantities useful for the study of the degree of common interest
    between senders and receivers
    """

    def __init__(self, game):
        """
        Initialise the class with the payoff matrices of the game

        Parameters
        ----------
        game : One of the Evoke two-population games.
            The game to be analyzed
        """

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

        Parameters
        ----------
        array : 2D numpy array
            The payoff array to be analyzed

        Returns
        -------
        float
            The K value for the array
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return intra_tau(self.state_chances, array)

    def sender_K(self):
        """
        Calculate K for the sender

        Returns
        -------
        float
            The K value for the sender
        """
        return self.K(self.sender)

    def receiver_K(self):
        """
        Calculate K for the receiver

        Returns
        -------
        float
            The K value for the receiver
        """
        return self.K(self.receiver)

    def C_chance(self):
        """
        Calculate C as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau

        Returns
        -------
        float
            The C value for the chance node game
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return tau_per_rows(self.state_chances, self.sender, self.receiver)

    def C_nonchance(self):
        """
        Calculate the C for non-chance games (using the total KTD)

        Returns
        -------
        float
            The C value for non-chance games
        """
        return total_tau(self.sender, self.receiver)


def intra_tau(unconds, array):
    """
    Calculate the average (weighted by <unconds> of the pairwise Kendall's tau
    distance between rows (states) of <array>

    Parameters
    ----------
    unconds : 1D numpy array
        The unconditioned probabilities of the states
    array : 2D numpy array
        The payoff array to be analyzed
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
        """
        Initialise the class with the payoff matrices of the game

        Parameters
        ----------
        game : One of the Evoke two-population games.
            The game to be analyzed
        """

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

        Parameters
        ----------
        array : 2D numpy array
            The payoff array to be analyzed

        Returns
        -------
        float
            The K value for the array
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return intra_tau(self.state_chances, array)

    def sender_K(self):
        """
        Calculate K for the sender

        Returns
        -------
        float
            The K value for the sender
        """
        return self.K(self.sender)

    def receiver_K(self):
        """
        Calculate K for the receiver

        Returns
        -------
        float
            The K value for the receiver
        """
        return self.K(self.receiver)

    def C_chance(self):
        """
        Calculate C as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau

        Returns
        -------
        float
            The C value for the chance node game
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return tau_per_rows(self.state_chances, self.sender, self.receiver)

    def C_nonchance(self):
        """
        Calculate the C for non-chance games (using the total KTD)

        Returns
        -------
        float
            The C value for non-chance games
        """
        return total_tau(self.sender, self.receiver)


def C(vector1, vector2):
    """
    Calculate C for two vectors

    Parameters
    ----------
    vector1 : 1D numpy array
        The first vector to be compared
    vector2 : 1D numpy array
        The second vector to be compared

    Returns
    -------
    float
        The C value for the two vectors
    """
    max_value = comb(len(vector1.flatten()), 2)
    return 1 - tau(vector1, vector2) / max_value


def tau(vector1, vector2):
    """
    Calculate the Kendall tau statistic among two vectors

    Parameters
    ----------
    vector1 : 1D numpy array
        The first vector to be compared
    vector2 : 1D numpy array
        The second vector to be compared

    Returns
    -------
    float
        The Kendall tau statistic for the two vectors
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


def total_tau(array1, array2):
    """
    Calculate the KTD between the flattened <array1> and <array2>. Useful for
    NonChance games

    Parameters
    ----------
    array1 : 2D numpy array
        The first array to be compared
    array2 : 2D numpy array
        The second array to be compared

    Returns
    -------
    float
        The KTD between the two arrays
    """
    return kendalltau(array1, array2)[0]


def tau_per_rows(unconds, array1, array2):
    """
    Calculate the average (weighted by <unconds> of the Kendall's tau distance
    between the corresponding rows (states) of <array1> and <array2>

    Parameters
    ----------
    unconds : 1D numpy array
        The unconditioned probabilities of the states
    array1 : 2D numpy array
        The first array to be compared
    array2 : 2D numpy array
        The second array to be compared

    Returns
    -------
    float
        The KTD between the two arrays
    """
    taus = np.array([kendalltau(row1, row2)[0] for row1, row2 in zip(array1, array2)])
    return unconds.dot(taus)


class Nash:
    """
    Calculate Nash equilibria
    """

    def __init__(self, game):
        """
        Initialize the class with the game to be analyzed
        """
        self.game = game

    def receivers_vs_sender(self, sender):
        """
        Calculate the list a receiver's average payoff against all possible
        senders

        Parameters
        ----------
        sender : 1D numpy array
            The sender to be analyzed

        Returns
        -------
        list
            The list of average payoffs for the receiver against all possible
            senders
        """
        receivers = np.identity(self.game.lrs)
        return [
            self.game.receiver_avg_payoff(receiver, sender) for receiver in receivers
        ]

    def senders_vs_receiver(self, receiver):
        """
        Calculate the list a sender's average payoff against all possible
        receivers

        Parameters
        ----------
        receiver : 1D numpy array
            The receiver to be analyzed

        Returns
        -------
        list
            The list of average payoffs for the sender against all possible
            receivers
        """
        senders = np.identity(self.game.lss)
        return [self.game.sender_avg_payoff(sender, receiver) for sender in senders]

    def is_Nash(self, sender, receiver):
        """
        Find out if sender and receiver are a Nash eqb

        Parameters
        ----------
        sender : 1D numpy array
            The sender to be analyzed
        receiver : 1D numpy array
            The receiver to be analyzed

        Returns
        -------
        bool
            True if sender and receiver are a Nash equilibrium
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

    Parameters
    ----------
    array : 2D numpy array
        The array to be analyzed

    Returns
    -------
    str
        "stable" if the array is stable, "periodic" if it is periodic, and
        "nonstable" if it is not stable
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

    Parameters
    ----------
    vector : 1D numpy array
        The vector to be analyzed

    Returns
    -------
    bool
        True if the vector is stable
    """
    return np.allclose(0, max(vector) - min(vector))


def periodic_vector(vector):
    """
    We take the FFT of a vector, and eliminate all components but the two main
    ones (i.e., the static and biggest sine amplitude) and compare the
    reconstructed wave with the original. Return true if close enough

    Parameters
    ----------
    vector : 1D numpy array
        The vector to be analyzed

    Returns
    -------
    bool
        True if the vector is periodic
    """
    rfft = np.fft.rfft(vector)
    magnitudes = np.abs(np.real(rfft))
    choice = magnitudes > sorted(magnitudes)[-3]
    newrfft = np.choose(choice, (np.zeros_like(rfft), rfft))
    newvector = np.fft.irfft(newrfft)
    return np.allclose(vector, newvector, atol=1e-2)
