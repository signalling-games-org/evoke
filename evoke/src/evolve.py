# -*- coding: utf-8 -*-
"""
Calculate equations to evolve populations in a game. Right now, we can
calculate the replicator (-mutator) dynamics, with one or two populations, in
discrete or continuous time
"""

## Built-in
from abc import abstractmethod
import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint

## Libraries
from tqdm import trange

## evoke
from evoke.src.info import Information


class OnePop:
    """
    Calculate the equations necessary to evolve one population. It takes as
    input a <game> and an array such that the <i,j> cell gives the expected
    payoff for the-strategist player of an encounter in which they  follow
    strategy i and their opponent follows strategy j.
    """

    def __init__(self, game, playertypes):
        """
        Initialise the class with a game and a matrix of payoffs.

        Parameters
        ----------
        game : one of the evoke game objects
            The game that this evolve class will provide the dynamics for.
        playertypes : array-like
            Array of available player types.
        """
        self.game = game
        self.avgpayoffs = self.game.avg_payoffs(playertypes)
        self.playertypes = playertypes
        self.lps = self.playertypes.shape[0]
        # By default, mutation matrices are the identity matrices. You can
        # change that.
        self.mm = np.identity(self.lps)
        # We can set a limit of precision in the calculation of diffeqs, to
        # avoid artifacts. By default, we do not
        self.precision = None

        ## Assortment defaults to zero
        self.e = 0

    def random_player(self):
        """
        Return frequencies of a random sender population

        Returns
        -------
        array-like
            Array of frequencies of a random sender population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lps)])

    def avg_payoff(self, player):
        """
        Return the average payoff that players get when the population vector
        is <player>

        Parameters
        ----------
        player : array-like
            Population vector describing a mixed population.

        Returns
        -------
        float
            Average payoff of the population.
        """
        return player.dot(self.avgpayoffs.dot(player))

    def avg_payoff_vector(self, player):
        """
        Get expected payoffs of every type when population vector is <player>.

        Depends on assortment.

        p(s_i meets s_i) = p(s_i) + self.e * (1-p(s_i))
        p(s_i meets s_j) = p(s_j) - self.e * p(s_j)

        Parameters
        ----------
        player : array-like
            Population vector describing a mixed population.

        Returns
        -------
        array-like
            Array of average payoffs of the population.
        """

        ## Use the simple version if there is no assortment.
        if self.e == 0:
            return player @ self.avgpayoffs

        # Create a square array of copies of <player>
        meeting_probabilities = np.tile(player, (self.lps, 1))

        # Modify meeting probabilities depending on level of assortment
        meeting_probabilities += self.e * (np.eye(self.lps) - meeting_probabilities)

        # Fancy vectorized sum to figure out the payoffs
        avg_payoffs = np.einsum("ij,ji->i", meeting_probabilities, self.avgpayoffs)

        return avg_payoffs

    def replicator_dX_dt_odeint(self, X, t):
        """
        Calculate the rhs of the system of odes for scipy.integrate.odeint

        Parameters
        ----------
        X : array-like
            Population vector at time t.
        t : float
            Time.

        Returns
        -------
        array-like
            Right-hand side of the system of ODEs.
        """
        avgpayoff = self.avg_payoff(X)
        result = X * (self.avgpayoffs @ X - avgpayoff)
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

    def replicator_jacobian_odeint(self, X, t=0):
        """
        Calculate the Jacobian of the system of odes for scipy.integrate.odeint

        Parameters
        ----------
        X : array-like
            Population vector at time t.
        t : float
            Time.

        Returns
        -------
        array-like
            Jacobian of the system of ODEs.
        """
        avgpayoff = self.avg_payoff(X)
        diagonal = np.eye(self.lps) * X
        jacobian = self.avgpayoffs * diagonal + self.avgpayoffs @ X - avgpayoff
        return jacobian

    def discrete_replicator_delta_X(self, X):
        """
        Calculate a population vector for t' given the vector for t, using the
        discrete time replicator dynamics (Huttegger 2007)

        Parameters
        ----------
        X : array-like
            Population vector at time t.

        Returns
        -------
        array-like
            Population vector at time t+1.
        """
        avgfitness = self.avg_payoff(X)
        delta = X * (self.avgpayoffs @ X) / avgfitness
        if self.precision:
            np.around(delta, decimals=self.precision, out=delta)
        return delta

    def replicator_odeint(self, init, time_vector, **kwargs):
        """
        Calculate one run of the game following the replicator(-mutator)
        dynamics, with starting points sinit and rinit, in times <times> (an
        evolve.Times instance), using scipy.integrate.odeint

        Parameters
        ----------
        init : array-like
            Initial population vector.
        time_vector : array-like
            Time vector.
        kwargs : dict
            Additional keyword arguments for odeint.

        Returns
        -------
        array-like
            Population vectors at each time point.
        """
        return odeint(
            self.replicator_dX_dt_odeint,
            init,
            time_vector,
            Dfun=self.replicator_jacobian_odeint,
            col_deriv=True,
            **kwargs
        )

    def replicator_discrete(self, initpop, steps):
        """
        Calculate one run of the game, following the discrete
        replicator(-mutator) dynamics, for <steps> steps with
        starting population vector <initpop> using the discrete time
        replicator dynamics.

        Parameters
        ----------
        initpop : array-like
            Initial population vector.
        steps : int
            Number of steps to take.

        Returns
        -------
        array-like
            Population vectors at each time point.
        """
        data = np.empty([steps, len(initpop)])
        data[0, :] = initpop
        for i in range(1, steps):
            data[i, :] = self.discrete_replicator_delta_X(data[i - 1, :])
        return data

    def pop_to_mixed_strat(self, pop):
        """
        Take a population vector and output the average strat implemented by
        the whole population

        Parameters
        ----------
        pop : array-like
            Population vector.

        Returns
        -------
        array-like
            Average strategy implemented by the whole population.
        """
        return self.game.calculate_mixed_strat(self.playertypes, pop)

    def assortment(self, e):
        """
        Set assortment level

        Parameters
        ----------
        e : float
            Amount of assortment as a proportion of the population.

        Returns
        -------
        None.

        """

        self.e = e


class TwoPops:
    """
    Calculate the equations necessary to evolve a population of senders and one
    of receivers. It takes as input a <game>, (which as of now only can be a
    Chance object), and a tuple: the first (second) member of the tuple is a
    nxm array such that the <i,j> cell gives the expected payoff for the sender
    (receiver) of an encounter in which the sender follows strategy i and the
    receiver follows strategy j.
    """

    def __init__(self, game, sendertypes, receivertypes):
        """
        Initialise the class with a game and two matrices of payoffs.

        Parameters
        ----------
        game : one of the evoke game objects
            The game that this evolve class will provide the dynamics for.
        sendertypes : array-like
            Array of available sender types.
        receivertypes : array-like
            Array of available receiver types.

        """
        self.game = game
        avgpayoffs = self.game.avg_payoffs(sendertypes, receivertypes)
        self.senderpayoffs = avgpayoffs[0]
        self.receiverpayoffs = avgpayoffs[1].T
        self.sendertypes = sendertypes
        self.receivertypes = receivertypes
        self.lss, self.lrs = self.senderpayoffs.shape
        # By default, mutation matrices are the identity matrices. You can
        # change that.
        self.mm_sender = np.identity(self.senderpayoffs.shape[0])
        self.mm_receiver = np.identity(self.receiverpayoffs.shape[0])
        # We can set a limit of precision in the calculation of diffeqs, to
        # avoid artifacts. By default, we do not
        self.precision = None

    def random_sender(self):
        """
        Return frequencies of a random sender population

        Returns
        -------
        array-like
            Array of frequencies of a random sender population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lss)])

    def random_receiver(self):
        """
        Return frequencies of a random receiver population

        Returns
        -------
        array-like
            Array of frequencies of a random receiver population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lrs)])

    def sender_avg_payoff(self, sender, receiver):
        """
        Return the average payoff that senders get when the population vectors
        are <sender> and <receiver>

        Parameters
        ----------
        sender : array-like
            Population vector describing a mixed sender population.
        receiver : array-like
            Population vector describing a mixed receiver population.

        Returns
        -------
        float
            Average payoff of the sender population.
        """
        return sender.dot(self.senderpayoffs.dot(receiver))

    def receiver_avg_payoff(self, receiver, sender):
        """
        Return the average payoff that receivers get when the population
        vectors are <sender> and <receiver>

        Parameters
        ----------
        sender : array-like
            Population vector describing a mixed sender population.
        receiver : array-like
            Population vector describing a mixed receiver population.

        Returns
        -------
        float
            Average payoff of the receiver population.
        """
        return receiver.dot(self.receiverpayoffs.dot(sender))

    def replicator_dX_dt_odeint(self, X, t):
        """
        Calculate the rhs of the system of odes for scipy.integrate.odeint

        Parameters
        ----------
        X : array-like
            Population vector at time t.
        t : float
            Time.

        Returns
        -------
        array-like
            Right-hand side of the system of ODEs.
        """
        # X's first part is the sender vector
        # its second part the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs * senderpops[..., None]).dot(receiverpops).dot(
            self.mm_sender
        ) - senderpops * avgfitsender
        receiverdot = (self.receiverpayoffs * receiverpops[..., None]).dot(
            senderpops
        ).dot(self.mm_receiver) - receiverpops * avgfitreceiver
        result = np.concatenate((senderdot, receiverdot))
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

    def replicator_dX_dt_ode(self, t, X):
        """
        Calculate the rhs of the system of odes for scipy.integrate.ode

        Parameters
        ----------
        t : float
            Time.
        X : array-like
            Population vector at time t.

        Returns
        -------
        array-like
            Right-hand side of the system of ODEs.
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs * senderpops[..., None]).dot(receiverpops).dot(
            self.mm_sender
        ) - senderpops * avgfitsender
        receiverdot = (self.receiverpayoffs * receiverpops[..., None]).dot(
            senderpops
        ).dot(self.mm_receiver) - receiverpops * avgfitreceiver
        result = np.concatenate((senderdot, receiverdot))
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

    def replicator_jacobian_odeint(self, X, t=0):
        """
        Calculate the Jacobian of the system for scipy.integrate.odeint

        Parameters
        ----------
        X : array-like
            Population vector at time t.
        t : float
            Time.

        Returns
        -------
        array-like
            Jacobian of the system of ODEs.
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        yS = self.senderpayoffs.dot(receiverpops)  # [y1P11 +...+ ynP1n, ...,
        # y1Pn1 + ... ynPnn] This one works
        xS = self.senderpayoffs * senderpops[..., None]
        xR = self.receiverpayoffs.dot(senderpops)
        yR = self.receiverpayoffs * receiverpops[..., None]
        tile1 = (self.mm_sender - senderpops[..., None]) * yS - np.identity(
            self.lss
        ) * avgfitsender
        tile2 = (self.mm_sender - senderpops).transpose().dot(xS)
        tile3 = (self.mm_receiver - receiverpops).transpose().dot(yR)
        tile4 = (self.mm_receiver - receiverpops[..., None]) * xR - np.identity(
            self.lrs
        ) * avgfitreceiver
        lefthalf = np.vstack((tile1.transpose(), tile2.transpose()))
        righthalf = np.vstack((tile3.transpose(), tile4.transpose()))
        jac = np.hstack((lefthalf, righthalf))
        if self.precision:
            np.around(jac, decimals=self.precision, out=jac)
        return jac

    def replicator_jacobian_ode(self, t, X):
        """
        Calculate the Jacobian of the system for scipy.integrate.ode

        Parameters
        ----------
        t : float
            Time.
        X : array-like
            Population vector at time t.

        Returns
        -------
        array-like
            Jacobian of the system of ODEs.
        """
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        yS = self.senderpayoffs.dot(receiverpops)  # [y1P11 +...+ ynP1n, ...,
        # y1Pn1 + ... ynPnn] This one works
        xS = self.senderpayoffs * senderpops[..., None]
        xR = self.receiverpayoffs.dot(senderpops)
        yR = self.receiverpayoffs * receiverpops[..., None]
        tile1 = (self.mm_sender - senderpops[..., None]) * yS - np.identity(
            self.lss
        ) * avgfitsender
        tile2 = (self.mm_sender - senderpops).transpose().dot(xS)
        tile3 = (self.mm_receiver - receiverpops).transpose().dot(yR)
        tile4 = (self.mm_receiver - receiverpops[..., None]) * xR - np.identity(
            self.lrs
        ) * avgfitreceiver
        lefthalf = np.vstack((tile1.transpose(), tile2.transpose()))
        righthalf = np.vstack((tile3.transpose(), tile4.transpose()))
        jac = np.hstack((lefthalf, righthalf))
        if self.precision:
            np.around(jac, decimals=self.precision, out=jac)
        return jac.transpose()

    def discrete_replicator_delta_X(self, X):
        """
        Calculate a population vector for t' given the vector for t, using the
        discrete time replicator dynamics (Huttegger 2007)

        Parameters
        ----------
        X : array-like
            Population vector at time t.

        Returns
        -------
        array-like
            Population vector at time t+1.
        """
        # X's first part is the sender vector
        # its second part the receiver vector
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdelta = (self.senderpayoffs * senderpops[..., None]).dot(
            receiverpops
        ).dot(self.mm_sender) / avgfitsender
        receiverdelta = (self.receiverpayoffs * receiverpops[..., None]).dot(
            senderpops
        ).dot(self.mm_receiver) / avgfitreceiver
        result = np.concatenate((senderdelta, receiverdelta))
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

    def replicator_odeint(self, sinit, rinit, times, **kwargs):
        """
        Calculate one run of the game following the replicator(-mutator)
        dynamics, with starting points sinit and rinit, in times <times> (an
        evolve.Times instance), using scipy.integrate.odeint

        Parameters
        ----------
        sinit : array-like
            Initial sender population vector.
        rinit : array-like
            Initial receiver population vector.
        times : evolve.Times object
        kwargs : dict
            Additional keyword arguments for odeint.

        Returns
        -------
        array-like
            Population vectors at each time point.
        """
        return odeint(
            self.replicator_dX_dt_odeint,
            np.concatenate((sinit, rinit)),
            times.time_vector,
            Dfun=self.replicator_jacobian_odeint,
            col_deriv=True,
            **kwargs
        )

    def replicator_ode(self, sinit, rinit, times, integrator="dopri5"):
        """
        Calculate one run of the game, following the replicator(-mutator)
        dynamics in continuous time, in <times> (an evolve.Times instance) with
        starting points sinit and rinit using scipy.integrate.ode

        Parameters
        ----------
        sinit : array-like
            Initial sender population vector.
        rinit : array-like
            Initial receiver population vector.
        times : evolve.Times object
        integrator : str
            Name of the integrator to use.

        Returns
        -------
        array-like
            Population vectors at each time point.
        """
        initialpop = np.concatenate((sinit, rinit))
        equations = ode(
            self.replicator_dX_dt_ode, self.replicator_jacobian_ode
        ).set_integrator(integrator)
        equations.set_initial_value(initialpop, times.initial_time)
        while equations.successful() and equations.t < times.final_time:
            newdata = equations.integrate(equations.t + times.time_inc)
            try:
                data = np.append(data, [newdata], axis=0)
            except NameError:
                data = [newdata]
        return data

    def replicator_discrete(self, sinit, rinit, times):
        """
        Calculate one run of the game, following the discrete
        replicator(-mutator) dynamics, in <times> (an evolve.Times object) with
        starting population vector <popvector> using the discrete time
        replicator dynamics. Note that this solver will just calculate n points
        in the evolution of the population, and will not try to match them to
        the times as provided.

        Parameters
        ----------
        sinit : array-like
            Initial sender population vector.
        rinit : array-like
            Initial receiver population vector.
        times : evolve.Times object

        Returns
        -------
        array-like
            Population vectors at each time point.
        """
        popvector = np.concatenate((sinit, rinit))
        data = np.empty([len(times.time_vector), len(popvector)])
        data[0, :] = popvector
        for i in range(1, len(times.time_vector)):
            data[i, :] = self.discrete_replicator_delta_X(data[i - 1, :])
        return data

    def vector_to_populations(self, vector):
        """
        Take one of the population vectors returned by the solvers, and output
        two vectors, for the sender and receiver populations respectively.

        Parameters
        ----------
        vector : array-like
            Population vector.

        Returns
        -------
        tuple
            Tuple of two arrays, one for the sender and one for the receiver
            populations.
        """
        return np.hsplit(vector, [self.lss])

    def sender_to_mixed_strat(self, senderpop):
        """
        Take a sender population vector and output the average
        sender strat implemented by the whole population

        Parameters
        ----------
        senderpop : array-like
            Sender population vector.

        Returns
        -------
        array-like
            Average sender strategy implemented by the whole population.
        """
        return self.game.calculate_sender_mixed_strat(self.sendertypes, senderpop)

    def receiver_to_mixed_strat(self, receiverpop):
        """
        Take a receiver population vector and output the average
        receiver strat implemented by the whole population

        Parameters
        ----------
        receiverpop : array-like
            Receiver population vector.

        Returns
        -------
        array-like
            Average receiver strategy implemented by the whole population
        """
        return self.game.calculate_receiver_mixed_strat(self.receivertypes, receiverpop)


class Reinforcement:
    """
    Evolving finite sets of agents by reinforcement learning.
    """

    def __init__(self, game, agents):
        """
        The game type determines the number of agents.

        Parameters
        ----------
        game : one of the evoke game objects
            The game that this evolve class will provide the dynamics for.
        agents: array-like
            list of evoke agent objects

        Returns
        -------
        None.

        """

        self.game = game
        self.agents = agents

        ## Initialise
        self.reset()

    def reset(self):
        """
        Initialise values and agents.

        Returns
        -------
        None.

        """

        ## Current iteration step is set to zero.
        self.iteration = 0

    def run(self, iterations, hide_progress=True, calculate_stats="step"):
        """
        Run the simulation for <iterations> steps.

        Parameters
        ----------
        iterations : int
            Number of times to call self.step().
        hide_progress : bool
            Whether to display tqdm progress bar
        calculate_stats : str
            When to calculate stats
            "step": every step
            "end": only at the last step

        Returns
        -------
        None.

        """

        for _ in trange(iterations, disable=hide_progress):
            if calculate_stats == "step":
                self.step(calculate_stats=True)
            if calculate_stats == "end":
                self.step(calculate_stats=False)

        if calculate_stats == "end":
            self.calculate_stats()

    def is_pooling(self, epsilon=1e-3):
        """
        Determine whether the current strategies are pooling or
        a signalling system.

        If the mutual information between states and acts at the current
        population state is within <epsilon> of the maximum possible,
        it's a signalling system.
        Otherwise, it's pooling.

        Clearly if the number of signals is lower than both the number of states
        and the number of acts, it will necessarily be pooling.

        Parameters
        ----------
        epsilon : float
            How close to the maximum possible mutual information must
            the current mutual information be in order to count
            as a signalling system?

        Returns
        -------
        pooling : bool
            True if the current strategies constitute a pooling equilibrium.

        """

        ## Check that the game is of the appropriate type for this method.
        ## i.e. In order to determine whether a simulation is currently pooling,
        ## there must minimally be states, signals and acts.
        assert (
            hasattr(self.game, "states")
            and hasattr(self.game, "messages")
            and hasattr(self.game, "acts")
        ), "Game must have states, messages and acts to determine pooling."

        ##########################################################
        ## NB: This is not true because the state_chances might be uneven enough
        ##      that a smaller number of signals is still sufficient.
        ##     Therefore we should let the direct comparison with self.game.max_mutual_info
        ##      do all the work.
        # ## Quick check: Is the number of signals lower than both the number of states
        # ##               and the number of acts?
        # ##              If so, the system must be pooling.
        # ##              The channel is not wide enough to achieve the maximum rate.
        # if self.game.messages < self.game.states and self.game.messages < self.game.acts:
        #     return True
        ##########################################################

        ## Now we know we have enough signals that we could potentially have a signalling system.
        ## Figure out the maximum possible mutual information,
        ##  and then see whether we are within <epsilon> of that.

        ## Maximum mutual information is determined by self.game.state_chances.
        ## Ask the game what it's maximum mutual information is.
        max_mutual_info = self.game.max_mutual_info

        ## The current mutual information
        ## Normalise strategy profiles
        snorm = (self.sender.strategies.T / self.sender.strategies.sum(1)).T
        rnorm = (self.receiver.strategies.T / self.receiver.strategies.sum(1)).T

        ## Create info object and get mutual information
        info = Information(self.game, snorm, rnorm)
        mut_info_states_acts = info.mutual_info_states_acts()

        ## Check we are within <epsilon> of the maximum.
        if mut_info_states_acts > max_mutual_info - epsilon:
            ## Current mutual info is within epsilon of the maximum.
            ## Therefore it's a signalling system, therefore it isn't pooling.
            return False

        ## It's not a signalling system, so it's pooling.
        return True

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def calculate_stats(self):
        pass


class Matching(Reinforcement):
    """
    Reinforcement according to Richard Herrnstein’s matching law.
    The probability of choosing an action is proportional to its accumulated rewards.
    """

    def __init__(self, game, agents):
        """
        Initialise the class with a game and a list of agents.

        Parameters
        ----------
        game : one of the evoke game objects
            The game that this evolve class will provide the dynamics for.
        agents: array-like
            list of evoke agent objects
        """
        super().__init__(game=game, agents=agents)


class MatchingSR(Matching):
    """
    Matching simulation for two-player sender-receiver game.
    """

    def __init__(self, game, sender_strategies, receiver_strategies):
        """
        Initialise an instance of the MatchingSR class.

        Parameters
        ----------
        game : one of the Evoke game objects.
            The game that this evolve class will provide the dynamics for.
        sender_strategies : array-like
            Array of available sender strategies (each of which is itself an array).
        receiver_strategies : array-like
            Array of available receiver strategies (each of which is itself an array).

        Returns
        -------
        None.

        """

        self.game = game

        ## Create the agents.
        ## Sender
        self.sender = Sender(sender_strategies)

        ## Receiver
        self.receiver = Receiver(receiver_strategies)

        super().__init__(game=game, agents=[self.sender, self.receiver])

    def step(self, calculate_stats=True):
        """
        Implement the matching rule and increment one step.

        In each step:
            1. Run one round of the game.
            2. Update the agents' strategies based on the payoffs they received.
            3. Calculate and store any required variables e.g. information.
            4. Update iteration.

        Parameters
        ----------
        calculate_stats : bool
            Whether to calculate statistics at the end of the step.

        Returns
        -------
        None.

        """

        ## 1. Run one round of the game.
        ## 1a. Choose a nature state.
        state = self.game.choose_state()

        ## 1b. Choose a signal.
        signal = self.sender.choose_strategy(state)

        ## 1c. Choose an act.
        act = self.receiver.choose_strategy(signal)

        ## 1d. get the payoff
        sender_payoff = self.game.sender_payoff(state, act)
        receiver_payoff = self.game.receiver_payoff(state, act)

        ## 2. Update the agent's strategies based on the payoffs they received.
        self.sender.update_strategies(state, signal, sender_payoff)
        self.receiver.update_strategies(signal, act, receiver_payoff)

        ## 3. Calculate and store any required variables e.g. information.
        if calculate_stats:
            self.calculate_stats()

        ## 4. Update iteration.
        self.iteration += 1

    def calculate_stats(self):
        """
        Calculate and store informational quantities at this point.

        Returns
        -------
        None.

        """

        ## Lazy instantiation
        if not hasattr(self, "statistics"):
            self.statistics = {}

        ## 1. Mutual information between states and signals
        if "mut_info_states_signals" not in self.statistics:
            ## Create empty array...
            self.statistics["mut_info_states_signals"] = np.empty((self.iteration,))

            ## ...and fill with NaN up to current iteration.
            self.statistics["mut_info_states_signals"][:] = np.nan

        ## Get the current mutual information
        ## Normalise strategy profiles
        snorm = (self.sender.strategies.T / self.sender.strategies.sum(1)).T
        rnorm = (self.receiver.strategies.T / self.receiver.strategies.sum(1)).T

        ## Create info object and get mutual information
        info = Information(self.game, snorm, rnorm)
        mut_info_states_signals = info.mutual_info_states_messages()

        ## Append the current mutual information
        self.statistics["mut_info_states_signals"] = np.append(
            self.statistics["mut_info_states_signals"], mut_info_states_signals
        )

        ## 2. Average probability of success
        if "avg_prob_success" not in self.statistics:
            ## Create empty array...
            self.statistics["avg_prob_success"] = np.empty((self.iteration,))

            ## ...and fill with NaN up to current iteration.
            self.statistics["avg_prob_success"][:] = np.nan

        ## Get the current average probability of success
        avg_prob_success = np.average(
            self.game.payoff(sender_strat=snorm, receiver_strat=rnorm)
        )

        ## Append the current probability of success
        self.statistics["avg_prob_success"] = np.append(
            self.statistics["avg_prob_success"], avg_prob_success
        )


class MatchingSRInvention(Matching):
    """
    Matching simulation for two-player sender-receiver game,
     with the possibility of new signals at every step.
    """

    def __init__(self, game, sender_strategies, receiver_strategies):
        """
        Initialise an instance of the MatchingSRInvention class.

        Parameters
        ----------
        game : one of the Evoke game objects.
            The game that this evolve class will provide the dynamics for.
        sender_strategies : array-like
            Array of available sender strategies (each of which is itself an array).
        receiver_strategies : array-like
            Array of available receiver strategies (each of which is itself an array).

        Returns
        -------
        None.

        """

        self.game = game

        ## Create the agents.
        ## Sender
        self.sender = Sender(sender_strategies)

        ## Receiver
        self.receiver = Receiver(receiver_strategies)

        super().__init__(game=game, agents=[self.sender, self.receiver])

    def step(self, calculate_stats=True):
        """
        Implement the matching rule and increment one step.

        In each step:
            1. Run one round of the game.
            2. Update the agents' strategies based on the payoffs they received.
            3. Calculate and store any required variables e.g. information.
            4. Update iteration.

        Parameters
        ----------
        calculate_stats : bool
            Whether to calculate statistics at the end of the step.

        Returns
        -------
        None.

        """

        ## 1. Run one round of the game.
        ## 1a. Choose a nature state.
        state = self.game.choose_state()

        ## 1b. Choose a signal.
        signal = self.sender.choose_strategy(state)

        ## 1c. Choose an act.
        act = self.receiver.choose_strategy(signal)

        ## 1d. get the payoff
        sender_payoff = self.game.sender_payoff(state, act)
        receiver_payoff = self.game.receiver_payoff(state, act)

        ## 2a. Update the agent's strategies based on the payoffs they received.
        self.sender.update_strategies(state, signal, sender_payoff)
        self.receiver.update_strategies(signal, act, receiver_payoff)

        ## 2b. If the signal was novel and there was success,
        ##      need to add the new signal to both agents' repertoires.
        if signal == len(self.receiver.strategies) - 1 and receiver_payoff == 1:
            ## It was a new signal
            self.sender.add_signal()
            self.receiver.add_signal()

        ## 3. Calculate and store any required variables e.g. information.
        if calculate_stats:
            self.calculate_stats()

        ## 4. Update iteration.
        self.iteration += 1

    def calculate_stats(self):
        """
        Calculate and store informational quantities at this point.

        Returns
        -------
        None.

        """

        ## Lazy instantiation
        if not hasattr(self, "statistics"):
            self.statistics = {}

        ## 1. Number of signals

        ## 1. Mutual information between states and signals
        if "number_of_signals" not in self.statistics:
            ## Create empty array...
            self.statistics["number_of_signals"] = np.empty((self.iteration,))

            ## ...and fill with NaN up to current iteration.
            self.statistics["number_of_signals"][:] = np.nan

        ## Append the current number of signals
        self.statistics["number_of_signals"] = np.append(
            self.statistics["number_of_signals"], len(self.receiver.strategies) - 1
        )


class MatchingSIR(Matching):
    """
    Reinforcement game for sender, intermediary, receiver.
    """

    def __init__(
        self, game, sender_strategies, intermediary_strategies, receiver_strategies
    ):
        """
        Initialise an instance of the MatchingSIR class.

        Parameters
        ----------
        game : one of the Evoke game objects.
            The game that this evolve class will provide the dynamics for.
        sender_strategies : array-like
            Array of available sender strategies (each of which is itself an array).
        receiver_strategies : array-like
            Array of available receiver strategies (each of which is itself an array).

        Returns
        -------
        None.

        """

        self.game = game

        ## Create the agents.
        ## Sender
        self.sender = Sender(sender_strategies)

        ## Intermediary. There is no dedicated intermediary class.
        self.intermediary = Agent(intermediary_strategies)

        ## Receiver
        self.receiver = Receiver(receiver_strategies)

        super().__init__(
            game=game, agents=[self.sender, self.intermediary, self.receiver]
        )

    def step(self, calculate_stats=True):
        """
        Implement the matching rule and increment one step.

        In each step:
            1. Run one round of the game.
            2. Update the agents' strategies based on the payoffs they received.
            3. Calculate and store any required variables e.g. probability of success.
            4. Update iteration.

        Parameters
        ----------
        calculate_stats : bool
            Whether to calculate statistics at the end of the step.

        Returns
        -------
        None.
        """

        ## 1. Run one round of the game.
        ## 1a. Choose a nature state.
        state = self.game.choose_state()

        ## 1b. Choose a signal.
        signal_sender = self.sender.choose_strategy(state)

        ## 1b. Choose an intermediary signal.
        signal_intermediary = self.intermediary.choose_strategy(signal_sender)

        ## 1c. Choose an act.
        act = self.receiver.choose_strategy(signal_intermediary)

        ## 1d. get the payoff
        payoff_sender = self.game.payoff_sender(state, act)
        payoff_intermediary = self.game.payoff_intermediary(state, act)
        payoff_receiver = self.game.payoff_receiver(state, act)

        ## 2. Update the agent's strategies based on the payoffs they received.
        self.sender.update_strategies(state, signal_sender, payoff_sender)
        self.intermediary.update_strategies(
            state, signal_intermediary, payoff_intermediary
        )
        self.receiver.update_strategies(signal_intermediary, act, payoff_receiver)

        ## 3. Calculate and store any required variables e.g. probability of success.
        if calculate_stats:
            self.calculate_stats()

        ## 4. Update iteration.
        self.iteration += 1

    def calculate_stats(self):
        """
        Just call the record_probability_of_success method.
        """

        self.record_probability_of_success()

    def record_probability_of_success(self):
        """
        For now, just store "probability of success."

        Returns
        -------
        None.

        """

        ## Lazy instantiation
        if not hasattr(self, "statistics"):
            self.statistics = {}

        ## Probability of success.
        ## Requires game to be "regular" i.e. all agents have identical payoff matrices.
        assert self.game.regular

        if "prob_success" not in self.statistics:
            ## Create empty array...
            self.statistics["prob_success"] = np.empty((self.iteration,))

            ## ...and fill with NaN up to current iteration.
            self.statistics["prob_success"][:] = np.nan

        ## Get the current probability of success
        ## Normalise strategy profiles

        ## Sender strategy profile normalised
        ## snorm is an array; each row is a list of conditional probabilities.
        snorm = (self.sender.strategies.T / self.sender.strategies.sum(1)).T

        ## Intermediary strategy profile normalised
        ## inorm is an array; each row is a list of conditional probabilities.
        inorm = (self.intermediary.strategies.T / self.intermediary.strategies.sum(1)).T

        ## Receiver strategy profile normalised
        ## rnorm is an array; each row is a list of conditional probabilities.
        rnorm = (self.receiver.strategies.T / self.receiver.strategies.sum(1)).T

        ## Ask the game for the average payoff, given these strategies.
        ## Because payoffs are np.eye(2), this is equal to the probability of success.
        payoff = self.game.avg_payoffs_regular(snorm, inorm, rnorm)

        ## Append the current payoff
        self.statistics["prob_success"] = np.append(
            self.statistics["prob_success"], payoff
        )


class BushMostellerSR(Reinforcement):
    """
    Bush_mosteller reinforcement simulation for two-player sender-receiver game.
    """

    def __init__(
        self, game, sender_strategies, receiver_strategies, learning_parameter
    ):
        """
        Initialise an instance of the BushMostellerSR class.

        Parameters
        ----------
        game : one of the Evoke game objects.
            The game that this evolve class will provide the dynamics for.
        sender_strategies : array-like
            Array of available sender strategies (each of which is itself an array).
        receiver_strategies : array-like
            Array of available receiver strategies (each of which is itself an array).
        learning_parameter : float
            The learning parameter for the Bush-Mosteller reinforcement algorithm.

        Returns
        -------
        None.

        """

        self.game = game

        ## Create the agents.
        ## Sender
        self.sender = Sender(sender_strategies)

        ## Receiver
        self.receiver = Receiver(receiver_strategies)

        ## Learning parameter
        self.learning_parameter = learning_parameter

        super().__init__(game=game, agents=[self.sender, self.receiver])

    def step(self, calculate_stats=True):
        """
        Implement the matching rule and increment one step.

        In each step:
            1. Run one round of the game.
            2. Update the agents' strategies based on the payoffs they received.
            3. Calculate and store any required variables e.g. information.
            4. Update iteration.

        Parameters
        ----------
        calculate_stats : bool
            Whether to calculate statistics at the end of the step.

        Returns
        -------
        None.

        """

        ## 1. Run one round of the game.
        ## 1a. Choose a nature state.
        state = self.game.choose_state()

        ## 1b. Choose a signal.
        signal = self.sender.choose_strategy(state)

        ## 1c. Choose an act.
        act = self.receiver.choose_strategy(signal)

        ## 1d. get the payoff
        sender_payoff = self.game.sender_payoff(state, act)
        receiver_payoff = self.game.receiver_payoff(state, act)

        ## 2. Update the agent's strategies based on the payoffs they received.
        self.sender.update_strategies_bush_mosteller(
            state, signal, sender_payoff, self.learning_parameter
        )
        self.receiver.update_strategies_bush_mosteller(
            signal, act, receiver_payoff, self.learning_parameter
        )

        ## 3. Calculate and store any required variables e.g. information.
        if calculate_stats:
            self.calculate_stats()

        ## 4. Update iteration.
        self.iteration += 1

    def calculate_stats(self):
        """
        Calculate and store informational quantities at this point.

        Returns
        -------
        None.

        """

        ## Lazy instantiation
        if not hasattr(self, "statistics"):
            self.statistics = {}

        ## 1. Mutual information between states and signals
        if "mut_info_states_signals" not in self.statistics:
            ## Create empty array...
            self.statistics["mut_info_states_signals"] = np.empty((self.iteration,))

            ## ...and fill with NaN up to current iteration.
            self.statistics["mut_info_states_signals"][:] = np.nan

        ## Get the current mutual information
        ## Normalise strategy profiles
        snorm = (self.sender.strategies.T / self.sender.strategies.sum(1)).T
        rnorm = (self.receiver.strategies.T / self.receiver.strategies.sum(1)).T

        ## Create info object and get mutual information
        info = Information(self.game, snorm, rnorm)
        mut_info_states_signals = info.mutual_info_states_messages()

        ## Append the current mutual information
        self.statistics["mut_info_states_signals"] = np.append(
            self.statistics["mut_info_states_signals"], mut_info_states_signals
        )

        ## 2. Average probability of success
        if "avg_prob_success" not in self.statistics:
            ## Create empty array...
            self.statistics["avg_prob_success"] = np.empty((self.iteration,))

            ## ...and fill with NaN up to current iteration.
            self.statistics["avg_prob_success"][:] = np.nan

        ## Get the current average probability of success
        avg_prob_success = np.average(
            self.game.payoff(sender_strat=snorm, receiver_strat=rnorm)
        )

        ## Append the current probability of success
        self.statistics["avg_prob_success"] = np.append(
            self.statistics["avg_prob_success"], avg_prob_success
        )


class Agent:
    """
    Finite, discrete agent used in Reinforcement() objects.
    """

    def __init__(self, strategies):
        """
        Initialise an agent with a set of strategies.

        Parameters
        ----------
        strategies : array-like
            Array of strategies, where each strategy is itself an array.
        """

        ## Probability distribution over deterministic strategies.
        self.strategies = strategies

    def choose_strategy(self, input_data):
        """
        Sample from self.strategies[input_data] to get a concrete response.

        When the strategies are simply a matrix,
         with each row defining a distribution over possible responses,
         <input_data> is an integer indexing a row of the array.
        So we choose that row and choose randomly from it,
         according to the conditional probabilities of the responses,
         which are themselves listed as entries in each row.

        E.g. if this is a sender, <input_data> is the index of the current state of the world,
         and the possible responses are the possible signals.
        If this is a receiver, <input_data> is the index of the signal sent,
         and the possible responses are the possible acts.

        Parameters
        ----------
        input_data : int
            The state (if sender) or signal (if receiver) the agent just observed.

        Returns
        -------
        response : int.
         The index of the agent's response.

        """

        ## Among which responses can we choose?
        possible_responses = range(len(self.strategies[input_data]))

        ## Assume the strategies are not yet normalised
        probabilities = self.strategies[input_data] / self.strategies[input_data].sum()

        ## Select the response and return it
        return np.random.choice(possible_responses, p=probabilities)

    def update_strategies(self, input_data, response, payoff):
        """
        The agent has just played <response> in response to <input_data>,
         and received <payoff> as a result.

        They now update the probability of choosing that response for
         that input data, proportionally to <payoff>.

        Parameters
        ----------
        input_data : int
            The state (if sender) or signal (if receiver) the agent just observed.
        response : int
            The signal (if sender) or act (if receiver) the agent just performed.
        payoff : int
            The payoff the agent just received.

        Returns
        -------
        None.

        """

        ## Add payoff to probability.
        self.strategies[input_data][response] += payoff

        ## It's of course possible that the payoff is negative.
        ## We should not allow the weight to go below zero.
        self.strategies[input_data][response] = max(
            self.strategies[input_data][response], 0
        )

    def update_strategies_bush_mosteller(
        self, input_data, response, payoff, learning_parameter
    ):
        """
        Update strategies according to the Bush-Mosteller reinforcement rule.

        From Skyrms 2010 page 86:

        "If an act is chosen and a reward is gotten
         the probability is incremented by adding some fraction of the
         distance between the original probability and probability one.
         Alternative action probabilities are decremented so that everything
         adds to one. The fraction used is the product of the reward and
         some learning parameter."

        Parameters
        ----------
        input_data : int
            The state (if sender) or signal (if receiver) the agent just observed.
        response : int
            The signal (if sender) or act (if receiver) the agent just performed.
        payoff : int
            The payoff the agent just received.
        learning_parameter : float
            Learning parameter for Bush-Mosteller reinforcement.

        Returns
        -------
        None.

        """

        ## The row is self.strategies[input_data].
        ## The entry self.strategies[input_data][response] is incremented by
        ##  (1 - self.strategies[input_data][response]) * learning_parameter.
        ## All other entries are decremented (presumably relative to their current values).

        ## Create a "delta" array that says how everything changes.
        ## Imagine first we are decrementing everything.
        ar_delta = self.strategies[input_data] * learning_parameter * -1

        ## Now replace the winner's delta with its positive reinforcement rate.
        ar_delta[response] = (
            1 - self.strategies[input_data][response]
        ) * learning_parameter

        ## Now update the strategies
        self.strategies[input_data] += ar_delta

    @abstractmethod
    def add_signal(self):
        """
        Add a new signal to the player's repertoire.
        """

        raise NotImplementedError("This method must be implemented in a subclass.")

class Sender(Agent):
    """
    Agent type that observes a state of the world and sends a signal.
    """

    def add_signal(self)->None:
        """
        Add a signal to the sender's repertoire.

        Returns
        -------
        None.
        """

        # Define the new signal array as a column of ones
        new_signal_array = np.ones((len(self.strategies), 1))

        # Add the new signal to the sender's strategies
        self.strategies = np.append(self.strategies, new_signal_array, axis=1)

class Receiver(Agent):
    """
    Agent that receives a signal and performs an action.
    """

    def add_signal(self)->None:
        """
        Add a new signal to the receiver's repertoire.

        Returns
        -------
        None.
        """

        # Define the new signal array as a row of ones.
        new_signal_array = np.ones((1, len(self.strategies[0])))

        # Add the new signal to the receiver's strategies.
        self.strategies = np.append(self.strategies, new_signal_array, axis=0)


class Times:
    """
    Provides a way of having a single time input to both odeint and ode
    """

    def __init__(self, initial_time, final_time, time_inc):
        """
        Takes the initial time for simulations <initial_time>, the final time
        <final_time> and the time increment <time_inc>, and creates an object
        with these values as attributes, and also a vector that can be fed into
        odeint.

        Parameters
        ----------
        initial_time : float
            Initial time for simulations.
        final_time : float
            Final time for simulations.
        time_inc : float
            Time increment for simulations.
        """
        self.initial_time = initial_time
        self.final_time = final_time
        self.time_inc = time_inc
        points = int((final_time - initial_time) / time_inc)
        self.time_vector = np.linspace(initial_time, final_time, points + 1)


def mutationmatrix(mutation, dimension):
    """
    Calculate a (square) mutation matrix with mutation rate
    given by <mutation> and dimension given by <dimension>

    Parameters
    ----------
    mutation : float
        Mutation rate.
    dimension : int
        Dimension of the matrix.

    Returns
    -------
    array-like
        Mutation matrix.
    """
    return np.array(
        [
            [
                1 - mutation if i == j else mutation / (dimension - 1)
                for i in np.arange(dimension)
            ]
            for j in np.arange(dimension)
        ]
    )
