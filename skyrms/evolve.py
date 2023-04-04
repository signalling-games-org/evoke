"""
Calculate equations to evolve populations in a game. Right now, we can
calculate the replicator (-mutator) dynamics, with one or two populations, in
discrete or continuous time
"""

## Built-in
import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
from abc import ABC, abstractmethod

## Libraries
from tqdm import trange

## Skyrms
from skyrms.asymmetric_games import Chance, NonChance, ChanceSIR
from skyrms.info import Information


class OnePop:
    """
    Calculate the equations necessary to evolve one population. It takes as
    input a <game> and an array such that the <i,j> cell gives the expected
    payoff for the-strategist player of an encounter in which they  follow
    strategy i and their opponent follows strategy j.
    """

    def __init__(self, game, playertypes):
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
        """
        return np.random.dirichlet([1 for i in np.arange(self.lps)])

    def avg_payoff(self, player):
        """
        Return the average payoff that players get when the population vector
        is <player>
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
        player : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        ## Use the simple version if there is no assortment.
        if self.e == 0: return player @ self.avgpayoffs
        
        avg_payoffs = []
        
        ## Otherwise, loop and specify the assortment-weighted probabilities.
        ## This can probably be vectorised.
        i = 0
        for p_i in player:
            
            meeting_probabilities = []
            
            j = 0
            for p_j in player:
                
                if i == j:
                    ## p_i is the proportion of individuals of type i.
                    meeting_probabilities.append(p_i + self.e * (1-p_i))
                
                else:
                    ## p_j are all the others
                    meeting_probabilities.append(p_j - self.e*p_j)
                
                j+= 1
            
            meeting_probabilities = np.array(meeting_probabilities)
            
            payoff_i = (meeting_probabilities @ self.avgpayoffs)[i]
            
            avg_payoffs.append(payoff_i)
            
            i += 1
        
        return avg_payoffs

    def replicator_dX_dt_odeint(self, X, t):
        """
        Calculate the rhs of the system of odes for scipy.integrate.odeint
        """
        avgfitplayer = self.avg_payoff(X)
        result = self.avgpayoffs.dot(X) - X * avgfitplayer
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

    def replicator_dX_dt_ode(self, t, X):
        """
        Calculate the rhs of the system of odes for scipy.integrate.ode
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
        """
        avgfitness = self.player_avg_payoff(X)
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
        """
        avgfitness = self.avg_payoff(X)
        delta = X * (self.avgpayoffs @ X) / avgfitness
        if self.precision:
            np.around(delta, decimals=self.precision, out=delta)
        return delta

    def replicator_odeint(self, sinit, rinit, times, **kwargs):
        """
        Calculate one run of the game following the replicator(-mutator)
        dynamics, with starting points sinit and rinit, in times <times> (a
        game.Times instance), using scipy.integrate.odeint
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
        dynamics in continuous time, in <times> (a game.Times instance) with
        starting points sinit and rinit using scipy.integrate.ode
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

    def replicator_discrete(self, initpop, times):
        """
        Calculate one run of the game, following the discrete
        replicator(-mutator) dynamics, in <times> (a game.Times object) with
        starting population vector <initpop> using the discrete time
        replicator dynamics. Note that this solver will just calculate n points
        in the evolution of the population, and will not try to match them to
        the times as provided.
        """
        data = np.empty([len(times.time_vector), len(initpop)])
        data[0, :] = initpop
        for i in range(1, len(times.time_vector)):
            data[i, :] = self.discrete_replicator_delta_X(data[i - 1, :])
        return data

    def pop_to_mixed_strat(self, pop):
        """
        Take a population vector and output the average strat implemented by
        the whole population
        """
        return self.game.calculate_mixed_strat(self.playertypes, pop)
    
    def assortment(self,e):
        """
        Set assortment level

        Parameters
        ----------
        e : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.e = e


class TwoPops:
    """
    Calculate the equations necessary to evolve a population of senders and one
    of receivers. It takes as input a <game>, (which as of now only can be a
    Chance object>, and a tuple: the first (second) member of the tuple is a
    nxm array such that the <i,j> cell gives the expected payoff for the sender
    (receiver) of an encounter in which the sender follows strategy i and the
    receiver follows strategy j.
    """

    def __init__(self, game, sendertypes, receivertypes):
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
        """
        return np.random.dirichlet([1 for i in np.arange(self.lss)])

    def random_receiver(self):
        """
        Return frequencies of a random receiver population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lrs)])

    def sender_avg_payoff(self, sender, receiver):
        """
        Return the average payoff that senders get when the population vectors
        are <sender> and <receiver>
        """
        return sender.dot(self.senderpayoffs.dot(receiver))

    def receiver_avg_payoff(self, receiver, sender):
        """
        Return the average payoff that receivers get when the population
        vectors are <sender> and <receiver>
        """
        return receiver.dot(self.receiverpayoffs.dot(sender))

    def replicator_dX_dt_odeint(self, X, t):
        """
        Calculate the rhs of the system of odes for scipy.integrate.odeint
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
        dynamics, with starting points sinit and rinit, in times <times> (a
        game.Times instance), using scipy.integrate.odeint
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
        dynamics in continuous time, in <times> (a game.Times instance) with
        starting points sinit and rinit using scipy.integrate.ode
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
        replicator(-mutator) dynamics, in <times> (a game.Times object) with
        starting population vector <popvector> using the discrete time
        replicator dynamics. Note that this solver will just calculate n points
        in the evolution of the population, and will not try to match them to
        the times as provided.
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
        """
        return np.hsplit(vector, [self.lss])

    def sender_to_mixed_strat(self, senderpop):
        """
        Take a sender population vector and output the average
        sender strat implemented by the whole population
        """
        return self.game.calculate_sender_mixed_strat(self.sendertypes, senderpop)

    def receiver_to_mixed_strat(self, receiverpop):
        """
        Take a receiver population vector and output the average
        receiver strat implemented by the whole population
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
        game : one of the Skyrms game objects
            DESCRIPTION.
        agents: array-like
            list of Skyrms agent objects

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

    def run(self, iterations):
        """
        Run the simulation for <iterations> steps.

        Parameters
        ----------
        iterations : int
            Number of times to call self.step().

        Returns
        -------
        None.

        """

        for _ in trange(iterations):
            self.step()

    @abstractmethod
    def step(self):
        pass


class Matching(Reinforcement):
    """
    Reinforcement according to Richard Herrnstein’s matching law.
    The probability of choosing an action is proportional to its accumulated rewards.
    """

    def __init__(self, game, agents):

        super().__init__(game=game, agents=agents)


class MatchingSR(Matching):
    """
    Matching simulation for two-player sender-receiver game.
    """

    def __init__(self, game, sender_strategies, receiver_strategies):
        """


        Parameters
        ----------
        game : TYPE
            DESCRIPTION.
        sender_strategies : TYPE
            DESCRIPTION.
        receiver_strategies : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.game = game

        ## Create the agents.
        ## Sender
        self.sender = Agent(sender_strategies)

        ## Receiver
        self.receiver = Agent(receiver_strategies)

        super().__init__(game=game, agents=[self.sender, self.receiver])

    def step(self):
        """
        Implement the matching rule and increment one step.

        In each step:
            1. Run one round of the game.
            2. Update the agents' strategies based on the payoffs they received.
            3. Calculate and store any required variables e.g. information.
            4. Update iteration.

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
        self.log_info()

        ## 4. Update iteration.
        self.iteration += 1

    def log_info(self):
        """
        Calculate and store informational quantities at this point.

        Returns
        -------
        None.

        """

        ## Lazy instantiation
        if not hasattr(self, "statistics"):
            self.statistics = {}

        ## Mutual information between states and signals
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


class MatchingSIR(Matching):
    """
    Reinforcement game for sender, intermediary, receiver.
    """

    def __init__(
        self, game, sender_strategies, intermediary_strategies, receiver_strategies
    ):
        """


        Parameters
        ----------
        game : TYPE
            DESCRIPTION.
        sender_strategies : TYPE
            DESCRIPTION.
        intermediary_strategies : TYPE
            DESCRIPTION.
        receiver_strategies : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.game = game

        ## Create the agents.
        ## Sender
        self.sender = Agent(sender_strategies)

        ## Intermediary
        self.intermediary = Agent(intermediary_strategies)

        ## Receiver
        self.receiver = Agent(receiver_strategies)

        super().__init__(
            game=game, agents=[self.sender, self.intermediary, self.receiver]
        )

    def step(self):
        """
        Implement the matching rule and increment one step.

        In each step:
            1. Run one round of the game.
            2. Update the agents' strategies based on the payoffs they received.
            3. Calculate and store any required variables e.g. probability of success.
            4. Update iteration.

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
        self.record_probability_of_success()

        ## 4. Update iteration.
        self.iteration += 1

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
        payoff = self.game.avg_payoffs_regular(snorm,inorm,rnorm)
        
        ## Append the current payoff
        self.statistics["prob_success"] = np.append(
            self.statistics["prob_success"], payoff
        )


class Agent:
    """
    Finite, discrete agent used in Reinforcement() objects.
    """

    def __init__(self, strategies):

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
        input_data : TYPE
            DESCRIPTION.
        response : TYPE
            DESCRIPTION.
        payoff : TYPE
            DESCRIPTION.

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
        """
        self.initial_time = initial_time
        self.final_time = final_time
        self.time_inc = time_inc
        points = (final_time - initial_time) / time_inc
        self.time_vector = np.linspace(initial_time, final_time, points)


def mutationmatrix(mutation, dimension):
    """
    Calculate a (square) mutation matrix with mutation rate
    given by <mutation> and dimension given by <dimension>
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
