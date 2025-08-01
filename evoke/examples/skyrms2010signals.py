"""
Evoke library examples from:
    
    Skyrms, B. (2010). Signals: Evolution, Learning, and Information. Oxford University Press.
"""

import numpy as np

from tqdm import trange

from evoke.src.figure import Scatter, Bar, Quiver2D, Quiver3D, Ternary
from evoke.src import games
from evoke.src import evolve as ev
from evoke.src.logger import logger, Colors


class Skyrms2010_1_1(Quiver2D):
    """
    Figure 1.1, page 11, of Skyrms (2010).
    The figure depicts the replicator dynamics of a population
    repeatedly playing a two-player cooperative game
    where each agent in the population either always plays sender
    or always plays receiver.
    Senders observe one of two randomly chosen states of the world and produce one of two signals.
    Receivers observe the signal and produce one of two acts.
    If the act matches the state, both players gain a payoff.
    """

    def __init__(self):
        """
        This class creates a fixed figure.
        It does not allow input parameters.
        """

        self.initialize_simulation()

        evo = self.run_simulation()

        super().__init__(evo=evo)

        self.show()

    def initialize_simulation(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """

        self.state_chances = np.array([0.5, 0.5])
        self.sender_payoff_matrix = np.eye(2)
        self.receiver_payoff_matrix = np.eye(2)
        self.messages = 2

    def run_simulation(self):
        """
        Technically this figure does not require a simulation.
        It describes how a population of two types would evolve
        if it were playing a cooperative game.

        Returns
        -------
        evo : an instance of evolve.TwoPops.
            The evolutionary scenario is represented by an object from the module evolve.

        """

        ## Create the game
        lewis22 = games.Chance(
            self.state_chances,
            self.sender_payoff_matrix,
            self.receiver_payoff_matrix,
            self.messages,
        )

        ## Just get the pure strategies.
        sender_strats = lewis22.sender_pure_strats()[1:-1]
        receiver_strats = lewis22.receiver_pure_strats()[1:-1]

        ## Create the two-population simulation object
        self.evo = ev.TwoPops(lewis22, sender_strats, receiver_strats)

        ## We want our frequencies to be neatly spaced in a 15x15 grid
        ##  (in order to replicate Skyrms's figure)
        freqs = np.linspace(0.01, 0.99, 15)

        ## X is a matrix that's just 15 rows of <freqs>
        ## Y is a matrix that's just 15 columns of <freqs>
        self.X, self.Y = np.meshgrid(freqs, freqs)

        # This line is doing A LOT in a short space.
        # From the manual:
        #  "Define a vectorized function which takes a nested sequence of objects or
        #   numpy arrays as inputs and returns a single numpy array or a tuple of numpy arrays.
        #   The vectorized function evaluates pyfunc over successive tuples of the input arrays
        #   like the python map function, except it uses the broadcasting rules of numpy."
        # So the resultant function is something that takes a nested sequence of arrays as inputs,
        #  and returns a tuple of arrays.
        # X and Y are the "nested sequence of arrays" as input
        # U and V are the "tuple of arrays" as output.
        # But uv_from_xy() actually only accepts two SCALAR inputs!
        # So vectorize is just iterating over rows and columns.
        # It takes EVERY CELL-BY-CELL PAIR and feeds them into uv_from_xy().

        # Also, uv_from_xy() returns four things: change in proportion of senders of strategy 1,
        # change in proportion of receivers of strategy 1,
        # change in proportion of senders of strategy 2,
        # change in proportion of receivers of strategy 2.
        # We only want the last two, so we ignore the first two.

        # Furthermore, Skyrms plots the senders on the Y axis and the receivers on the X axis,
        # so we need to feed these into uv_from_xy() in the opposite order.
        # And U and V correspond to the X and Y axis, respectively.
        # So we need to map senders to V and receivers to U.
        _, _, self.V, self.U = np.vectorize(self.uv_from_xy)(self.Y, self.X)

        # Finally, we're telling uv_from_xy() what the proportions of the FIRST sender and receiver strategies are,
        # and reading off it what the change in proportions of the SECOND sender and receiver strategies are.
        # So we need to flip the X axis (to get the second receiver strategy)
        # and flip the Y axis (to get the second sender strategy).
        self.X = 1 - self.X
        self.Y = 1 - self.Y

        return self.evo


class Skyrms2010_1_2(Quiver3D):
    """
    Figure 1.2, page 12, of Skyrms (2010).
    The figure depicts the replicator dynamics of a population
    repeatedly playing a two-player cooperative game
    where each agent in the population can switch between playing sender or receiver.
    """

    def __init__(self):
        """
        This class creates a fixed figure.
        It does not allow input parameters.
        """

        self.initialize_simulation()

        evo = self.run_simulation()

        super().__init__(evo=evo, noaxis=True)

        self.show()

    def initialize_simulation(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """

        ## Create interaction payoff matrix
        self.payoffs = np.array(
            [
                [
                    1,
                    0.5,
                    0.5,
                    0,
                ],  # Payoffs received by type 1 when encountering types 1, 2, 3, 4
                [
                    0.5,
                    0,
                    1,
                    0.5,
                ],  # Payoffs received by type 2 when encountering types 1, 2, 3, 4
                [
                    0.5,
                    1,
                    0,
                    0.5,
                ],  # Payoffs received by type 3 when encountering types 1, 2, 3, 4
                [
                    0,
                    0.5,
                    0.5,
                    1,
                ],  # Payoffs received by type 4 when encountering types 1, 2, 3, 4
            ]
        )

        ## And the player types are all of the pure strategies,
        ##  which just means four vectors each saying "strategy 1", "strategy 2" etc
        self.playertypes = np.array(
            [
                [1, 0, 0, 0],  # "I'm playing the first strategy!"
                [0, 1, 0, 0],  # "I'm playing the second strategy!"
                [0, 0, 1, 0],  # "I'm playing the third strategy!"
                [0, 0, 0, 1],  # "I'm playing the fourth strategy!"
            ]
        )

        ## Tetrahedron vertices
        self.vertices = np.array(
            [
                [1, 1, 5],  # summit
                [0, 0, 0],  # bottom left
                [2, 2, 0],  # bottom right
                [1.75, 4.5, 0],  # back
            ]
        )

    def run_simulation(self):
        """
        Technically this figure does not require a simulation.
        It describes how a population of one type would evolve
        if it were playing a cooperative game.

        Returns
        -------
        evo : an instance of evolve.OnePop.
            The evolutionary scenario is represented by an object from the module evolve.

        """

        ## Create the game...
        game = games.NoSignal(self.payoffs)

        ## ...and the simulation.
        self.evo = ev.OnePop(game, self.playertypes)

        ## Create arrows at roughly the places Skyrms depicts them.
        pop_vectors = np.array(
            [
                [0.75, 0.25, 0, 0],  # left-upward
                [0.25, 0.75, 0, 0],  # left-upward
                # [0.75,0,0.25,0],
                # [0.25,0,0.75,0],
                [0, 0.75, 0.25, 0],  # front-inward (from left)
                [0, 0.25, 0.75, 0],  # front-inward (from right)
                [0.75, 0, 0, 0.25],
                [0.25, 0, 0, 0.75],
                [0, 0.25, 0, 0.75],
                [0, 0, 0.25, 0.75],
            ]
        )

        pop_vectors_new = self.evo.discrete_replicator_delta_X(pop_vectors[0])

        for i in range(1, len(pop_vectors)):
            pop_vectors_new = np.vstack(
                (pop_vectors_new, self.evo.discrete_replicator_delta_X(pop_vectors[i]))
            )

        ## Get barycentric coordinates for the original population positions
        pop_vector_bary = np.apply_along_axis(
            self.vector_to_barycentric, axis=1, arr=pop_vectors
        )

        ## Get barycentric coordinates for new population positions
        new_pop_vector_bary = np.apply_along_axis(
            self.vector_to_barycentric, axis=1, arr=pop_vectors_new
        )

        ## ... then get the difference between that and the current step.
        arrows = new_pop_vector_bary - pop_vector_bary

        ## Define big list of points and arrows
        # soa = np.hstack((vertices,vertices))

        soa = np.hstack((pop_vector_bary, arrows))

        self.X, self.Y, self.Z, self.U, self.V, self.W = zip(*soa)

        return self.evo


class Skyrms2010_3_3(Scatter):
    """
    Figure 3.3, page 40, of Skyrms (2010).
    The figure depicts the mutual information between signal and state
    over successive trials of a two-player cooperative game
    in which agents learn via reinforcement.
    Typically the mutual information will increase over time as the agents learn
    to use specific signals as indicators of specific states.
    However, the stochastic nature of the simulation means that the figure will look different
    each time it is run.
    """

    def __init__(self, iterations=100):
        """
        Run the simulation and create the figure, specifying the number of iterations
        i.e. the number of times the game is played.
        More iterations means a greater chance of positive reinforcement,
        leading to a greater chance of increased mutual information between signal and state.

        Parameters
        ----------
        iterations : int, optional
            Number of steps to run the game for. The default is 100.

        Returns
        -------
        None.

        """

        self.initialize_simulation()

        evo = self.run_simulation(iterations)

        ## Get info attribute
        y = evo.statistics["mut_info_states_signals"]

        super().__init__(evo=evo)

        self.reset(x=range(iterations), y=y, xlabel="Iterations", ylabel="Information")

        self.show()

    def initialize_simulation(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """

        self.state_chances = np.array([0.5, 0.5])
        self.sender_payoff_matrix = np.eye(2)
        self.receiver_payoff_matrix = np.eye(2)
        self.messages = 2

    def run_simulation(self, iterations):
        """
        Create a game object and an evolution object,
        and run the game <iterations> times.

        Parameters
        ----------
        iterations : int
            Number of timesteps in the simulation i.e. number of repetitions of the game.

        Returns
        -------
        evo : evolve.MatchingSR
            The evolve object controls simulations.

        """

        ## Create game
        game = games.Chance(
            state_chances=self.state_chances,
            sender_payoff_matrix=self.sender_payoff_matrix,
            receiver_payoff_matrix=self.receiver_payoff_matrix,
            messages=self.messages,
        )

        ## Define strategies
        sender_strategies = np.ones((2, 2))
        receiver_strategies = np.ones((2, 2))

        ## Create simulation
        evo = ev.MatchingSR(game, sender_strategies, receiver_strategies)

        ## Run simulation for <iterations> steps
        evo.run(iterations)

        return evo


class Skyrms2010_3_4(Scatter):
    """
    Figure 3.4, page 46, of Skyrms (2010).
    The figure depicts the change over time of the average probability of success
    in a cooperative signalling chain game.
    In this game there is a sender, an intermediary, and a receiver.
    There are two signals, between the sender and intermediary and between the
    intermediary and receiver.
    It takes a lot longer for signalling to become established in this game.
    The original figure uses one million iterations (1e6); however,
    the probability of success often reaches 1 after just one hundred thousand iterations (1e5).
    """

    def __init__(self, iterations=int(1e2)):
        """
        Create object.
        NB change iterations to int(1e6) to run the same number of iterations
        as in the original figure. Warning: this takes a while.
        You should still get convergence often at 1e5.

        Parameters
        ----------
        iterations : int, optional
            Number of timesteps in the simulation i.e. number of repetitions of the game.
            The default is 100.
        """

        self.initialize_simulation()

        evo = self.run_simulation(iterations)

        ## Get info attribute
        y = evo.statistics["prob_success"]

        # Create superclass
        super().__init__(evo=evo)

        # The data points only include the start point and each power of 10.
        self.reset(
            x=range(iterations),
            y=y,
            xlabel="Iterations",
            ylabel="Average Probability of Success",
            ylim=[0, 1],
            xscale="log",
        )

        self.show()

    def initialize_simulation(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """

        self.state_chances = np.array([0.5, 0.5])
        self.sender_payoff_matrix = np.eye(2)
        self.intermediary_payoff_matrix = np.eye(2)
        self.receiver_payoff_matrix = np.eye(2)
        self.messages_sender = 2
        self.messages_intermediary = 2

    def run_simulation(self, iterations):
        """
        Create a game object and an evolution object,
        and run the game <iterations> times.

        Parameters
        ----------
        iterations : int
            Number of timesteps in the simulation i.e. number of repetitions of the game.

        Returns
        -------
        evo : evolve.MatchingSIR
            The evolve object controls simulations.

        """

        ## Create game
        game = games.ChanceSIR(
            state_chances=self.state_chances,
            sender_payoff_matrix=self.sender_payoff_matrix,
            intermediary_payoff_matrix=self.intermediary_payoff_matrix,
            receiver_payoff_matrix=self.receiver_payoff_matrix,
            messages_sender=self.messages_sender,
            messages_intermediary=self.messages_intermediary,
        )

        ## Define initial strategies
        sender_strategies = np.ones((2, 2))
        intermediary_strategies = np.ones((2, 2))
        receiver_strategies = np.ones((2, 2))

        ## Create simulation
        evo = ev.MatchingSIR(
            game, sender_strategies, intermediary_strategies, receiver_strategies
        )

        ## Run simulation for <iterations> steps
        evo.run(iterations)

        return evo


class Skyrms2010_4_1(Ternary):
    """
    Figure 4.1, page 59, of Skyrms (2010).
    The figure depicts cycles in the replicator dynamics of a rock-paper-scissors game.
    """

    def __init__(self,colors=None):
        """
        This class creates a fixed figure.
        It only allows the colors of the orbits to be changed.

        Parameters
        ----------
        colors : list, optional
            List of colors for the orbits. If None, all orbits are black.
            The default is None.
        """

        self.initialize_simulation()

        evo = self.run_orbits()

        super().__init__(evo=evo)

        self.reset(
            right_corner_label  = "y", 
            top_corner_label    = "z",
            left_corner_label   = "x",
            fontsize            = 10,
            colors              = colors,
        )

        self.show()

    def initialize_simulation(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """

        self.initplayer1 = np.array([0.8, 0.1, 0.1])
        self.initplayer2 = np.array([0.6, 0.2, 0.2])
        self.initplayer3 = np.array([0.4, 0.3, 0.3])
        self.rps_payoff_matrix = np.array([[1, 2, 0], [0, 1, 2], [2, 0, 1]])

    def run_orbits(self):
        """
        Generate the cycles in the replicator dynamics.

        The cycles are stored as class attribute self.xyzs.
        This is used by the superclass Ternary in the method self.show().

        Returns
        -------
        evo : evolve.OnePop
            The evolve object for a one-population game.

        """

        # Create game
        game = games.NoSignal(self.rps_payoff_matrix)

        # Create evolve object
        evo = ev.OnePop(game, game.pure_strats())

        # Find the cycles
        self.xyzs = [
            evo.replicator_odeint(initplayer, np.linspace(0, 100, num=1000))
            for initplayer in [self.initplayer1, self.initplayer2, self.initplayer3]
        ]

        return evo


class Skyrms2010_5_2(Scatter):
    """
    Figure 5.2, page 72, of Skyrms (2010).
    The figure shows the value of assortment that is required to destabilise pooling
    in a 2x2x2 cooperative signalling game.
    *Assortment* is defined as the probability of meeting another player in the population
    who is the same type as you.
    *Pooling* is any strategy that produces the same signal for more than one state.

    Skyrms describes this model on page 71:

    "Here we consider a one-population model, in which
    nature assigns roles of sender or receiver on flip of a fair coin. We
    focus on four strategies, written as a vector whose components are:
    signal sent in state 1, signal sent in state 2, act done after signal 1, act
    done after signal 2.

    s1 = <1, 2, 1, 2>

    s2 = <2, 1, 2, 1>

    s3 = <1, 1, 2, 2>

    s4 = <2, 2, 2, 2>"

    """

    def __init__(self):
        """
        This class creates a fixed figure.
        It does not allow input parameters.
        """

        self.initialize_simulations()

        y_axis = self.run_simulations()

        ## Create superclass
        super().__init__()

        ## Set data for the graph.
        self.reset(
            x=self.pr_state_2_list,
            y=y_axis,
            xlabel="Pr State 2",
            ylabel="Value of Assortment to Destabilize Pooling e",
            xlim=[min(self.pr_state_2_list), max(self.pr_state_2_list)],
            ylim=[0, 1],
        )

        # Show the line along with the datapoints
        self.show_line = True

        # Show the figure
        self.show()

    def initialize_simulations(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """

        self.pr_state_2_list = np.linspace(0.5, 1, 10)
        self.sender_payoff_matrix = np.eye(2)
        self.receiver_payoff_matrix = np.eye(2)
        self.messages = 2

    def run_simulations(self):
        """
        Create games and run simulations.

        The vectors s1-s4 defined on page 71 of Skyrms (2010)
        define the playertypes payoffs.

        Returns
        -------
        y_axis : list
            Results; the required assortment level for each value of pr_state_2.
        """

        y_axis = []

        for pr_state_2 in self.pr_state_2_list:
            ## Define payoffs based on pr_state_2.
            ## Here payoffs is an nxn matrix, with n the number of players (here 4).
            ## Entry (i,j) gives the payoff of player i upon meeting player j.
            payoffs = np.array(
                [
                    [
                        1,
                        0,
                        (0.5 * pr_state_2 + 0.5 * (1 - pr_state_2)),
                        (0.5 * pr_state_2 + 0.5 * pr_state_2),
                    ],
                    [
                        0,
                        1,
                        (0.5 * pr_state_2 + 0.5 * pr_state_2),
                        (0.5 * pr_state_2 + 0.5 * (1 - pr_state_2)),
                    ],
                    [
                        (0.5 * (1 - pr_state_2) + 0.5 * pr_state_2),
                        0.5 * pr_state_2 + 0.5 * pr_state_2,
                        pr_state_2,
                        pr_state_2,
                    ],
                    [
                        (0.5 * pr_state_2 + 0.5 * pr_state_2),
                        (0.5 * (1 - pr_state_2) + 0.5 * pr_state_2),
                        pr_state_2,
                        pr_state_2,
                    ],
                ]
            )

            ## We pretend it's a straight encounter game, because we already calculated the payoffs.
            game = games.NoSignal(payoffs)

            ## Playertypes are 50/50 types 3 and 4
            playertypes = np.array(
                [
                    [1, 0, 0, 0],  # "I'm playing the first strategy!"
                    [0, 1, 0, 0],  # "I'm playing the second strategy!"
                    [0, 0, 1, 0],  # "I'm playing the third strategy!"
                    [0, 0, 0, 1],  # "I'm playing the fourth strategy!"
                ]
            )

            ## Create simulation
            evo = ev.OnePop(game, playertypes)

            ## Pooling equilibria are destabilized when the expected payoffs of type 1 and/or type 2
            ##  become equal to that of types 3 and 4.
            ## And presumably you get that just by multiplying the strategy profile by the payoff vector
            ##  for each player.
            ## Well, that works when there is no assortment.
            ## When there is assortment, you have to use Wright's special rules to get the probabilities
            ##  of meeting each type (See Skyrms 2010 page 71).

            ## Loop at assortment levels until you find the one at which
            ##  the expected payoffs of 1 and 2 are >= the expected payoffs of 3 and 4.
            assortment_level = 0
            for e in np.arange(0, 1, 0.001):
                assortment_level = e

                evo.assortment(e)

                payoff_vector = evo.avg_payoff_vector(np.array([0, 0, 0.5, 0.5]))

                ## Check whether types 1 or 2 beat types 3 or 4
                if (
                    payoff_vector[0] >= payoff_vector[2]
                    or payoff_vector[0] >= payoff_vector[3]
                    or payoff_vector[1] >= payoff_vector[2]
                    or payoff_vector[1] >= payoff_vector[3]
                ):
                    break

            y_axis.append(assortment_level)

        return y_axis


class Skyrms2010_8_1(Scatter):
    """
    Figure 8.1, page 95, of Skyrms (2010).
    The figure depicts the change over time of the probability of success
    in a two-player cooperative game where the agents learn by reinforcement.
    """

    def __init__(self, iterations=int(1e3)):
        """
        Create object.

        Parameters
        ----------
        iterations : int, optional
            Number of timesteps in the simulation. The default is int(1e3).

        """
        self.initialize_simulation()

        evo = self.run_simulation(iterations)

        ## Get info attribute
        y = evo.statistics["avg_prob_success"]

        super().__init__(evo=evo)

        # Set figure parameters
        self.reset(
            x=range(iterations),
            y=y,
            xlabel="Iterations",
            ylabel="Average Probability of Success",
            ylim=[0, 1],
            marker_size=5,
        )

        # Show the line along with the datapoints
        self.show_line = True

        # Show the figure
        self.show()

    def initialize_simulation(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """

        self.state_chances = np.array([0.5, 0.5])
        self.sender_payoff_matrix = np.eye(2)
        self.receiver_payoff_matrix = np.eye(2)
        self.messages = 2

    def run_simulation(self, iterations):
        """
        Create game and run simulation.

        Parameters
        ----------
        iterations : int
            Number of timesteps.

        Returns
        -------
        evo : evolve.MatchingSR
            The simulation object.

        """
        ## Create game
        game = games.Chance(
            state_chances=self.state_chances,
            sender_payoff_matrix=self.sender_payoff_matrix,
            receiver_payoff_matrix=self.receiver_payoff_matrix,
            messages=self.messages,
        )

        ## Define strategies
        ## These are the initial weights for Roth-Erev reinforcement.
        sender_strategies = np.ones((2, 2))
        receiver_strategies = np.ones((2, 2))

        ## Create simulation
        evo = ev.MatchingSR(game, sender_strategies, receiver_strategies)

        ## Run simulation for <iterations> steps
        evo.run(iterations)

        return evo


class Skyrms2010_8_2(Scatter):
    """
    Figure 8.2, page 97, of Skyrms (2010).
    The figure depicts the probability of pooling in a signalling game with
    reinforcement learning for different initial weights.
    Initial weights determine how difficult it is to learn something new:
    large initial weights mean that learning is slower.

    Skyrms does not explicitly state the number of trials or number of iterations
    used to generate his figure.
    We suspect the parameter values are something like
    `trials=int(1e3)` and `iterations=int(1e5)`.
    However, attempting to generate this figure with those values
    will take an exceptionally long time.

    Even with `iterations=int(1e4)`, it's looking like 12 minutes per weight,
    so about an hour overall.

    This, combined with the difficulty of figuring out exactly how Skyrms is
    identifying pooling equilibria, leads to us overestimating
    the probability of pooling.
    You are warned!
    """

    def __init__(self, trials=100, iterations=int(1e3), pooling_epsilon=1e-3):
        """
        Create object.

        Parameters
        ----------
        trials : int, optional
            Number of times to repeat a simulation with specific parameters.
            The default is 100.
        iterations : int, optional
            Number of timesteps in each simulation.
            The default is int(1e3).
        pooling_epsilon : float, optional
            How close to the maximum possible mutual information must
            the current mutual information be in order to count
            as a signalling system?
            See evolve.Reinforcement.is_pooling() for more information.
            The default is 1e-3.

        """

        # Set object properties
        self.pooling_epsilon = pooling_epsilon

        # Set parameters for simulations
        self.initialize_simulation()

        # Run the simluations
        self.run_simulation(trials, iterations)

        # Create the figure superclass
        super().__init__()

        # Set parameters for the figure
        self.reset(
            x=self.initial_weights,
            y=self.probability_of_pooling,
            xlabel="Initial Weights",
            ylabel="Probability of Pooling",
            ylim=[0, 1],
            marker_size=5,
            xscale="log",
        )
        
        # Show the line along with the datapoints
        self.show_line = True

        # Show the figure
        self.show()

    def initialize_simulation(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """
        self.state_chances = np.array([0.9, 0.1])
        self.sender_payoff_matrix = np.eye(2)
        self.receiver_payoff_matrix = np.eye(2)
        self.messages = 2

        self.initial_weights = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])

    def run_simulation(self, trials, iterations) -> None:
        """
        Create game and run simulations.

        Parameters
        ----------
        trials : int, optional
            Number of times to repeat a simulation with specific parameters.
            The default is 100.
        iterations : int, optional
            Number of timesteps in each simulation.
            The default is int(1e3).
        """
        ## Create game
        game = games.Chance(
            state_chances=self.state_chances,
            sender_payoff_matrix=self.sender_payoff_matrix,
            receiver_payoff_matrix=self.receiver_payoff_matrix,
            messages=self.messages,
        )

        self.probability_of_pooling = []

        for initial_weight in self.initial_weights:
            ## This could really take a long time, so print a report and tqdm progress bar.
            logger.info(
                f"Initial weight {initial_weight}. Simulating {trials} games with {iterations} iterations each..."
            )

            count_pooling = 0

            for trial in trange(trials):
                ## Define strategies
                ## These are the initial weights for Roth-Erev reinforcement.
                sender_strategies = np.ones((2, 2)) * initial_weight
                receiver_strategies = np.ones((2, 2)) * initial_weight

                ## Create simulation
                evo = ev.MatchingSR(game, sender_strategies, receiver_strategies)

                ## Run simulation for <iterations> steps
                ## Tell it to only calculate at end
                evo.run(iterations, calculate_stats="end")

                ## NB the is_pooling() method has a parameter <epsilon>.
                ## To make the figure look most similar to Skyrms's results,
                ##  epsilon should be sensitive to the number of iterations.
                ## That's because the more iterations the simulation runs for,
                ##  the easier it is to reach a signalling system
                ##  (where "reaching a signalling system" is determined in part by
                ##   the value of epsilon).
                ## So if your figure doesn't look sufficiently similar to the original,
                ##  the relative values of <epsilon> and <iterations> might be
                ##  a good place to start.
                if evo.is_pooling(epsilon=self.pooling_epsilon):
                    count_pooling += 1

            ## For each initial weight, get the proportion of evo games (out of 1000)
            ##        that led to partial pooling.
            self.probability_of_pooling.append(count_pooling / trials)


class Skyrms2010_8_3(Scatter):
    """
    Figure 8.3, page 98, of Skyrms (2010).
    The figure depicts the probability of signalling for different
    values of the learning parameter in a Bush–Mosteller reinforcement scenario.

    Our recreation of this figure is clearly undercounting signalling.
    That's because we are defining signalling as "not pooling",
    and we are overcounting pooling (see the docstring for class Skyrms2010_8_2).

    In future, we need to try and count both pooling and signaling
    more accurately; this is difficult, since we don't know exactly
    how Skyrms defines them for the purposes of his figures.
    """

    def __init__(
        self,
        trials=100,
        iterations=300,
        learning_params=[0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        pooling_epsilon=1e-3,
    ):
        """
        Create object.

        Not sure how many trials Skyrms is reporting -- probably at least 1000.
        He explicitly states 300 iterations on p.98.

        Parameters
        ----------
        trials : int, optional
            Number of times to repeat each simulation.
            The default is 100.
        iterations : int, optional
            Number of timesteps per simulation.
            The default is 300.
        learning_params : array-like, optional
            Learning parameters to run simulations for.
            The default is [0.01,0.03,0.05,0.07,0.1,0.15,0.2].
        pooling_epsilon : float, optional
            How close to the maximum possible mutual information must
            the current mutual information be in order to count
            as a signalling system?
            See evolve.Reinforcement.is_pooling() for more information.
            The default is 1e-3.

        Returns
        -------
        None.

        """

        # Set object properties
        self.pooling_epsilon = pooling_epsilon

        # Set simulation parameters
        self.initialize_simulation(learning_params)

        # Run the simulations
        self.run_simulation(trials, iterations)

        ## Set graph data and display parameters
        self.reset(
            x=np.array(self.learning_params).astype(str),
            y=self.prob_of_signalling,
            xlabel="Learning Parameter",
            ylabel="Signalling",
            ylim=[0, 1],
            marker_size=5,
        )

        # Create superclass
        super().__init__()

        # Ensure the line is shown
        self.show_line = True

        # Show the figure
        self.show()

    def initialize_simulation(self, learning_params) -> None:
        """
        Sets the figure parameters as class attributes.

        Parameters
        ----------
        learning_params : array-like, optional
            Learning parameters to run simulations for.
        """

        ## Fixed parameters
        self.state_chances = np.array([0.9, 0.1])
        self.sender_payoff_matrix = np.eye(2)
        self.receiver_payoff_matrix = np.eye(2)
        self.messages = 2

        ## User-supplied parameters
        self.learning_params = learning_params

    def run_simulation(self, trials, iterations) -> None:
        """
        Create game and run simulations.

        Parameters
        ----------
        trials : int, optional
            Number of times to repeat each simulation.
        iterations : int, optional
            Number of timesteps per simulation.

        Returns
        -------
        evo : evolve.BushMostellerSR
            Simulation object.

        """

        self.prob_of_signalling = []

        ## Create game
        game = games.Chance(
            state_chances=self.state_chances,
            sender_payoff_matrix=self.sender_payoff_matrix,
            receiver_payoff_matrix=self.receiver_payoff_matrix,
            messages=self.messages,
        )

        for learning_param in self.learning_params:
            ## This could really take a long time, so print a report and tqdm progress bar.
            logger.info(
                f"Learning parameter: {learning_param}. Simulating {trials} games with {iterations} iterations each..."
            )

            count_signalling = 0

            for trial in trange(trials):
                ## Define strategies
                ## These are the initial weights for Bush-Mosteller reinforcement.
                sender_strategies = (
                    np.ones((2, 2)) * 0.5
                )  # Strategies are now conditional prob distributions on each row
                receiver_strategies = (
                    np.ones((2, 2)) * 0.5
                )  # Strategies are now conditional prob distributions on each row

                ## Create simulation
                evo = ev.BushMostellerSR(
                    game, sender_strategies, receiver_strategies, learning_param
                )

                ## Run simulation for <iterations> steps
                ## Tell it to only calculate at end
                evo.run(iterations, calculate_stats="end")

                # Because we don't know Skyrms's pooling threshold cutoff,
                # this attribute can miscount (e.g. overcount pooling and undercount signalling).
                if not evo.is_pooling(epsilon=self.pooling_epsilon):
                    count_signalling += 1

            ## For each initial weight, get the proportion of evo games (out of 1000)
            ##        that led to partial pooling.
            self.prob_of_signalling.append(count_signalling / trials)


class Skyrms2010_10_5(Bar):
    """
    Figure 10.5, page 130, of Skyrms (2010).
    The figure depicts the number of signals at the end of reinforcement
    for a cooperative game in which senders can invent new signals.

    NB Skyrms uses trials=1000 and iterations=int(1e5) but this will take a very long time.
    """

    def __init__(self, trials=1000, iterations=int(1e4)):
        """
        NB Skyrms uses trials=1000 and iterations=int(1e5) but this will take a very long time.

        Parameters
        ----------
        trials : int, optional
            The number of simulations to run.
            The default is 1000.

        iterations : int
            The number of iterations per trial.
            The default is int(1e4).
        """

        self.initialize_simulation()

        self.run_simulation(trials, iterations)

        ## Get graph info
        results_as_array = np.array(sorted(self.signal_frequencies.items()))
        x_axis = results_as_array.T[0]
        y_axis = results_as_array.T[1]

        ## Y-axis limits
        ylim = [0, max(y_axis) + 0.1 * max(y_axis)]

        # Create superclass
        super().__init__()

        self.reset(
            x=x_axis,
            y=y_axis,
            xlabel="Number of signals",
            ylabel="Frequency",
            ylim=ylim,
        )

        self.show()

    def initialize_simulation(self) -> None:
        """
        Sets the figure parameters as class attributes.
        """

        self.state_chances = np.array([1 / 3, 1 / 3, 1 / 3])
        self.sender_payoff_matrix = np.eye(3)
        self.receiver_payoff_matrix = np.eye(3)
        self.messages = 1  # Skyrms says zero, but the first message is a "phantom".

    def run_simulation(self, trials, iterations) -> None:
        """
        Run <trials> trials with <iterations> iterations each.

        Parameters
        ----------
        trials : int
            Number of simulations.
        iterations : int
            Number of iterations per trial.
        """

        ## Initialise data dict: keys will be x-axis, values will be y-axis.
        self.signal_frequencies = {s: 0 for s in range(1, 30)}

        ## Create game
        game = games.Chance(
            state_chances=self.state_chances,
            sender_payoff_matrix=self.sender_payoff_matrix,
            receiver_payoff_matrix=self.receiver_payoff_matrix,
            messages=self.messages,
        )

        for trial in trange(trials):
            ## Define strategies
            ## Players start with only 1 "phantom" signal available.
            sender_strategies = np.ones((3, 1))
            receiver_strategies = np.ones((1, 3))

            ## Create simulation
            evo = ev.MatchingSRInvention(game, sender_strategies, receiver_strategies)

            ## Run simulation for <iterations> steps
            ## Tell it to only calculate at end
            evo.run(iterations, calculate_stats="end")

            n_signals_int = int(evo.statistics["number_of_signals"][-1])

            ## Add the number of signals to the log
            if n_signals_int in self.signal_frequencies:
                self.signal_frequencies[n_signals_int] += 1
            else:
                self.signal_frequencies[n_signals_int] = 1
