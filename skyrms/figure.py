# -*- coding: utf-8 -*-
"""
Figure objects.
"""

import numpy as np
from itertools import combinations
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
## 3D plotting
from mpl_toolkits.mplot3d import Axes3D


from skyrms import asymmetric_games as asy
from skyrms.evolve import OnePop,TwoPops,MatchingSR
from skyrms.symmetric_games import NoSignal


class Figure(ABC):
    """
        Abstract superclass for all figures.
    """
    
    def __init__(self,
            evo, # Evolve object
            **kwargs
             ):
        
        ## Set the evolve object as a class attribute
        self.evo = evo
        
        ## Set keyword arguments as class attributes.
        for k, v in kwargs.items(): setattr(self, k, v)
        
        ## Do we show the plot immediately?
        if hasattr(self,'show_immediately') and self.show_immediately: self.show()
    
    @abstractmethod
    def show(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass


class Scatter(Figure):
    """
        Superclass for scatter plots
    """
    
    def __init__(
            self,
            evo,
            **kwargs
            ):
        
        super().__init__(
            evo=evo,
            **kwargs)
        
    def reset(
            self,
            x,
            y,
            xlabel,
            ylabel,
            marker_size = 10,
            marker_color = 'k'
            ):
        """
        Update figure parameters

        Parameters
        ----------
        x : array-like
            x-axis coordinates.
        y : array-like
            y-axis coordinates.

        Returns
        -------
        None.

        """
        
        ## Update global attributes, which can then be plotted in self.show()
        self.x = x
        self.y = y
        
        ## Labels
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        ## Marker design
        self.s = marker_size
        self.c = marker_color
        
    
    def show(self):
        
        ## Data
        plt.scatter(
            x = self.x, 
            y = self.y,
            s = self.s,
            c = self.c
            )
        
        ## Labels
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        
        ## Show plot
        plt.show()
        
    
    """
        SCATTER ATTRIBUTES AND ALIASES
    """
    """
        Marker size
    """
    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, inp):
        self._s = inp

    @s.deleter
    def s(self):
        del self._s

    # Alias
    marker_size = s
    
    """
        Marker color
    """
    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, inp):
        self._c = inp

    @c.deleter
    def c(self):
        del self._c

    # Alias
    marker_color = c
        
    
class Skyrms2010_3_3(Scatter):
    """
        Figure 3.3, page 40, of Skyrms 2010
    """
    
    def __init__(self,iterations=100):
        
        self.initialize_simulation()
        
        evo = self.run_simulation(iterations)
        
        ## Get info attribute
        y = evo.statistics["mut_info_states_signals"]
        
        super().__init__(evo)
        
        self.reset(
            x = range(iterations),
            y = y,
            xlabel = "Iterations",
            ylabel = "Information"
            )
        
        self.show()
    
    def initialize_simulation(self):
        
        self.state_chances           = np.array([.5, .5])
        self.sender_payoff_matrix    = np.eye(2)
        self.receiver_payoff_matrix  = np.eye(2)
        self.messages                = 2
        
    
    def run_simulation(self,iterations):
        
        ## Create game
        game = asy.Chance(
            state_chances = self.state_chances,
            sender_payoff_matrix = self.sender_payoff_matrix,
            receiver_payoff_matrix = self.receiver_payoff_matrix,
            messages = self.messages
            )
        
        ## Define strategies
        sender_strategies = np.ones((2,2))
        receiver_strategies = np.ones((2,2))
        
        
        ## Create simulation
        evo = MatchingSR(
            game,
            sender_strategies,
            receiver_strategies
            )
        
        ## Run simulation for <iterations> steps
        evo.run(iterations)
        
        return evo


class Quiver(Figure):
    """
        Superclass for Quiver plots
    """
    
    def __init__(
            self,
            evo,
            **kwargs
            ):
        
        super().__init__(
            evo=evo,
            **kwargs)
    


class Quiver2D(Quiver):
    """
        Plot a 2D quiver plot.
    """
    
    def __init__(
            self,
            evo,
            scale = 20,
            **kwargs
            ):
        
        self.scale = scale
        
        super().__init__(
            evo = evo,
            **kwargs)
    
    
    def show(self):
        
        ## Create the figure
        fig, ax = plt.subplots()
        
        ## Create the quiver plot as a function of:
        ## X: x-coordinates of arrows
        ## Y: y-coordinates of arrows
        ## U: velocities of arrows in x direction
        ## V: velocities of arrows in y direction
        
        self.q = ax.quiver(
            self.X, 
            self.Y, 
            self.U, 
            self.V, 
            scale   = self.scale, 
            )
        
        plt.show()
    
    def uv_from_xy(self, x, y):
        """
        Parameters
        ----------
        x : float
            Current proportion of the first sender strategy.
        y : float
            Current proportion of the first receiver strategy.
        
        Returns
        velocity of SECOND sender strategy.
        velocity of SECOND receiver strategy.
        """
        
        ## TODO we're currently accepting the proportion of the first strategy
        ##  and returning the velocities of the second.
        ## Rewrite both this and the calling method to supply, accept and return the SECOND strategy only.
        senders = np.array([x, 1 - x])
        receivers = np.array([y, 1 - y])
        
        new_pop_vector = self.evo.discrete_replicator_delta_X(np.concatenate((senders, receivers)))
        new_senders, new_receivers = self.evo.vector_to_populations(new_pop_vector)
        return (1-x) - new_senders[1], (1-y) - new_receivers[1]
        
        

class Skyrms2010_1_1(Quiver2D):
    """
        Run the Skyrms (2010) Figure 1 simulation and plot.
    """
    
    def __init__(self):
        
        self.initialize_simulation()
        
        evo = self.run_simulation()
        
        super().__init__(
            evo=evo,
            show_immediately=True)
    
    def initialize_simulation(self):
        
        self.states                  = np.array([.5, .5])
        self.sender_payoff_matrix    = np.eye(2)
        self.receiver_payoff_matrix  = np.eye(2)
        self.messages                = 2
    
    def run_simulation(self):
        
        ## Create the game
        lewis22 = asy.Chance(
            self.states, 
            self.sender_payoff_matrix, 
            self.receiver_payoff_matrix, 
            self.messages
            )
        
        ## Just get the pure strategies.
        sender_strats = lewis22.sender_pure_strats()[1:-1]
        receiver_strats = lewis22.receiver_pure_strats()[1:-1]
        
        ## Create the two-population simulation object
        self.evo = TwoPops(lewis22, sender_strats, receiver_strats)
        
        ## We want our frequencies to be neatly spaced in a 15x15 grid 
        ##  (in order to replicate Skyrms's figure)
        freqs = np.linspace(0.01, .99, 15)
        
        ## X is a matrix that's just 15 rows of <freqs>
        ## Y is a matrix that's just 15 columns of <freqs>
        self.X, self.Y = np.meshgrid(freqs, freqs)
        
        ## This line is doing A LOT in a short space.
        ## From the manual:
        ##  "Define a vectorized function which takes a nested sequence of objects or 
        ##   numpy arrays as inputs and returns a single numpy array or a tuple of numpy arrays. 
        ##   The vectorized function evaluates pyfunc over successive tuples of the input arrays 
        ##   like the python map function, except it uses the broadcasting rules of numpy."
        ## So the resultant function is something that takes a nested sequence of arrays as inputs,
        ##  and returns a tuple of arrays.
        ## X and Y are the "nested sequence of arrays" as input
        ## U and V are the "tuple of arrays" as output.
        ## But uv_from_xy() actually only accepts two SCALAR input!
        ## So vectorize is just iterating over rows and columns.
        ## It takes EVERY CELL-BY-CELL PAIR and feeds them into uv_from_xy().
        self.U, self.V = np.vectorize(self.uv_from_xy)(self.X, self.Y)
        
        return self.evo


class Quiver3D(Quiver):
    """
        Plot a 3D quiver plot.
    """
    
    def __init__(
            self,
            evo,
            color='k',
            normalize=True,
            length = 0.5,
            arrow_length_ratio = 0.5,
            pivot = 'middle',
            **kwargs
            ):
        
        self.color              = color
        self.normalize          = normalize
        self.length             = length
        self.arrow_length_ratio = arrow_length_ratio
        self.pivot              = pivot
        
        super().__init__(
            evo = evo,
            **kwargs)
    
    
    def show(self):
        
        ## Create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ## Parameters at https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.quiver
        ax.quiver(
            self.X, 
            self.Y, 
            self.Z, 
            self.U, 
            self.V, 
            self.W,
            color               = self.color,
            normalize           = self.normalize,
            length              = self.length,
            arrow_length_ratio  = self.arrow_length_ratio,
            pivot               = self.pivot
        )
        
        ax.set_xlim([-0.05, 2.1])  # TODO: determine from self.vertices
        ax.set_ylim([-0.05, 3.05]) # TODO: determine from self.vertices
        ax.set_zlim([-0.05, 3.05]) # TODO: determine from self.vertices
        
        if hasattr(self,'noaxis') and self.noaxis: ax.set_axis_off()
        
        ## Tetrahedron lines
        ## TODO tidy this up.
        lines = combinations(self.vertices,2)
        i=0
        for x in lines:
            i+=1
            line=np.transpose(np.array(x))
        
            ## Make the back line a double dash
            linestyle = '--' if i == 5 else '-'
        
            ## https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot3D
            ax.plot3D(
                line[0],
                line[1],
                line[2],
                c='0',
                linestyle = linestyle,
                linewidth = 0.8
                )
        
        plt.show()
    
    def vector_to_barycentric(
            self,
            vector
            ):
        """
        Convert a 4d vector location to its equivalent within a tetrahedron

        Parameters
        ----------
        vector : TYPE
            DESCRIPTION.

        Returns
        -------
        barycentric_location : TYPE
            DESCRIPTION.

        """
        
        ## Initialise
        
        ## Normalise vector v to get vector u
        u = vector / vector.sum()
        
        ## Multiply vector u by the tetrahedron coordinates 
        barycentric_location = u @ self.vertices
        
        return barycentric_location
    
        
class Skyrms2010_1_2(Quiver3D):
    """
        Run the Skyrms (2010) Figure 2 simulation and plot.
    """
    
    def __init__(self):
        
        self.initialize_simulation()
        
        evo = self.run_simulation()
        
        super().__init__(
            evo                 = evo,
            noaxis              = True,
            show_immediately    = True)
    
    def initialize_simulation(self):
        
        ## Create interaction payoff matrix 
        self.payoffs = np.array(
            [
                [1,0.5,0.5,0], # Payoffs received by type 1 when encountering types 1, 2, 3, 4
                [0.5,0,1,0.5], # Payoffs received by type 2 when encountering types 1, 2, 3, 4
                [0.5,1,0,0.5], # Payoffs received by type 3 when encountering types 1, 2, 3, 4
                [0,0.5,0.5,1]  # Payoffs received by type 4 when encountering types 1, 2, 3, 4
            ]
        )
        
        ## And the player types are all of the pure strategies,
        ##  which just means four vectors each saying "strategy 1", "strategy 2" etc
        self.playertypes = np.array(
            [
                [1,0,0,0], # "I'm playing the first strategy!"
                [0,1,0,0], # "I'm playing the second strategy!"
                [0,0,1,0], # "I'm playing the third strategy!"
                [0,0,0,1]  # "I'm playing the fourth strategy!"
            ]
        )
        
        ## Tetrahedron vertices
        self.vertices = np.array([
            [1,1,5],          # summit
            [0,0,0],            # bottom left
            [2,2,0],            # bottom right
            [1.75,4.5,0]           # back
        ])
        
        
        
    def run_simulation(self):
        
        
        ## Create the game...
        game = NoSignal(self.payoffs)
        
        ## ...and the simulation.
        self.evo = OnePop(game,self.playertypes)
        
        
        ## Create arrows at roughly the places Skyrms depicts them.
        pop_vectors = np.array([
            [0.75,0.25,0,0], # left-upward
            [0.25,0.75,0,0], # left-upward
            # [0.75,0,0.25,0],
            # [0.25,0,0.75,0],
            [0,0.75,0.25,0], # front-inward (from left)
            [0,0.25,0.75,0], # front-inward (from right)
            [0.75,0,0,0.25],
            [0.25,0,0,0.75],
            [0,0.25,0,0.75],
            [0,0,0.25,0.75]
        ])
        
        pop_vectors_new = self.evo.discrete_replicator_delta_X(pop_vectors[0])
        
        for i in range(1,len(pop_vectors)):
            pop_vectors_new = np.vstack(
                (pop_vectors_new,
                 self.evo.discrete_replicator_delta_X(pop_vectors[i])
                 )
            )
        
        ## Get barycentric coordinates for the original population positions
        pop_vector_bary = np.apply_along_axis(
                                self.vector_to_barycentric,
                                axis=1,
                                arr=pop_vectors)
        
        ## Get barycentric coordinates for new population positions
        new_pop_vector_bary     = np.apply_along_axis(
                                self.vector_to_barycentric,
                                axis=1,
                                arr=pop_vectors_new)
        
        ## ... then get the difference between that and the current step.
        arrows = new_pop_vector_bary - pop_vector_bary
        
        ## Define big list of points and arrows
        # soa = np.hstack((vertices,vertices))
        
        soa = np.hstack((pop_vector_bary,arrows))
        

        self.X, self.Y, self.Z, self.U, self.V, self.W = zip(*soa)
        
        return self.evo
        