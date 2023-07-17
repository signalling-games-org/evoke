# -*- coding: utf-8 -*-
"""

evoke library examples from:
    
    Godfrey-Smith, P., & Martínez, M. (2013). Communication and Common Interest. 
    PLOS Computational Biology, 9(11), e1003282. https://doi.org/10.1371/journal.pcbi.1003282

"""

# Standard libraries
import numpy as np
from tqdm import tqdm,trange
import pygambit
import pickle
import json

# Custom library
from evoke.lib.figure import Scatter, Bar, Quiver2D, Quiver3D
from evoke.lib import asymmetric_games as asy
from evoke.lib import evolve as ev
from evoke.lib.symmetric_games import NoSignal
# from evoke.lib.common_interest import tau_per_rows as C_measure
from evoke.lib.info import Information

## Global variables

## Values of the common interest measure C for 3x3 equiprobable games
c_3x3_equiprobable = np.array([i/9 for i in range(0,10)]) 

## In demo mode, omit c=0 and c=0.111... as they are difficult to find quickly.
c_3x3_equiprobable_demo = np.array([i/9 for i in range(2,10)]) 

## Figures
## Each class inherits from a superclass from figure.py


class GodfreySmith2013_1(Scatter):
    """
    Original figure: https://doi.org/10.1371/journal.pcbi.1003282.g001

    How probable is an information-using equilibrium in a randomly-chosen game
     with a particular level of common interest?

    Common interest here is defined as Godfrey-Smith and Martínez's measure C.
    
    =====================
    How to use this class
    =====================
    
    You have two options to create this figure: demo mode and full mode.
    + Demo mode omits hard-to-find classes and places an upper limit on <games_per_c>.
       This allows it to run in a reasonable amount of time.
    + Full mode requires an existing set of pickled classes.
       These can be created via the class GodfreySmith2013_1_prep.
    
    The reason for demo mode is that the figure takes a VERY long time to create 
     with the published parameter of games_per_c=1500.
    Realistically we need to prepare by finding <games_per_c> games for each value of c,
     pickling them, and calling them at runtime to count the equilibria.
    Demo mode omits games with c=0.000 and c=0.111 because they are especially hard to find.
    
    """

    def __init__(self, games_per_c=50, demo=True, dir_games='/'):
        
        ## Is it demo mode?
        if demo: 
            
            ## Warn user of demo mode
            self.demo_warning()
            
            ## Set C values for demo mode
            self.c_values = c_3x3_equiprobable_demo
            
            ## Create games in demo mode
            self.create_games_demo(games_per_c)
            
        else:
            
            ## Set C values for full mode
            self.c_values = c_3x3_equiprobable
            
            ## Load games for full mode
            self.load_saved_games(dir_games,games_per_c)
        
        ## The main calculation method.
        self.calculate_results_per_c()

        # There are no evolving populations in this figure.
        # All the calculations are static.
        super().__init__(evo=None)

        # Set the data for the graph
        self.reset(x=self.c_values, 
                   y=self.info_using_equilibria,
                   xlabel="C", 
                   ylabel="Proportion of games",
                   xlim = [0,1],
                   ylim = [0,1])

        # # Show the graph, with the line between the scatter points
        self.show(line=True)
    
    def load_saved_games(self,dir_games,games_per_c):
        """
        Get sender and receiver matrices and load them into game objects.
        
        Put them into dictionary self.games.

        Parameters
        ----------
        dir_games : str
            Directory containing JSON files with sender and receiver matrices.
        games_per_c : int
            Number of games per value of C.

        """
        
        ## Game objects will be stored in a dictionary by C value.
        self.games = {f'{k:.3f}':[] for k in self.c_values}
        
        ## State chances and messages are always the same.
        state_chances = np.array([1/3,1/3,1/3])
        messages = 3
        
        ## Loop at C values...
        for c_value, games_list in self.games.items():
            
            ## Get name of JSON file
            fpath_json = f"{dir_games}games_c{c_value}_n{games_per_c}.json"
            
            ## The filename MUST have this format. Otherwise tell the user
            ##  they should use find_games_3x3() to find exactly this many games.
            with open(fpath_json,"r") as f:
                matrices = json.load(f)
            
            ## The entries in <matrices> go [sender,receiver,sender,receiver,...]
            my_matrix_iterator = iter(matrices)
            
            ## This is a nice way to iterate and jump two steps each time
            while (matrix := next(my_matrix_iterator, None)) is not None:
                
                ## Get sender and receiver, incrementing i
                sender_payoff_matrix = np.array(matrix)
                receiver_payoff_matrix = np.array(next(my_matrix_iterator))
                
                ## Create game
                self.games[c_value].append(asy.Chance(
                    state_chances,
                    sender_payoff_matrix,
                    receiver_payoff_matrix,
                    messages)
                )
            
    
    def create_games_demo(self,games_per_c):
        """
        Create game objects in demo mode.
        
        Put them into dictionary self.games.

        Parameters
        ----------
        games_per_c : int
            Number of games per value of C.

        """
        
        ## Game objects will be stored in a dictionary by C value.
        self.games = {f'{k:.3f}':[] for k in self.c_values}
        
        ## State chances and messages are always the same.
        state_chances = np.array([1/3,1/3,1/3])
        messages = 3
        
        ## Some helper variables
        total_games_surveyed = 0
        games_added_to_count = 0
        total_games_required = int(games_per_c * len(self.c_values))
        
        while True:
            total_games_surveyed +=1
            
            ## Generate random payoff matrices.
            sender_payoff_matrix = np.random.randint(0,9,(3,3))
            receiver_payoff_matrix = np.random.randint(0,9,(3,3))
            
            ## Check common interest
            c = calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix)
            
            ## Skip 0 and 1/9 values in demo mode
            if f'{c:.3f}' == '0.000' or f'{c:.3f}' == '0.111': continue
            
            ## Do we already have enough results for this value of C?
            if len(self.games[f'{c:.3f}']) >= games_per_c: continue
        
            ## This value of C still needs games.
            self.games[f'{c:.3f}'].append(asy.Chance(
                state_chances,
                sender_payoff_matrix,
                receiver_payoff_matrix,
                messages)
            )
            
            ## Do we have all the required games yet?
            games_added_to_count += 1
            
            ## Report progress
            if games_added_to_count % (np.floor(total_games_required/100)) == 0:
                print(f"Surveyed {total_games_surveyed} games; added {games_added_to_count} of {total_games_required}")
            
            if games_added_to_count == total_games_required: break
        

    def calculate_results_per_c(self):
        """

        For each value of <self.c_values>, count how many games
         have info-using equilibria.

        Returns
        -------
        None.

        """
        
        ## 1. Initialize
        results = {f'{k:.3f}':[] for k in self.c_values}
        
        ## 2. Loop at C values...
        for c_value, games_list in tqdm(self.games.items()):
            
            ## Loop at games per C value...
            for game in tqdm(games_list, disable=True):
        
                ## Assume there is no info-using equilibrium.
                results[c_value].append(False)
                
                ## Is there an info-using equilibrium?
                ## TODO: These next 10-15 lines should be a native method in the game.Chance() class.
                ## First get the gambit game object
                gambit_game = game.create_gambit_game()
                
                ## Now get the equilibria
                equilibria_gambit = pygambit.nash.lcp_solve(gambit_game, rational=False)
                
                ## Convert to python list
                equilibria = eval(str(equilibria_gambit))
                
                ## Now for each equilibrium, check if it is info-using
                for equilibrium in equilibria:
                    
                    ## Figure out whether the strategies at this equilibrium 
                    ##  lead to an information-using situation
                    sender_strategy = equilibrium[0]
                    receiver_strategy = equilibrium[1]
                    
                    info = Information(game,sender_strategy,receiver_strategy)
                    
                    if info.mutual_info_states_acts() > 0:
                        results[c_value][-1] = True
                        break
        
        ## Count the total number of info-using equilibria per level of C
        self.info_using_equilibria = []
        for key in sorted(results):
            self.info_using_equilibria.append(sum(results[key])/len(results[key]))
        

class GodfreySmith2013_2(Scatter):
    """
    Original figure: https://doi.org/10.1371/journal.pcbi.1003282.g002

    What is the highest level of information transmission at equilibrium
     across a sample of games with a particular level of common interest?

    Common interest here is defined as Godfrey-Smith and Martínez's measure C.
    
    =====================
    How to use this class
    =====================
    
    You have two options to create this figure: demo mode and full mode.
    + Demo mode omits hard-to-find classes and places an upper limit on <games_per_c>.
       This allows it to run in a reasonable amount of time.
    + Full mode requires an existing set of pickled classes.
       These can be created via the class GodfreySmith2013_2_prep.
    
    The reason for demo mode is that the figure takes a VERY long time to create 
     with the published parameter of games_per_c=1500.
    Realistically we need to prepare by finding <games_per_c> games for each value of c,
     pickling them, and calling them at runtime to count the equilibria.
    Demo mode omits games with c=0.000 and c=0.111 because they are especially hard to find.
    
    """

    def __init__(self, games_per_c=50, demo=True, dir_games='/'):
        
        ## Is it demo mode?
        if demo: 
            
            ## Warn user of demo mode
            self.demo_warning()
            
            ## Set C values for demo mode
            self.c_values = c_3x3_equiprobable_demo
            
            ## Create games in demo mode
            self.create_games_demo(games_per_c)
            
        else:
            
            ## Set C values for full mode
            self.c_values = c_3x3_equiprobable
            
            ## Load games for full mode
            self.load_saved_games(dir_games,games_per_c)
        
        ## The main calculation method.
        self.calculate_results_per_c(games_per_c)

        # There are no evolving populations in this figure.
        # All the calculations are static.
        super().__init__(evo=None)

        # Set the data for the graph
        self.reset(x=self.c_values, 
                   y=self.highest_mi,
                   xlabel="C", 
                   ylabel="Highest MI",
                   xlim = [0,1],
                   ylim = [0,2])

        # # Show the graph, with the line between the scatter points
        self.show(line=True)
    
    def load_saved_games(self,dir_games,games_per_c):
        """
        Get sender and receiver matrices and load them into game objects.
        
        Put them into dictionary self.games.

        Parameters
        ----------
        dir_games : str
            Directory containing JSON files with sender and receiver matrices.
        games_per_c : int
            Number of games per value of C.

        """
        
        ## Game objects will be stored in a dictionary by C value.
        self.games = {f'{k:.3f}':[] for k in self.c_values}
        
        ## State chances and messages are always the same.
        state_chances = np.array([1/3,1/3,1/3])
        messages = 3
        
        ## Loop at C values...
        for c_value, games_list in self.games.items():
            
            ## Get name of JSON file
            fpath_json = f"{dir_games}games_c{c_value}_n{games_per_c}.json"
            
            ## The filename MUST have this format. Otherwise tell the user
            ##  they should use find_games_3x3() to find exactly this many games.
            with open(fpath_json,"r") as f:
                matrices = json.load(f)
            
            ## The entries in <matrices> go [sender,receiver,sender,receiver,...]
            my_matrix_iterator = iter(matrices)
            
            ## This is a nice way to iterate and jump two steps each time
            while (matrix := next(my_matrix_iterator, None)) is not None:
                
                ## Get sender and receiver, incrementing i
                sender_payoff_matrix = np.array(matrix)
                receiver_payoff_matrix = np.array(next(my_matrix_iterator))
                
                ## Create game
                self.games[c_value].append(asy.Chance(
                    state_chances,
                    sender_payoff_matrix,
                    receiver_payoff_matrix,
                    messages)
                )
    
    def create_games_demo(self,games_per_c):
        """
        Create game objects in demo mode.
        
        Put them into dictionary self.games.

        Parameters
        ----------
        games_per_c : int
            Number of games per value of C.

        """
        
        ## Game objects will be stored in a dictionary by C value.
        self.games = {f'{k:.3f}':[] for k in self.c_values}
        
        ## State chances and messages are always the same.
        state_chances = np.array([1/3,1/3,1/3])
        messages = 3
        
        ## Some helper variables
        total_games_surveyed = 0
        games_added_to_count = 0
        total_games_required = int(games_per_c * len(self.c_values))
        
        while True:
            total_games_surveyed +=1
            
            ## Generate random payoff matrices.
            sender_payoff_matrix = np.random.randint(0,9,(3,3))
            receiver_payoff_matrix = np.random.randint(0,9,(3,3))
            
            ## Check common interest
            c = calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix)
            
            ## Skip 0 and 1/9 values in demo mode
            if f'{c:.3f}' == '0.000' or f'{c:.3f}' == '0.111': continue
            
            ## Do we already have enough results for this value of C?
            if len(self.games[f'{c:.3f}']) >= games_per_c: continue
        
            ## This value of C still needs games.
            self.games[f'{c:.3f}'].append(asy.Chance(
                state_chances,
                sender_payoff_matrix,
                receiver_payoff_matrix,
                messages)
            )
            
            ## Do we have all the required games yet?
            games_added_to_count += 1
            
            ## Report progress
            if games_added_to_count % (np.floor(total_games_required/100)) == 0:
                print(f"Surveyed {total_games_surveyed} games; added {games_added_to_count} of {total_games_required}")
            
            if games_added_to_count == total_games_required: break
        

    def calculate_results_per_c(self, games_per_c):
        """

        For each value of <self.c_values>, count how many out of <games_per_c> games
         have info-using equilibria.

        Parameters
        ----------
        games_per_c : int
            Number of games to generate per level of common interest.

        Returns
        -------
        None.

        """
        
        ## 1. Initialize
        results = {f'{k:.3f}':[] for k in self.c_values}
        
        ## 2. Loop at C values...
        for c_value, games_list in tqdm(self.games.items()):
            
            ## Loop at games per C value...
            for game in tqdm(games_list, disable=True):
        
                ## Assume there is no info-using equilibrium.
                results[c_value].append(False)
                
                ## Is there an info-using equilibrium?
                ## TODO: These next 10-15 lines should be a native method in the game.Chance() class.
                ## First get the gambit game object
                gambit_game = game.create_gambit_game()
                
                ## Now get the equilibria
                equilibria_gambit = pygambit.nash.lcp_solve(gambit_game, rational=False)
                
                ## Convert to python list
                equilibria = eval(str(equilibria_gambit))
                
                ## Now for each equilibrium, check if it is info-using
                current_highest_mi = 0
                for equilibrium in equilibria:
                    
                    ## Figure out whether the strategies at this equilibrium 
                    ##  lead to an information-using situation
                    sender_strategy = equilibrium[0]
                    receiver_strategy = equilibrium[1]
                    
                    info = Information(game,sender_strategy,receiver_strategy)
                    
                    ## Does this equilibrium have a higher level of mutual information
                    ##  than the previous highest for this game?
                    new_mi = info.mutual_info_states_acts()
                    
                    if new_mi > current_highest_mi:
                        results[c_value][-1] = new_mi
                        
                        ## The current highest mutual information is this one now.
                        current_highest_mi = new_mi
        
        ## Count the total number of info-using equilibria per level of C
        self.highest_mi = []
        for key in sorted(results):
            self.highest_mi.append(max(results[key]))
        
        
        
        
"""
    Shared methods
"""      
        
def calculate_C(state_chances,sender_payoff_matrix,receiver_payoff_matrix):
    """
    I'm not sure common_interest.tau_per_rows() is doing what it should be,
     so I'm going to recreate the C measure here to check.

    Returns
    -------
    c : float
        PGS & MM's measure C.

    """
    
    ## 1. Initialise
    sum_total = 0

    ## Loop around states and pairs of acts
    for i in range(len(state_chances)):
        
        ## This will loop around acts
        for j in range(len(sender_payoff_matrix[i])-1):
            
            ## This will loop around acts with a higher index than j
            for k in range(j+1,len(sender_payoff_matrix[i])):
                
                ## Calculate sender's d-component
                if sender_payoff_matrix[i][j] < sender_payoff_matrix[i][k]:
                    d_sender_component = 0
                elif sender_payoff_matrix[i][j] == sender_payoff_matrix[i][k]:
                    d_sender_component = 1/2
                else:
                    d_sender_component = 1
                 
                ## Calculate receiver's d-component
                if receiver_payoff_matrix[i][j] < receiver_payoff_matrix[i][k]:
                    d_receiver_component = 0
                elif receiver_payoff_matrix[i][j] == receiver_payoff_matrix[i][k]:
                    d_receiver_component = 1/2
                else:
                    d_receiver_component = 1
          
                ## Get this component of the sum
                factor = np.floor(abs(d_sender_component - d_receiver_component))
                
                ## Add this component to the total sum
                sum_total += factor * state_chances[i]
          
    subtractor = 2*sum_total / 6
    
    c = 1 - subtractor
    
    return c
    

def find_games_3x3(dir_out,games_per_c=1500,c_values=c_3x3_equiprobable):
    """
    Finds <games_per_c> 3x3 sender and receiver matrices
     and save them as JSON files, storing them by C value in <dir_out>.
    
    Since it's hard to find games for certain values of C, we'll save each
     JSON file individually once we've found it.
    Then if you have to terminate early, you can come back and just search for
     games with the values of C you need later on.

    Parameters
    ----------
    fpath_out : str
        File location of the stored pickled dictionary of matrices
    games_per_c : int, optional
        Number of games to find per value of c. The default is 1500.

    Returns
    -------
    None.

    """
    
    ## Initialise
    ## Values of C in truncated string format
    c_outstanding = {f'{k:.3f}' for k in c_values}
    
    ## Dict to conveniently store the matrices before outputting.
    results = {f'{k:.3f}':[] for k in c_values}
    
    ## State chances (required to calculate C)
    state_chances = np.array([1/3,1/3,1/3])
    
    ## While there are values of C that have not yet had all games found and pickled...
    while len(c_outstanding)>0:
    
        ## Create a game
        sender_payoff_matrix = np.random.randint(0,9,(3,3))
        receiver_payoff_matrix = np.random.randint(0,9,(3,3))
        
        ## Check common interest
        c_value = calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix)
        
        ## Does this value of C require pickling?
        if f'{c_value:.3f}' in c_outstanding:
        
            ## If yes, add to the output dict.
            ## We just add the sender and receiver one by one.
            ## We will always know they should be loaded that way too.
            results[f'{c_value:.3f}'].append(sender_payoff_matrix.tolist())
            results[f'{c_value:.3f}'].append(receiver_payoff_matrix.tolist())
            
            ## has this value of C now had all its games found?
            if len(results[f'{c_value:.3f}'])>= games_per_c*2:
            
                ## If yes, save JSON file and remove this value of C from the "to-do" list.
                fpath_out = f"{dir_out}games_c{c_value:.3f}_n{games_per_c}.json"
                
                with open(fpath_out, "w") as f:
                    json.dump(results[f'{c_value:.3f}'],f)
                
                ## Remove this value of C from the "to-do" list
                c_outstanding.remove(f'{c_value:.3f}')
                
                ## Are there any values of C left outstanding? If not, quit the while-loop.
                if len(c_outstanding)==0: break
            
                ## Otherwise, print remaining values
                print(f"C values remaining: {c_outstanding}")

    