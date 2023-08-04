# -*- coding: utf-8 -*-
"""

evoke library examples from:
    
    Godfrey-Smith, P., & Martínez, M. (2013). Communication and Common Interest. 
    PLOS Computational Biology, 9(11), e1003282. https://doi.org/10.1371/journal.pcbi.1003282

======================
HOW TO USE THIS SCRIPT
======================

Quick run: Create figure objects with demo=True.

Full run, minimal parameters:
1. Decide how many games per value of C you want to analyse, <games_per_c>. 
    Godfrey-Smith and Martínez use 1500.
2. Run find_games_3x3(<games_per_c>). This will generate <games_per_c> games per value of C
    and store them in a local directory.
3. Run analyse_games_3x3(<games_per_c>). This will calculate values required to create
    figures 1 and 2. This can take a long time! 1500 games takes about 30 minutes.
4. Run GodfreySmith2013_1(<games_per_c>, demo=False) to create Figure 1.
5. Run GodfreySmith2013_2(<games_per_c>, demo=False) to create Figure 2.

"""

# Standard libraries
import numpy as np
from tqdm import tqdm, trange
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

# Global variables

# Values of the common interest measure C for 3x3 equiprobable games
c_3x3_equiprobable = np.array([i / 9 for i in range(0, 10)])

# In demo mode, omit c=0 and c=0.111... as they are difficult to find quickly.
c_3x3_equiprobable_demo = np.array([i / 9 for i in range(2, 10)])

# Figures
# Each class inherits from a superclass from figure.py


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

    def __init__(self, games_per_c=50, demo=True, dir_games="../data/"):
        # Is it demo mode?
        if demo:
            # Warn user of demo mode
            self.demo_warning()

            # Set C values for demo mode
            self.c_values = c_3x3_equiprobable_demo

            # Create games in demo mode
            self.create_games_demo(games_per_c)

        else:
            # Set C values for full mode
            self.c_values = c_3x3_equiprobable

            # Load games for full mode
            self.load_saved_games(dir_games, games_per_c)

        # The main calculation method.
        self.calculate_results_per_c()

        # There are no evolving populations in this figure.
        # All the calculations are static.
        super().__init__(evo=None)

        # Set the data for the graph
        self.reset(
            x=self.c_values,
            y=self.info_using_equilibria,
            xlabel="C",
            ylabel="Proportion of games",
            xlim=[0, 1],
            ylim=[0, 1],
        )

        # # Show the graph, with the line between the scatter points
        self.show(line=True)

    def load_saved_games(self, dir_games, games_per_c):
        """
        Get sender and receiver matrices and load them into game objects.
         Put them into dictionary self.games.

        The games should already exist in dir_games with filenames of the form:

            f"{dir_games}games_c{c_value:.3f}_n{games_per_c}.json"

        Parameters
        ----------
        dir_games : str
            Directory containing JSON files with sender and receiver matrices.
        games_per_c : int
            Number of games per value of C.

        """

        # Game objects will be stored in a dictionary by C value.
        self.games = {f"{k:.3f}": [] for k in self.c_values}

        # State chances and messages are always the same.
        state_chances = np.array([1 / 3, 1 / 3, 1 / 3])
        messages = 3

        # Loop at C values...
        for c_value, games_list in self.games.items():
            # Get name of JSON file
            fpath_json = f"{dir_games}games_c{c_value}_n{games_per_c}.json"

            # The filename MUST have this format. Otherwise tell the user
            # they should use find_games_3x3() to find exactly this many games.
            with open(fpath_json, "r") as f:
                games_list_loaded = json.load(f)

            ## Load each game into an object
            for game_dict in games_list_loaded:
                ## Create game
                game = asy.Chance(
                    state_chances,
                    np.array(game_dict["s"]),  # sender payoff matrix
                    np.array(game_dict["r"]),  # receiver payoff matrix
                    messages,
                )

                ## Append information-using equilibria as game object attribute
                game.best_equilibrium = game_dict["e"]
                game.info_transmission_at_best_equilibrium = game_dict["i"]

                ## Append this game to the figure object's game list.
                self.games[c_value].append(game)

    def create_games_demo(self, games_per_c):
        """
        Create game objects in demo mode.

        Put them into dictionary self.games.

        Parameters
        ----------
        games_per_c : int
            Number of games per value of C.

        """

        # Game objects will be stored in a dictionary by C value.
        self.games = {f"{k:.3f}": [] for k in self.c_values}

        # State chances and messages are always the same.
        state_chances = np.array([1 / 3, 1 / 3, 1 / 3])
        messages = 3

        # Some helper variables
        total_games_surveyed = 0
        games_added_to_count = 0
        total_games_required = int(games_per_c * len(self.c_values))

        while True:
            total_games_surveyed += 1

            # Generate random payoff matrices.
            sender_payoff_matrix = np.random.randint(0, 9, (3, 3))
            receiver_payoff_matrix = np.random.randint(0, 9, (3, 3))

            # Check common interest
            c = calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix)

            # Skip 0 and 1/9 values in demo mode
            if f"{c:.3f}" == "0.000" or f"{c:.3f}" == "0.111":
                continue

            # Do we already have enough results for this value of C?
            if len(self.games[f"{c:.3f}"]) >= games_per_c:
                continue

            # This value of C still needs games.
            self.games[f"{c:.3f}"].append(
                asy.Chance(
                    state_chances,
                    sender_payoff_matrix,
                    receiver_payoff_matrix,
                    messages,
                )
            )

            # Do we have all the required games yet?
            games_added_to_count += 1

            # Report progress
            if games_added_to_count % (np.floor(total_games_required / 100)) == 0:
                print(
                    f"Surveyed {total_games_surveyed} games; added {games_added_to_count} of {total_games_required}"
                )

            if games_added_to_count == total_games_required:
                break

    def calculate_results_per_c(self):
        """

        For each value of <self.c_values>, count how many games
         have info-using equilibria.

        Returns
        -------
        None.

        """

        # 1. Initialize
        results = {f"{k:.3f}": [] for k in self.c_values}

        # 2. Loop at C values...
        for c_value, games_list in tqdm(self.games.items()):
            # Loop at games per C value...
            for game in tqdm(games_list, disable=True):
                ## If this game's info transmission at its best equilibrium
                ##  is greater than zero...
                if game.info_transmission_at_best_equilibrium > 0:
                    ## Append True to the results list.
                    results[c_value].append(True)

                else:  # otherwise...
                    ## Append False to the results list.
                    results[c_value].append(False)

        ## Count the total number of info-using equilibria per level of C
        self.info_using_equilibria = []
        for key in sorted(results):  # for each level of C...
            self.info_using_equilibria.append(
                sum(results[key]) / len(results[key])
            )  # ...get the proportion of info-using equilibria.


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

    def __init__(self, games_per_c=50, demo=True, dir_games="../data/"):
        # Is it demo mode?
        if demo:
            # Warn user of demo mode
            self.demo_warning()

            # Set C values for demo mode
            self.c_values = c_3x3_equiprobable_demo

            # Create games in demo mode
            self.create_games_demo(games_per_c)

        else:
            # Set C values for full mode
            self.c_values = c_3x3_equiprobable

            # Load games for full mode
            self.load_saved_games(dir_games, games_per_c)

        # The main calculation method.
        self.calculate_results_per_c(games_per_c)

        # There are no evolving populations in this figure.
        # All the calculations are static.
        super().__init__(evo=None)

        # Set the data for the graph
        self.reset(
            x=self.c_values,
            y=self.highest_mi,
            xlabel="C",
            ylabel="Highest MI",
            xlim=[0, 1],
            ylim=[0, 2],
        )

        # # Show the graph, with the line between the scatter points
        self.show(line=True)

    def load_saved_games(self, dir_games, games_per_c):
        """
        Get sender and receiver matrices and load them into game objects.
         Put them into dictionary self.games.

        The games should already exist in dir_games with filenames of the form:

            f"{dir_games}games_c{c_value:.3f}_n{games_per_c}.json"

        Parameters
        ----------
        dir_games : str
            Directory containing JSON files with sender and receiver matrices.
        games_per_c : int
            Number of games per value of C.

        """

        # Game objects will be stored in a dictionary by C value.
        self.games = {f"{k:.3f}": [] for k in self.c_values}

        # State chances and messages are always the same.
        state_chances = np.array([1 / 3, 1 / 3, 1 / 3])
        messages = 3

        # Loop at C values...
        for c_value, games_list in self.games.items():
            # Get name of JSON file
            fpath_json = f"{dir_games}games_c{c_value}_n{games_per_c}.json"

            # The filename MUST have this format. Otherwise tell the user
            # they should use find_games_3x3() to find exactly this many games.
            with open(fpath_json, "r") as f:
                games_list_loaded = json.load(f)

            ## Load each game into an object
            for game_dict in games_list_loaded:
                ## Create game
                game = asy.Chance(
                    state_chances,
                    np.array(game_dict["s"]),  # sender payoff matrix
                    np.array(game_dict["r"]),  # receiver payoff matrix
                    messages,
                )

                ## Append information-using equilibria as game object attribute
                game.best_equilibrium = game_dict["e"]
                game.info_transmission_at_best_equilibrium = game_dict["i"]

                ## Append this game to the figure object's game list.
                self.games[c_value].append(game)

    def create_games_demo(self, games_per_c):
        """
        Create game objects in demo mode.

        Put them into dictionary self.games.

        Parameters
        ----------
        games_per_c : int
            Number of games per value of C.

        """

        # Game objects will be stored in a dictionary by C value.
        self.games = {f"{k:.3f}": [] for k in self.c_values}

        # State chances and messages are always the same.
        state_chances = np.array([1 / 3, 1 / 3, 1 / 3])
        messages = 3

        # Some helper variables
        total_games_surveyed = 0
        games_added_to_count = 0
        total_games_required = int(games_per_c * len(self.c_values))

        while True:
            total_games_surveyed += 1

            # Generate random payoff matrices.
            sender_payoff_matrix = np.random.randint(0, 9, (3, 3))
            receiver_payoff_matrix = np.random.randint(0, 9, (3, 3))

            # Check common interest
            c = calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix)

            # Skip 0 and 1/9 values in demo mode
            if f"{c:.3f}" == "0.000" or f"{c:.3f}" == "0.111":
                continue

            # Do we already have enough results for this value of C?
            if len(self.games[f"{c:.3f}"]) >= games_per_c:
                continue

            # This value of C still needs games.
            self.games[f"{c:.3f}"].append(
                asy.Chance(
                    state_chances,
                    sender_payoff_matrix,
                    receiver_payoff_matrix,
                    messages,
                )
            )

            # Do we have all the required games yet?
            games_added_to_count += 1

            # Report progress
            if games_added_to_count % (np.floor(total_games_required / 100)) == 0:
                print(
                    f"Surveyed {total_games_surveyed} games; added {games_added_to_count} of {total_games_required}"
                )

            if games_added_to_count == total_games_required:
                break

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

        # 1. Initialize
        results = {f"{k:.3f}": [] for k in self.c_values}

        # 2. Loop at C values...
        for c_value, games_list in tqdm(self.games.items()):
            # Loop at games per C value...
            for game in tqdm(games_list, disable=True):
                ## Get this game's highest info-transmission.
                results[c_value].append(game.info_transmission_at_best_equilibrium)

        # Count the total number of info-using equilibria per level of C
        self.highest_mi = []
        for key in sorted(results):
            self.highest_mi.append(max(results[key]))


"""
    Shared methods
"""


def calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix) -> float:
    """

    Calculate C as per Godfrey-Smith and Martínez's definition.

    (It's not clear whether common_interest.tau_per_rows() is doing what it should be,
     so we will calculate C explicitly here instead.)

    Returns
    -------
    c : float
        PGS & MM's measure C.

    """

    # 1. Initialise
    sum_total = 0

    # Loop around states and pairs of acts
    for i in range(len(state_chances)):
        # This will loop around acts
        for j in range(len(sender_payoff_matrix[i]) - 1):
            # This will loop around acts with a higher index than j
            for k in range(j + 1, len(sender_payoff_matrix[i])):
                # Calculate sender's d-component
                if sender_payoff_matrix[i][j] < sender_payoff_matrix[i][k]:
                    d_sender_component = 0
                elif sender_payoff_matrix[i][j] == sender_payoff_matrix[i][k]:
                    d_sender_component = 1 / 2
                else:
                    d_sender_component = 1

                # Calculate receiver's d-component
                if receiver_payoff_matrix[i][j] < receiver_payoff_matrix[i][k]:
                    d_receiver_component = 0
                elif receiver_payoff_matrix[i][j] == receiver_payoff_matrix[i][k]:
                    d_receiver_component = 1 / 2
                else:
                    d_receiver_component = 1

                # Get this component of the sum
                factor = np.floor(abs(d_sender_component - d_receiver_component))

                # Add this component to the total sum
                sum_total += factor * state_chances[i]

    subtractor = 2 * sum_total / 6

    c = 1 - subtractor

    return c


def find_games_3x3(
    games_per_c=1500, c_values=c_3x3_equiprobable, dir_games="../data/"
) -> None:
    """
    Finds <games_per_c> 3x3 sender and receiver matrices
     and saves them as JSON files, storing them by C value in <dir_games>.

    Since it's hard to find games for certain values of C, we'll save each
     JSON file individually once we've found it.
    Then if you have to terminate early, you can come back and just search for
     games with the values of C you need later on.

    Parameters
    ----------
    dir_games : str
        Directory to place JSON files
    games_per_c : int, optional
        Number of games to find per value of c. The default is 1500.
    c_values : array-like
        List of C values to find games for.
        The default is the global variable c_3x3_equiprobable.

    Returns
    -------
    None.

    """

    ## Initialise
    ## Values of C in truncated string format
    c_outstanding = {f"{k:.3f}" for k in c_values}

    ## Dict to conveniently store the matrices before outputting.
    results = {f"{k:.3f}": [] for k in c_values}

    ## State chances (required to calculate C)
    state_chances = np.array([1 / 3, 1 / 3, 1 / 3])

    ## While there are values of C that have not yet had all games found and saved...
    while len(c_outstanding) > 0:
        ## Create a game
        sender_payoff_matrix = np.random.randint(0, 9, (3, 3))
        receiver_payoff_matrix = np.random.randint(0, 9, (3, 3))

        ## Check common interest
        c_value = calculate_C(
            state_chances, sender_payoff_matrix, receiver_payoff_matrix
        )

        ## Does this value of C require saving?
        if f"{c_value:.3f}" in c_outstanding:
            ## If yes, add to the output dict.
            ## Each value in the output dict is a list of dicts (i.e. a list of games)
            ##  with this format:
            ##
            ## {
            ##      "s": <sender payoff matrix>
            ##      "r": <receiver payoff matrix>
            ##      "e": <equilibrium with the highest information transmission>
            ##      "i": <mutual information between states and acts at this equilibrium>
            ## }
            ##
            ## The values for e and i will be left blank here.
            ## The function analyse_games() will fill them in.

            ## Create game dict.
            results[f"{c_value:.3f}"].append(
                {
                    "s": sender_payoff_matrix.tolist(),
                    "r": receiver_payoff_matrix.tolist(),
                }
            )

            ## Has this value of C now had all its games found?
            if len(results[f"{c_value:.3f}"]) >= games_per_c:
                ## If yes, save JSON file and remove this value of C from the "to-do" list.
                fpath_out = f"{dir_games}games_c{c_value:.3f}_n{games_per_c}.json"

                with open(fpath_out, "w") as f:
                    json.dump(results[f"{c_value:.3f}"], f)

                ## Remove this value of C from the "to-do" list
                c_outstanding.remove(f"{c_value:.3f}")

                ## Are there any values of C left outstanding? If not, quit the while-loop.
                if len(c_outstanding) == 0:
                    break

                ## Otherwise, print remaining values
                print(f"C values remaining: {c_outstanding}")


def analyse_games_3x3(
    games_per_c=1500, c_values=c_3x3_equiprobable, dir_games="../data/", sigfig = 5
) -> None:
    """
    Find information-using equilibria of 3x3 games
     and the mutual information between states and acts at those equilibria.

    The games should already exist in dir_games with filenames of the form:

        f"{dir_games}games_c{c_value:.3f}_n{games_per_c}.json"

    Each file should be a list of dicts. Each dict corresponds to a game:

        {
            "s": <sender payoff matrix>
            "r": <receiver payoff matrix>
            "e": <equilibrium with the highest information transmission>
            "i": <mutual information between states and acts at this equilibrium>
        }

    s and r already exist; this function fills in e and i.

    Parameters
    ----------
    dir_games : str
        Directory to find and store JSON files
    games_per_c : int, optional
        Number of games to find per value of c. The default is 1500.
    c_values : array-like
        List of C values to find games for.
        The default is the global variable c_3x3_equiprobable.

    Returns
    -------
    None

    """

    ## Game objects will be stored in a dictionary by C value.
    games = {f"{k:.3f}": [] for k in c_values}

    ## State chances and messages are always the same.
    state_chances = np.array([1 / 3, 1 / 3, 1 / 3])
    messages = 3

    ## Loop at C values...
    for c_value, games_list in tqdm(games.items()):
        ## Get name of JSON file
        fpath_json = f"{dir_games}games_c{c_value}_n{games_per_c}.json"

        ## The filename MUST have this format. Otherwise tell the user
        ##  they should use find_games_3x3() to find exactly this many games.
        with open(fpath_json, "r") as f:
            games_list_loaded = json.load(f)

        ## Load each game into an object
        for game_dict in games_list_loaded:
            ## Create the game
            game = asy.Chance(
                state_chances=state_chances,
                sender_payoff_matrix=np.array(game_dict["s"]),
                receiver_payoff_matrix=np.array(game_dict["r"]),
                messages=messages
            )

            # Is there an info-using equilibrium?
            # TODO: These next 10-15 lines should be a native method in the game.Chance() class.
            # First get the gambit game object
            gambit_game = game.create_gambit_game()

            # Now get the equilibria
            equilibria_gambit = pygambit.nash.lcp_solve(gambit_game, rational=False)

            # Convert to python list
            equilibria = eval(str(equilibria_gambit))

            ## Initialize
            current_highest_info_at_equilibrium = 0

            # Now for each equilibrium, check if it is info-using
            for equilibrium in equilibria:
                
                # Figure out whether the strategies at this equilibrium
                # lead to an information-using situation.
                
                ## Sometimes gambit gives back long decimals e.g. 0.999999996
                ## We want to round these before dumping to a file.
                sender_strategy = np.around(np.array(equilibrium[0]),sigfig)
                receiver_strategy = np.around(np.array(equilibrium[1]),sigfig)

                ## Create info object to make info measurements
                info = Information(game, sender_strategy, receiver_strategy)

                ## Get current mutual info
                ## Sometimes it spits out -0.0, which should be 0.0.
                current_mutual_info = abs(info.mutual_info_states_acts())

                if current_mutual_info >= current_highest_info_at_equilibrium:
                    
                    ## Update game details
                    ## The equilibrium is just the current sender strategy
                    ##  followed by the current receiver strategy.
                    game_dict["e"] = [sender_strategy.tolist(),receiver_strategy.tolist()]
                    
                    ## The mutual information is the current mutual info at this equilibrium.
                    game_dict[
                        "i"
                    ] = current_highest_info_at_equilibrium = round(current_mutual_info,sigfig)

        ## Dump this updated game file
        with open(fpath_json, "w") as f:
            json.dump(games_list_loaded, f)
