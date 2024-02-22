# -*- coding: utf-8 -*-
"""

evoke library examples from:
    
    Godfrey-Smith, P., & Martínez, M. (2013). Communication and Common Interest. 
    *PLOS Computational Biology*, 9(11), e1003282. https://doi.org/10.1371/journal.pcbi.1003282

The supporting information (including important definitions)
can be found at https://doi.org/10.1371/journal.pcbi.1003282.s001

======================
How to use this script
======================

Quick run: Create figure objects with `demo=True`.

Figures 1 and 2, full run, minimal parameters:
    
    1. Decide how many games per value of C you want to analyse, `games_per_c`. 
       Godfrey-Smith and Martínez use 1500.
    2. Run `find_games_3x3(games_per_c)`. 
       This will generate `games_per_c` games per value of C and store them in a local directory.
    3. Run `analyse_games_3x3(games_per_c)`. This will calculate values required to create figures 1 and 2. 
       This can take a long time! 1500 games takes about 30 minutes.
    4. Run `GodfreySmith2013_1(games_per_c, demo=False)` to create Figure 1.
    5. Run `GodfreySmith2013_2(games_per_c, demo=False)` to create Figure 2.

Figure 3a (sender), full run, minimal parameters:
    
    1. Decide how many games per value of C and K you want to analyse, `games_per_c_and_k`. 
       Godfrey-Smith and Martínez use 1500.
    2. Run `find_games_3x3_c_and_k(games_per_c_and_k,sender=True)`.
       This will generate `games_per_c_and_k` games per pair of values C and K and store them locally.
    3. Run `analyse_games_3x3_c_and_k(games_per_c_and_k,sender=True)`. This will calculate values required to create figure 3.
       This can take a long time!
    4. Run `GodfreySmith2013_3_sender(games_per_c_and_k,demo=False)` to create Figure 3a.

Figure 3b (receiver), full run, minimal parameters:
    
    1. Decide how many games per value of C and K you want to analyse, `games_per_c_and_k`. 
       Godfrey-Smith and Martínez use 1500.
    2. Run `find_games_3x3_c_and_k(games_per_c_and_k,sender=False)`.
       This will generate `games_per_c_and_k` games per pair of values C and K and store them locally.
    3. Run `analyse_games_3x3_c_and_k(games_per_c_and_k,sender=False)`. This will calculate values required to create figure 3.
       This can take a long time!
    4. Run `GodfreySmith2013_3_receiver(games_per_c_and_k,demo=False)` to create Figure 3a.

"""

# Standard libraries
import numpy as np
from tqdm import tqdm
import json
from itertools import combinations

# Custom library
from evoke.src.figure import Scatter, Surface
from evoke.src import asymmetric_games as asy

# from evoke.lib.common_interest import tau_per_rows as C_measure
import evoke.src.exceptions as ex

# Global variables

# Values of the common interest measure C for 3x3 equiprobable games
c_3x3_equiprobable = np.array([i / 9 for i in range(0, 10)])

# In demo mode, omit c=0 and c=0.111... as they are difficult to find quickly.
c_3x3_equiprobable_demo = np.array([i / 9 for i in range(2, 10)])

# Values of the K measure for 3x3 games
k_3x3 = np.array([i / 3 for i in range(0, 7)])

# Values of K to exclude when C=0 (because the combination is impossible)
k_3x3_excluded_at_c_0 = np.array([1 / 3, 3 / 3, 5 / 3])

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

    + Demo mode omits hard-to-find classes and places an upper limit on `games_per_c`.
      This allows it to run in a reasonable amount of time.
    + Full mode requires an existing set of game data stored in JSON files.
      These can be created via the functions `find_games_3x3()` and `analyse_games_3x3()`.

    The reason for demo mode is that the figure takes a VERY long time to create
    with the published parameter of `games_per_c=1500`.
    Realistically we need to prepare by finding `games_per_c` games for each value of C,
    storing them in a local JSON file, and calling them at runtime to count the equilibria.
    Demo mode omits games with `c=0.000` and `c=0.111` because they are especially hard to find.

    """

    def __init__(self, games_per_c=50, demo=True, dir_games="../data/") -> None:
        """
        Create an instance of the class.

        Parameters
        ----------
        games_per_c : int, optional
            Number of games per value of C to create. The default is 50.
        demo : bool, optional
            Whether to create the figure in demo mode or not.
            The default is True.
        dir_games : str, optional
            Directory path to find stored JSON game data.
            Only used if `demo=False`.
            The default is "../data/".
        """

        # Is it demo mode?
        if demo:
            # Warn user of demo mode
            self.demo_warning()

            # Set C values for demo mode
            self.c_values = c_3x3_equiprobable_demo

            # Create games in demo mode
            try:
                self.create_games_demo(games_per_c)
            except ex.ModuleNotInstalledException as e:
                print(e)
                return

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

        # Show the graph, with the line between the scatter points
        self.show_line = True
        self.show()

    def load_saved_games(self, dir_games, games_per_c) -> None:
        """
        Get sender and receiver matrices and load them into game objects.
        Put them into dictionary self.games.

        The games should already exist in dir_games with filenames of the form:

        `f"{dir_games}games_c{c_value:.3f}_n{games_per_c}.json"`

        Parameters
        ----------
        dir_games : str
            Directory containing JSON files with sender and receiver matrices.
        games_per_c : int
            Number of games per value of C.

        """

        # Game objects will be stored in a dictionary by C value.
        self.games = {f"{k:.3f}": [] for k in self.c_values}

        # Loop at C values...
        for c_value, games_list in self.games.items():
            # Get name of JSON file
            fpath_json = f"{dir_games}games_c{c_value}_n{games_per_c}.json"

            # The filename MUST have this format. Otherwise tell the user
            # they should use find_games_3x3() to find exactly this many games.
            try:
                with open(fpath_json, "r") as f:
                    games_list_loaded = json.load(f)

            except FileNotFoundError:
                raise ex.NoDataException(
                    f"File {fpath_json} was not found. Have you run find_games_3x3() and analyse_games_3x3() for games_per_c={games_per_c} yet?"
                )

            # Append these game to the figure object's game list.
            self.games[c_value] = games_list_loaded

    def create_games_demo(self, games_per_c) -> None:
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
            sender_payoff_matrix = get_random_payoffs()
            receiver_payoff_matrix = get_random_payoffs()

            # Check common interest
            c = calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix)

            # Skip 0 and 1/9 values in demo mode
            if f"{c:.3f}" == "0.000" or f"{c:.3f}" == "0.111":
                continue

            # Do we already have enough results for this value of C?
            if len(self.games[f"{c:.3f}"]) >= games_per_c:
                continue

            # This value of C still needs games.
            # Create the game...
            game = asy.Chance(
                state_chances, sender_payoff_matrix, receiver_payoff_matrix, messages
            )

            # ...and associate it with this value of C.
            # We are just storing a dict with the key features of the game.
            # That's what calculate_results_per_c() expects.

            # Get the highest info-using equilibrium
            # and the mutual info between states and acts at that point.
            e, i = game.highest_info_using_equilibrium

            # Build the game dict and add it to self.games.
            game_dict = {
                "s": sender_payoff_matrix,
                "r": receiver_payoff_matrix,
                "e": e,
                "i": i,
            }
            self.games[f"{c:.3f}"].append(game_dict)

            # Do we have all the required games yet?
            games_added_to_count += 1

            # Report progress
            if (
                total_games_required > 100
                and games_added_to_count % (np.floor(total_games_required / 100)) == 0
            ):
                print(
                    f"Surveyed {total_games_surveyed} games; added {games_added_to_count} of {total_games_required}"
                )

            if games_added_to_count == total_games_required:
                break

    def calculate_results_per_c(self) -> None:
        """
        For each value of <self.c_values>, count how many games
        have info-using equilibria.
        """

        # 1. Initialize
        # results = {f"{k:.3f}": [] for k in self.c_values}
        self.info_using_equilibria = []

        # 2. Loop at sorted C values...
        for c_value, games_list in tqdm(sorted(self.games.items())):
            # games_list is a list of dicts.
            # Get the highest info-using equilibrium per game.
            i_values = np.array([game["i"] for game in games_list])

            # Now count how many of these are greater than 0,
            # and get the proportion relative to the total number of games.
            self.info_using_equilibria.append((i_values > 0).sum() / len(i_values))


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

    + Demo mode omits hard-to-find classes and places an upper limit on `games_per_c`.
      This allows it to run in a reasonable amount of time.
    + Full mode requires an existing set of game data stored in JSON files.
      These can be created via the functions `find_games_3x3()` and `analyse_games_3x3()`.

    The reason for demo mode is that the figure takes a VERY long time to create
    with the published parameter of `games_per_c=1500`.
    Realistically we need to prepare by finding `games_per_c` games for each value of C,
    storing them in a local JSON file, and calling them at runtime to count the equilibria.
    Demo mode omits games with `c=0.000` and `c=0.111` because they are especially hard to find.

    """

    def __init__(self, games_per_c=50, demo=True, dir_games="../data/") -> None:
        """
        Create an instance of the class.

        Parameters
        ----------
        games_per_c : int, optional
            Number of games per value of C to create. The default is 50.
        demo : bool, optional
            Whether to create the figure in demo mode or not.
            The default is True.
        dir_games : str, optional
            Directory path to find stored JSON game data.
            Only used if `demo=False`.
            The default is "../data/".
        """

        # Is it demo mode?
        if demo:
            # Warn user of demo mode
            self.demo_warning()

            # Set C values for demo mode
            self.c_values = c_3x3_equiprobable_demo

            # Create games in demo mode
            try:
                self.create_games_demo(games_per_c)
            except ex.ModuleNotInstalledException as e:
                print(e)
                return

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

        # Show the graph, with the line between the scatter points
        self.show_line = True
        self.show()

    def load_saved_games(self, dir_games, games_per_c) -> None:
        """
        Get sender and receiver matrices and load them into game objects.
        Put them into dictionary self.games.

        The games should already exist in dir_games with filenames of the form:

        `f"{dir_games}games_c{c_value:.3f}_n{games_per_c}.json"`

        Parameters
        ----------
        dir_games : str
            Directory containing JSON files with sender and receiver matrices.
        games_per_c : int
            Number of games per value of C.

        """

        # Game objects will be stored in a dictionary by C value.
        self.games = {f"{k:.3f}": [] for k in self.c_values}

        # Loop at C values...
        for c_value, games_list in self.games.items():
            # Get name of JSON file
            fpath_json = f"{dir_games}games_c{c_value}_n{games_per_c}.json"

            # The filename MUST have this format. Otherwise tell the user
            # they should use find_games_3x3() to find exactly this many games.
            try:
                with open(fpath_json, "r") as f:
                    games_list_loaded = json.load(f)

            except FileNotFoundError:
                raise ex.NoDataException(
                    f"File {fpath_json} was not found. Have you run find_games_3x3() and analyse_games_3x3() for games_per_c={games_per_c} yet?"
                )

            # Append these game to the figure object's game list.
            self.games[c_value] = games_list_loaded

    def create_games_demo(self, games_per_c) -> None:
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
            sender_payoff_matrix = get_random_payoffs()
            receiver_payoff_matrix = get_random_payoffs()

            # Check common interest
            c = calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix)

            # Skip 0 and 1/9 values in demo mode
            if f"{c:.3f}" == "0.000" or f"{c:.3f}" == "0.111":
                continue

            # Do we already have enough results for this value of C?
            if len(self.games[f"{c:.3f}"]) >= games_per_c:
                continue

            # This value of C still needs games.
            game = asy.Chance(
                state_chances,
                sender_payoff_matrix,
                receiver_payoff_matrix,
                messages,
            )

            # ...and associate it with this value of C.
            # We are just storing a dict with the key features of the game.
            # That's what calculate_results_per_c() expects.

            # Get the highest info-using equilibrium
            # and the mutual info between states and acts at that point.
            e, i = game.highest_info_using_equilibrium

            # Build the game dict and add it to self.games.
            game_dict = {
                "s": sender_payoff_matrix,
                "r": receiver_payoff_matrix,
                "e": e,
                "i": i,
            }
            self.games[f"{c:.3f}"].append(game_dict)

            # Do we have all the required games yet?
            games_added_to_count += 1

            # Report progress
            if total_games_required>100 and games_added_to_count % (np.floor(total_games_required / 100)) == 0:
                print(
                    f"Surveyed {total_games_surveyed} games; added {games_added_to_count} of {total_games_required}"
                )

            if games_added_to_count == total_games_required:
                break

    def calculate_results_per_c(self, games_per_c) -> None:
        """
        For each value of <self.c_values>, count how many out of <games_per_c> games
        have info-using equilibria.

        Parameters
        ----------
        games_per_c : int
            Number of games to generate per level of common interest.
        """

        # 1. Initialize
        self.highest_mi = []

        # 2. Loop at sorted C values...
        for c_value, games_list in tqdm(sorted(self.games.items())):
            # Get max mutual info at equilibrium
            self.highest_mi.append(max([game["i"] for game in games_list]))


class GodfreySmith2013_3(Surface):
    """
    See figure at https://doi.org/10.1371/journal.pcbi.1003282.g003

    This object requires an existing set of game data stored in JSON files.
    These can be created with `find_games_3x3_c_and_k()` and
    `analyse_games_3x3_c_and_k()`.
    See the section **How to use this script** for more.

    Demo mode is not yet available for this figure.
    """

    def __init__(
        self, games_per_c_and_k=150, k_indicator=None, demo=False, dir_games="../data/"
    ) -> None:
        """
        Constructor for GodfreySmith2013_3 object.

        Parameters
        ----------
        games_per_c_and_k : int, optional
            Number of games to analyse per combination of C value and K value.
            The default is 150.
        k_indicator : str, optional
            This MUST be set to either "ks" or "kr".
            Indicates whether the plot is created for sender values of K or receiver values.
            The default is None, forcing the user to specify the value.
        demo : bool, optional
            If True, runs in demo mode.
            The default is False.
        dir_games : str, optional
            Directory for stored game data. The default is "../data/".

        """

        if k_indicator:
            self.k_indicator = k_indicator

        # Set attributes
        self.games_per_c_and_k = games_per_c_and_k

        # Is it demo mode?
        if demo:
            # # Warn user of demo mode
            # self.demo_warning()

            # # Set C values for demo mode
            # self.c_values = c_3x3_equiprobable_demo

            # # K values
            # self.k_values = k_3x3

            # # Create games in demo mode
            # self.create_games_demo(games_per_c)

            raise NotImplementedError("Demo mode not available for this figure yet!")

        else:
            # Set C values for full mode
            self.c_values = c_3x3_equiprobable

            # K values
            self.k_values = k_3x3

            # Load games for full mode
            self.load_saved_games(dir_games)

        # The main calculation method.
        self.calculate_results_per_c_and_k()

        # There are no evolving populations in this figure.
        # All the calculations are static.
        super().__init__(evo=None)

        # Set the data for the graph
        self.reset(
            x=self.c_values,
            y=self.k_values / max(self.k_values),  # Normalise
            z=self.info_using_equilibria,
            xlabel="C",
            ylabel="K_S",
            zlabel="Proportion of games",
            xlim=[0, 1],
            ylim=[0, 1],
            zlim=[0, 1],
        )

        # Show the graph
        self.show()

        # Warn user about impossible combinations
        print(
            "Note that the following combinations of C and K are impossible and have been artificially set to zero:"
        )
        for k_value in k_3x3_excluded_at_c_0:
            print(f"C={0:.3f}, K={k_value:.3f}")

    def load_saved_games(self, dir_games) -> None:
        """
        Get sender and receiver matrices and load them into game objects.
        Put them into dictionary self.games.

        The games should already exist in dir_games with filenames of the form:

        `f"{dir_games}games_c{c_value:.3f}_{ks or kr}{k_value:.3f}_n{games_per_c_and_k}.json"`

        Parameters
        ----------
        dir_games : str
            Directory containing JSON files with sender and receiver matrices.

        """

        # Game objects will be stored in a dictionary by C and K value.
        self.games = {
            f"{c_value:.3f}_{k_value:.3f}": []
            for c_value in self.c_values
            for k_value in self.k_values
        }

        # Remove impossible combinations of C and K
        for k_value in k_3x3_excluded_at_c_0:
            # Define the game that is to be excluded.
            combination_to_exclude = f"{0:.3f}_{k_value:.3f}"

            # Delete that entry from <self.games>
            del self.games[combination_to_exclude]

        # Loop at C values...
        for value_string, games_list in self.games.items():
            # Get name of JSON file
            fpath_json = f"{dir_games}games_c{value_string[:5]}_{self.k_indicator}{value_string[6:]}_n{self.games_per_c_and_k}.json"

            # The filename MUST have this format. Otherwise tell the user
            # they should use find_games_3x3() to find exactly this many games.
            try:
                with open(fpath_json, "r") as f:
                    games_list_loaded = json.load(f)

            except FileNotFoundError:
                raise ex.NoDataException(
                    f"File {fpath_json} was not found. Have you run find_games_3x3() and analyse_games_3x3() for games_per_c_and_k={self.games_per_c_and_k} yet?"
                )

            # Load each game into an object
            for game_dict in games_list_loaded:
                # To avoid creating the game object (takes a long time),
                # just say whether there's an info-using equilibrium.
                if game_dict["i"] > 0:
                    self.games[value_string].append(True)

                if game_dict["i"] == 0:
                    self.games[value_string].append(False)

    def calculate_results_per_c_and_k(self) -> None:
        """
        For each pair of `self.c_values` and `self.k_values`, count how many games
        have info-using equilibria.
        """

        # 1. Initialize
        # TODO: get this from the keys of self.games.
        results = {
            f"{c_value:.3f}_{k_value:.3f}": []
            for c_value in self.c_values
            for k_value in self.k_values
        }

        # Set dummy data for impossible combinations of C and K.
        # We'll warn the user that this data is artificially set to zero.
        # Because of the need for this dummy data,
        # it's easier to use an intermediate dictionary <results>.
        for k_value in k_3x3_excluded_at_c_0:
            # Define the game that is to be excluded.
            combination_to_dummy = f"{0:.3f}_{k_value:.3f}"

            # Delete that entry from <self.games>
            results[combination_to_dummy] = np.zeros((self.games_per_c_and_k,)).tolist()

        # Now get the real data.
        # 2. Loop at combinations...
        for value_string, games_list in tqdm(self.games.items()):
            results[value_string] = games_list

        # Count the total number of info-using equilibria per combination of C and K
        self.info_using_equilibria = []

        # Loop helpers
        c_last = -1
        index = -1

        for key in sorted(results):  # for each level of C...
            # Is this a new value of C?
            # If so, we need to create a new row of the results matrix.
            if float(key[:5]) > c_last:
                # New row
                index += 1
                c_last = float(key[:5])
                self.info_using_equilibria.append([])

            # Now <index> is the index of the appropriate row
            self.info_using_equilibria[index].append(
                sum(results[key]) / len(results[key])
            )  # ...get the proportion of info-using equilibria.

        # Transpose and array-ify
        self.info_using_equilibria = np.array(self.info_using_equilibria).T


class GodfreySmith2013_3_sender(GodfreySmith2013_3):
    """
    Wrapper for GodfreySmith2013_3(),
    calling with parameter `self.k_indicator = "ks"`
    to create figure 3a.
    """

    def __init__(self, **kwargs):
        self.k_indicator = "ks"
        super().__init__(**kwargs)


class GodfreySmith2013_3_receiver(GodfreySmith2013_3):
    """
    Wrapper for GodfreySmith2013_3(),
    calling with parameter `self.k_indicator = "kr"`
    to create figure 3b.
    """

    def __init__(self, **kwargs):
        self.k_indicator = "kr"
        super().__init__(**kwargs)


"""
    Shared methods
"""


def calculate_D(payoff_matrix, state, act_1, act_2) -> float:
    """
    Calculate an agent's relative preference of acts `act_1` and `act_2`
    in state `state`.

    The measure is defined in the supplement of Godfrey-Smith and Martínez (2013), page 1.

    Parameters
    ----------
    payoff_matrix : array-like
        The agent's payoff matrix.
    state : int
        Index of the state.
    act_1 : int
        Index of the first act to be compared.
    act_2 : int
        Index of the second act to be compared.

    Returns
    -------
    D : float
        Godfrey-Smith and Martínez's measure D.   
        
        + 0   if act 1 is preferred
        + 0.5 if the payoffs are equal
        + 1   if act 2 is preferred

    """

    ## Sanity check
    assert act_1 != act_2

    ## Get the sign of the difference, then convert the scale [-1, 0, 1] to [0, 0.5, 1]
    ##  by adding 1 and dividing by 2.
    return (np.sign(payoff_matrix[state][act_2] - payoff_matrix[state][act_1]) + 1) / 2


def calculate_C(state_chances, sender_payoff_matrix, receiver_payoff_matrix) -> float:
    """

    Calculate C as per Godfrey-Smith and Martínez's definition.

    See page 2 of the supporting information at
    https://doi.org/10.1371/journal.pcbi.1003282.s001

    Returns
    -------
    c : float
        PGS & MM's measure C.

    """

    # It's only defined when the number of states is equal to the number of acts.
    assert len(state_chances) == len(sender_payoff_matrix[0])

    # Get the number of states
    n = len(state_chances)

    # Get the (j,k) pairs as defined in the supplement.
    pairs = list(combinations(range(n), r=2))

    # Get a 3D matrix where each COLUMN is a state,
    # there are always two ROWS comparing a pair of acts in that state,
    # and each AISLE/TUBE/SLICE iterates through pairs.
    sender_pairs = sender_payoff_matrix.T[np.array(pairs)]
    receiver_pairs = receiver_payoff_matrix.T[np.array(pairs)]

    # Now we say: for each pair of acts in a given state,
    # which act has the higher payoff?
    # The sign tells us which is higher, and it's 0 if there's a tie.
    sender_sign = np.sign(sender_pairs[:, 0] - sender_pairs[:, 1])
    receiver_sign = np.sign(receiver_pairs[:, 0] - receiver_pairs[:, 1])

    # Here we're doing two things at once.
    # First is np.abs(sender_sign - receiver_sign).
    # That's asking whether sender and receiver agree about
    # which act of a given pair is best in each state.
    # If they totally disagree, this value will be 2 (because the raw value is either +2 or -2).
    # So with the comparator "== 2" we get the value TRUE in entries corresponding to
    # pairs for which sender and receiver disagree on which is best.
    # Treating these TRUEs as numeric value 1, we multiply each by that state's probability.
    # Then we just do a big sum of them all.
    sum_total = np.sum(
        np.array(state_chances) * (np.abs(sender_sign - receiver_sign) == 2)
    )

    # Finally, we scale and convert the sum as per the definition in the supplement.
    subtractor = (2 * sum_total) / (n * (n - 1))
    c = 1 - subtractor

    return c


def calculate_Ks_and_Kr(sender_payoff_matrix, receiver_payoff_matrix):
    """
    Calculate the extent to which an agent's preference ordering
    over receiver actions varies with the state of the world.

    Defined as `K_S` and `K_R` in the supplement of Godfrey-Smith and Martínez (2013), page 2.

    Parameters
    ----------
    payoff_matrix : array-like
        The agent's payoff matrix.

    Returns
    -------
    K : float

    """

    # It's only defined when the number of states is equal to the number of acts.
    assert len(sender_payoff_matrix) == len(sender_payoff_matrix[0])

    n = len(sender_payoff_matrix)

    # ## 1. Initialise
    sum_total_sender = 0
    sum_total_receiver = 0

    # Loop around pairs of states and pairs of acts
    # All states except the first (with index 0)
    for i in range(1, len(sender_payoff_matrix)):
        # All states with a lower index than i
        for j in range(i):
            # All acts except the first (with index 0)
            for k in range(1, len(sender_payoff_matrix[i])):
                # All acts with a lower index than k
                for l in range(k):
                    # Get the sender and receiver D-measures

                    # SENDER

                    # Calculate sender's d-components
                    d_sender_component_1 = calculate_D(sender_payoff_matrix, i, k, l)
                    d_sender_component_2 = calculate_D(sender_payoff_matrix, j, k, l)

                    # Get this component of the sum
                    factor_sender = np.floor(
                        abs(d_sender_component_1 - d_sender_component_2)
                    )

                    # Add to sender total
                    # The definition has the multiplication factor inside the sum.
                    # It doesn't really matter, but we'll put it here for consistency.
                    sum_total_sender += (2 * factor_sender) / (n * (n - 1))

                    # RECEIVER

                    # Calculate receiver's d-component
                    d_receiver_component_1 = calculate_D(
                        receiver_payoff_matrix, i, k, l
                    )
                    d_receiver_component_2 = calculate_D(
                        receiver_payoff_matrix, j, k, l
                    )

                    # Get this component of the sum
                    factor_receiver = np.floor(
                        abs(d_receiver_component_1 - d_receiver_component_2)
                    )

                    # Add to receiver total
                    # The definition has the multiplication factor inside the sum.
                    # It doesn't really matter, but we'll put it here for consistency.
                    sum_total_receiver += (2 * factor_receiver) / (n * (n - 1))

    # Return both
    # Note that these are NOT NORMALISED.
    # For a range of values of K_S and K_R, you have to normalise them manually
    #  by finding the maximum value.
    return sum_total_sender, sum_total_receiver


def calculate_Ks_and_Kr_from_game(game):
    return calculate_Ks_and_Kr(game.sender_payoff_matrix, game.receiver_payoff_matrix)


def find_games_3x3(
    games_per_c=1500, c_values=c_3x3_equiprobable, dir_games="../data/"
) -> None:
    """
    Finds `games_per_c` 3x3 sender and receiver matrices
    and saves them as JSON files, storing them by C value in `dir_games`.

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

    # Initialise
    # Values of C in truncated string format
    c_outstanding = {f"{k:.3f}" for k in c_values}

    # Dict to conveniently store the matrices before outputting.
    results = {f"{k:.3f}": [] for k in c_values}

    # State chances (required to calculate C)
    state_chances = np.array([1 / 3, 1 / 3, 1 / 3])

    # While there are values of C that have not yet had all games found and saved...
    while len(c_outstanding) > 0:
        # Create a game
        sender_payoff_matrix = get_random_payoffs()
        receiver_payoff_matrix = get_random_payoffs()

        # Check common interest
        c_value = calculate_C(
            state_chances, sender_payoff_matrix, receiver_payoff_matrix
        )

        # Does this value of C require saving?
        if f"{c_value:.3f}" in c_outstanding:
            # If yes, add to the output dict.
            # Each value in the output dict is a list of dicts (i.e. a list of games)
            # with this format:
            ##
            # {
            # "s": <sender payoff matrix>
            # "r": <receiver payoff matrix>
            # "e": <equilibrium with the highest information transmission>
            # "i": <mutual information between states and acts at this equilibrium>
            # }
            ##
            # The values for e and i will be left blank here.
            # The function analyse_games_3x3() will fill them in.

            # Create game dict.
            results[f"{c_value:.3f}"].append(
                {
                    "s": sender_payoff_matrix.tolist(),
                    "r": receiver_payoff_matrix.tolist(),
                }
            )

            # Has this value of C now had all its games found?
            if len(results[f"{c_value:.3f}"]) >= games_per_c:
                # If yes, save JSON file and remove this value of C from the "to-do" list.
                fpath_out = f"{dir_games}games_c{c_value:.3f}_n{games_per_c}.json"

                with open(fpath_out, "w") as f:
                    json.dump(results[f"{c_value:.3f}"], f)

                # Remove this value of C from the "to-do" list
                c_outstanding.remove(f"{c_value:.3f}")

                # Are there any values of C left outstanding? If not, quit the while-loop.
                if len(c_outstanding) == 0:
                    break

                # Otherwise, print remaining values
                print(f"C values remaining: {c_outstanding}")


def analyse_games_3x3(
    games_per_c=1500, c_values=c_3x3_equiprobable, dir_games="../data/", sigfig=5
) -> None:
    """
    Find information-using equilibria of 3x3 games
    and the mutual information between states and acts at those equilibria.

    The games should already exist in `dir_games` with filenames of the form:

    ``f"{dir_games}games_c{c_value:.3f}_n{games_per_c}.json"``

    Each file should be a list of dicts. Each dict corresponds to a game:

    ``{
    "s": <sender payoff matrix>
    "r": <receiver payoff matrix>
    "e": <equilibrium with the highest information transmission>
    "i": <mutual information between states and acts at this equilibrium>
    }``

    ``s`` and ``r`` already exist; this function fills in ``e`` and ``i``.

    Parameters
    ----------
    dir_games : str
        Directory to find and store JSON files
    games_per_c : int, optional
        Number of games to find per value of c. The default is 1500.
    c_values : array-like
        List of C values to find games for.
        The default is the global variable c_3x3_equiprobable.
    sigfig : int, optional.
        The number of significant figures to report values in.
        Since gambit sometimes has problems rounding, it generates values like 0.9999999999996.
        We want to report these as 1.0000, especially if we're dumping to a file.
        The default is 5.
    """

    # Game objects will be stored in a dictionary by C value.
    games = {f"{c_value:.3f}": [] for c_value in c_values}

    # State chances and messages are always the same.
    state_chances = np.array([1 / 3, 1 / 3, 1 / 3])
    messages = 3

    # Loop at C values...
    for c_value, games_list in tqdm(games.items()):
        # Get name of JSON file
        fpath_json = f"{dir_games}games_c{c_value}_n{games_per_c}.json"

        # The filename MUST have this format. Otherwise tell the user
        # they should use find_games_3x3() to find exactly this many games.
        with open(fpath_json, "r") as f:
            games_list_loaded = json.load(f)

        # Load each game into an object
        for game_dict in games_list_loaded:
            # Create the game
            game = asy.Chance(
                state_chances=state_chances,
                sender_payoff_matrix=np.array(game_dict["s"]),
                receiver_payoff_matrix=np.array(game_dict["r"]),
                messages=messages,
            )

            # Is there an info-using equilibrium?
            try:
                game_dict["e"], game_dict["i"] = game.highest_info_using_equilibrium
            except ex.ModuleNotInstalledException as e:
                print(e)
                return

        # Dump this updated game file
        with open(fpath_json, "w") as f:
            json.dump(games_list_loaded, f)


"""
    Find and analyse games by K value as well as by C value
"""


def find_games_3x3_c_and_k(
    games_per_c_and_k=1500,
    sender=True,
    c_values=c_3x3_equiprobable,
    k_values=k_3x3,
    dir_games="../data/",
) -> None:
    """
    Finds ``games_per_c_and_k`` 3x3 sender and receiver matrices
    and saves them as JSON files, storing them by C and K values in ``dir_games``.

    Note that it is EXTREMELY difficult to find games for some combinations
    of C and K, especially when C=0.
    Expect this to take a long time!

    Since it's hard to find games for certain combinations of C and K, we'll save each
    JSON file individually once we've found it.
    Then if you have to terminate early, you can come back and just search for
    games with the combinations of C and K you need later on.

    Parameters
    ----------
    games_per_c_and_k : int, optional
        Number of games to find per pair of c and k. The default is 1500.
    sender: bool
        If True, the operative value of K is the sender's K_S
        If False, the operative value of K is the receiver's K_R
    c_values : array-like
        List of C values to find games for.
        The default is the global variable c_3x3_equiprobable.
    k_values : array-like
        List of K values to find games for.
        The default is the global variable k_3x3.
    dir_games : str
        Directory to place JSON files
    """

    # Initialise
    # Values of C in truncated string format
    c_outstanding = {f"{c_value:.3f}": 0 for c_value in c_values}
    games_outstanding = {
        f"{c_value:.3f}_{k_value:.3f}" for c_value in c_values for k_value in k_values
    }

    # Dict to conveniently store the matrices before outputting.
    results = {
        f"{c_value:.3f}_{k_value:.3f}": []
        for c_value in c_values
        for k_value in k_values
    }

    # Remove excluded combinations of C and K
    for k_value in k_3x3_excluded_at_c_0:
        # Define the game that is to be excluded.
        combination_to_exclude = f"{0:.3f}_{k_value:.3f}"

        # Delete that entry from <games_outstanding>
        games_outstanding.remove(combination_to_exclude)

        # Delete that entry from <results>
        del results[combination_to_exclude]

    # State chances (required to calculate C)
    state_chances = np.array([1 / 3, 1 / 3, 1 / 3])

    # While there are values of C and K that have not yet had all games found and saved...
    while len(games_outstanding) > 0:
        # Create a game
        sender_payoff_matrix = get_random_payoffs()
        receiver_payoff_matrix = get_random_payoffs()

        # Check common interest
        c_value = calculate_C(
            state_chances, sender_payoff_matrix, receiver_payoff_matrix
        )

        # Does this value of C require saving?
        if f"{c_value:.3f}" in c_outstanding:
            # If yes, calculate the k values.
            k_sender, k_receiver = calculate_Ks_and_Kr(
                sender_payoff_matrix, receiver_payoff_matrix
            )

            # Which k-value are we using now?
            k_value = k_sender if sender else k_receiver

            # Does this combination of C and K require saving?
            if f"{c_value:.3f}_{k_value:.3f}" in games_outstanding:
                # If yes, add to the output dict.
                # Each value in the output dict is a list of dicts (i.e. a list of games)
                # with this format:
                ##
                # {
                # "s": <sender payoff matrix>
                # "r": <receiver payoff matrix>
                # "e": <equilibrium with the highest information transmission>
                # "i": <mutual information between states and acts at this equilibrium>
                # }
                ##
                # The values for e and i will be left blank here.
                # The function analyse_games_3x3_c_and_k() will fill them in.

                # Create game dict.
                results[f"{c_value:.3f}_{k_value:.3f}"].append(
                    {
                        "s": sender_payoff_matrix.tolist(),
                        "r": receiver_payoff_matrix.tolist(),
                    }
                )

                # Has this combination of C and K now had all its games found?
                if len(results[f"{c_value:.3f}_{k_value:.3f}"]) >= games_per_c_and_k:
                    # If yes, save JSON file and remove this value of C from the "to-do" list.
                    fpath_indicator = "ks" if sender else "kr"
                    fpath_out = f"{dir_games}games_c{c_value:.3f}_{fpath_indicator}{k_value:.3f}_n{games_per_c_and_k}.json"

                    with open(fpath_out, "w") as f:
                        json.dump(results[f"{c_value:.3f}_{k_value:.3f}"], f)

                    # Remove this combination of C and K from the "to-do" list
                    games_outstanding.remove(f"{c_value:.3f}_{k_value:.3f}")

                    # Add these games to c_outstanding
                    c_outstanding[f"{c_value:.3f}"] += games_per_c_and_k

                    # Has this value of C now had all its games found?
                    # For C=0 some combinations are excluded.
                    # That's why the second disjunct is slightly more complicated.
                    if (
                        c_value > 0
                        and c_outstanding[f"{c_value:.3f}"]
                        >= games_per_c_and_k * len(k_values)
                    ) or (
                        c_value == 0
                        and c_outstanding[f"{c_value:.3f}"]
                        >= games_per_c_and_k
                        * (len(k_values) - len(k_3x3_excluded_at_c_0))
                    ):
                        # If yes, remove it
                        del c_outstanding[f"{c_value:.3f}"]

                        # Are there any combinations of C and K left outstanding? If not, quit the while-loop.
                        if len(c_outstanding) == 0:
                            break

                    # Otherwise, print remaining values
                    print(
                        f"C values found (out of {games_per_c_and_k * len(k_values)}): {c_outstanding}"
                    )

    # Warn the user of impossible combinations
    print(
        "Note that the following combinations of C and K are impossible and have been excluded:"
    )
    for k_value in k_3x3_excluded_at_c_0:
        print(f"C={0:.3f}, K={k_value:.3f}")


def analyse_games_3x3_c_and_k(
    games_per_c_and_k=1500,
    sender=True,
    c_values=c_3x3_equiprobable,
    k_values=k_3x3,
    dir_games="../data/",
    sigfig=5,
) -> None:
    """
    Find information-using equilibria of 3x3 games
    and the mutual information between states and acts at those equilibria.

    The games should already exist in dir_games with filenames of the form:

    ``f"{dir_games}games_c{c_value:.3f}_{ks or kr}{k_value:.3f}_n{games_per_c}.json"``

    Each file should be a list of dicts. Each dict corresponds to a game:

    ``{
    "s": <sender payoff matrix>
    "r": <receiver payoff matrix>
    "e": <equilibrium with the highest information transmission>
    "i": <mutual information between states and acts at this equilibrium>
    }``

    s and r already exist; this function fills in e and i.

    Parameters
    ----------
    games_per_c_and_k : int, optional
        Number of games to analyse per pair of c and k. The default is 1500.
    sender: bool
        If True, the operative value of K is the sender's K_S
        If False, the operative value of K is the receiver's K_R
    c_values : array-like
        List of C values to analyse games for.
        The default is the global variable c_3x3_equiprobable.
    k_values : array-like
        List of K values to analyse games for.
        The default is the global variable k_3x3.
    dir_games : str
        Directory to find and update JSON files
    sigfig : int, optional.
        The number of significant figures to report values in.
        Since gambit sometimes has problems rounding, it generates values like 0.9999999999996.
        We want to report these as 1.0000, especially if we're dumping to a file.
        The default is 5.
    """

    # Game objects will be stored in a dictionary by C value.
    games = {
        f"{c_value:.3f}_{k_value:.3f}": []
        for c_value in c_values
        for k_value in k_values
    }

    # Exclude impossible combinations
    # Remove excluded combinations of C and K
    for k_value in k_3x3_excluded_at_c_0:
        # Define the game that is to be excluded.
        combination_to_exclude = f"{0:.3f}_{k_value:.3f}"

        # Delete that entry from <games>
        del games[combination_to_exclude]

    # State chances and messages are always the same.
    state_chances = np.array([1 / 3, 1 / 3, 1 / 3])
    messages = 3

    # Loop at C values...
    for value_string, games_list in tqdm(games.items()):
        # Get name of JSON file
        fpath_indicator = "ks" if sender else "kr"
        fpath_json = f"{dir_games}games_c{value_string[:5]}_{fpath_indicator}{value_string[6:]}_n{games_per_c_and_k}.json"

        # The filename MUST have this format. Otherwise tell the user
        # they should use find_games_3x3_c_and_k() to find exactly this many games.
        with open(fpath_json, "r") as f:
            games_list_loaded = json.load(f)

        # Load each game into an object
        for game_dict in games_list_loaded:
            # Create the game
            game = asy.Chance(
                state_chances=state_chances,
                sender_payoff_matrix=np.array(game_dict["s"]),
                receiver_payoff_matrix=np.array(game_dict["r"]),
                messages=messages,
            )

            # Is there an info-using equilibrium?
            try:
                game_dict["e"], game_dict["i"] = game.highest_info_using_equilibrium
            except ex.ModuleNotInstalledException as e:
                print(e)
                return

        # Dump this updated game file
        with open(fpath_json, "w") as f:
            json.dump(games_list_loaded, f)


"""
    Helper functions specific to this script
"""


def get_random_payoffs(states=3, acts=3, min_payoff=0, max_payoff=100):
    """
    Generate a random payoff matrix.

    Parameters
    ----------
    states : int, optional
        Number of states observable by the sender. The default is 3.
    acts: int, optional.
        Number of acts available to the receiver.
    min_payoff : int, optional
        Smallest possible payoff. The default is 0.
    max_payoff : int, optional
        Largest possible payoff. The default is 100.

    Returns
    -------
    payoffs : array-like
        A random payoff matrix of shape (states,acts).

    """

    return np.random.randint(min_payoff, max_payoff, (states, acts))
