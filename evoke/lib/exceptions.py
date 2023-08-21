"""
Some custom exceptions and errors
"""


class ChanceNodeError(Exception):
    """
    Error to raise when the user is attempting to do something with a chance
    node that doesn't exist
    """

    pass

class NoDataException(Exception):
    """
    Error to raise when the user tries to show a plot but the figure object
    doesn't have the required data.
    """
    
    pass