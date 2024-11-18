# -*- coding: utf-8 -*-
"""
    Figure objects.
    
    Figures can be plotted by calling an instance of the relevant class,
     e.g. f = Skyrms2010_3_3() will create a figure object and simultaneously plot it.
    
"""

import numpy as np
from itertools import combinations
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from matplotlib import cm

## 3D plotting
# from mpl_toolkits.mplot3d import Axes3D

## ternary plots
from ternary import figure  # from https://github.com/marcharper/python-ternary

# from tqdm import tqdm, trange


# from evoke.lib import asymmetric_games as asy
# from evoke.lib import symmetric_games as sym
# from evoke.lib import evolve as ev
# from evoke.lib.symmetric_games import NoSignal
import evoke.src.exceptions as ex


class Figure(ABC):
    """
    Abstract superclass for all figures.
    """

    def __init__(self, evo=None, game=None, **kwargs):  # Evolve object
        # If there's an evolve object, set it as a class attribute
        if evo is not None:
            self.evo = evo

        # If there's a game object, set it as a class attribute
        if game is not None:
            self.game = game

        ## Set keyword arguments as class attributes.
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def show(self):
        """
        Show the plot of the figure with the current parameters, typically with plt.show().

        This is an abstract method that must be redefined for each subclass.

        Returns
        -------
        None.

        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset parameters for the figure.

        This is an abstract method that must be redefined for each subclass.

        Returns
        -------
        None.

        """
        pass

    @classmethod
    def demo_warning(cls):
        """
        Warn the user that they are running in demo mode.

        Returns
        -------
        None.

        """

        print(
            f"Note: figure will be created in demo mode. To create the full figure use {cls.__name__}(demo=False)."
        )

    @property
    def properties(self):
        """
        Get a dict of all the editable properties of the figure, with their current values.

        Typically includes properties like plot color, axis labels, plot scale etc.

        Returns
        -------
        list_of_properties : list
            A list of the editable properties of the object.

        """

        # Lazy instantiation of the property names
        if not hasattr(self, "_list_of_properties"):
            # Initialise
            self._list_of_properties = []

            # We need to check properties defined by the class itself,
            # as well as all properties defined by its superclasses,
            # up to and including the Figure class.

            # We need to find where the Figure class is in the hierarchy of superclasses.
            # First we need to create a list of class names in that hierarchy.
            superclass_list = [c.__name__ for c in type(self).__mro__]

            # Then get the index of the string "Figure" in that list of names.
            figure_location = superclass_list.index("Figure")

            # Now start from the base class and step through each superclass.
            for i in range(figure_location + 1):
                # Add this superclass's properties to the big list of properties.
                # In future we might need to do further tweaking here
                # to exclude properties that aren't in fact editable.
                # For now I'm assuming all properties created with the @property decorator
                # are intended as user-editable properties of the figure plot.
                self._list_of_properties.extend(
                    [
                        k
                        for k, v in vars(type(self).__mro__[i]).items()
                        if isinstance(v, property)
                    ]
                )

            # Omit the "properties" property!
            self._list_of_properties.remove("properties")

            self._list_of_properties = sorted(self._list_of_properties)

        # Current property values
        # Create a dict from this list, including the current values.
        dict_of_properties = {k: getattr(self, k) for k in self._list_of_properties}

        return dict_of_properties


class Scatter(Figure):
    """
    Superclass for scatter plots
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(
        self,
        x,
        y,
        xlabel,
        ylabel,
        marker_size=10,
        marker_color="k",
        xlim=None,
        ylim=None,
        xscale=None,
        yscale=None,
    ):
        """
        Update figure parameters, which can then be plotted with self.show().

        Parameters
        ----------
        x : array-like
            x-axis coordinates.
        y : array-like
            y-axis coordinates.
        xlabel : str
            x-axis label.
        ylabel : str
            y-axis label.
        marker_size : int, optional
            Size of the markers for each data point. The default is 10.
        marker_color : str, optional
            Color of the datapoint markers. The default is "k".
        xlim : array-like, optional
            Minimum and maximum values of x-axis. The default is None.
        ylim : array-like, optional
            Minimum and maximum values of y-axis. The default is None.
        xscale : str, optional
            x-axis scaling i.e. linear or logarithmic. The default is None.
        yscale : str, optional
            y-axis scaling i.e. linear or logarithmic. The default is None.

        Returns
        -------
        None.

        """

        ## X and Y axis values
        self.x = x
        self.y = y

        ## Labels
        self.xlabel = xlabel
        self.ylabel = ylabel

        ## Marker design
        self.s = marker_size
        self.c = marker_color

        ## Limits of axes
        self.xlim = xlim
        self.ylim = ylim

        ## Axes Scaling
        self.xscale = xscale
        self.yscale = yscale

    def show(self):
        """
        Show figure with the current parameters.

        Parameters
        ----------
        line : bool, optional
            Whether to show a line connecting the datapoints.
            The default is False.

        Returns
        -------
        None.

        """

        # Check data exists
        if not hasattr(self, "x") or self.x is None:
            raise ex.NoDataException("Axis X has no data")
        if not hasattr(self, "y") or self.y is None:
            raise ex.NoDataException("Axis Y has no data")

        ## Data
        plt.scatter(x=self.x, y=self.y, s=self.s, c=self.c)

        if self.show_line:
            plt.plot(self.x, self.y, color=self.c)

        ## Labels
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        ## Limits of axes
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.ylim is not None:
            plt.ylim(self.ylim)

        ## Axes Scale
        if self.xscale is not None:
            plt.xscale(self.xscale)
        if self.yscale is not None:
            plt.yscale(self.yscale)

        ## Show plot
        plt.show()

        # If this is the first time this method has been called,
        # we want to allow the user to change cosmetic properties
        # and show the plot immediately when those properties are changed.
        # Therefore, if the show_immediately flag has not yet been created,
        # create it here and set it to True.
        if not hasattr(self, "show_immediately"):
            self.show_immediately = True

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

        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

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

        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

    @c.deleter
    def c(self):
        del self._c

    # Alias
    marker_color = c

    """
        Line connecting the markers
    """

    @property
    def show_line(self):
        # Lazy instantiation: default is False
        if not hasattr(self, "_show_line"):
            self._show_line = False

        return self._show_line

    @show_line.setter
    def show_line(self, inp):
        self._show_line = inp

        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

    @show_line.deleter
    def show_line(self):
        del self._show_line


class Quiver(Figure):
    """
    Superclass for Quiver plots
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    """ Property: marker color """

    @property
    def color(self):
        # Lazy instantiation: default to black
        if not hasattr(self, "_color"):
            self._color = "k"

        return self._color

    @color.setter
    def color(self, color):
        self._color = color

        # Update automatically?
        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

    @color.deleter
    def color(self):
        del self._color

    """ Property: x-axis label """

    @property
    def xlabel(self):
        # Lazy instantiation: default to None
        if not hasattr(self, "_xlabel"):
            self._xlabel = None

        return self._xlabel

    @xlabel.setter
    def xlabel(self, xlabel):
        self._xlabel = xlabel

        # Update automatically?
        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

    @xlabel.deleter
    def xlabel(self):
        del self._xlabel

    """ Property: y-axis label """

    @property
    def ylabel(self):
        # Lazy instantiation: default to None
        if not hasattr(self, "_ylabel"):
            self._ylabel = None

        return self._ylabel

    @ylabel.setter
    def ylabel(self, ylabel):
        self._ylabel = ylabel

        # Update automatically?
        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

    @ylabel.deleter
    def ylabel(self):
        del self._ylabel

    """ Property: axis toggle """

    @property
    def noaxis(self):
        # Lazy instantiation: default to None
        if not hasattr(self, "_noaxis"):
            self._noaxis = None

        return self._noaxis

    @noaxis.setter
    def noaxis(self, noaxis):
        self._noaxis = noaxis

        # Update automatically?
        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

    @noaxis.deleter
    def noaxis(self):
        del self._noaxis


class Quiver2D(Quiver):
    """
    Plot a 2D quiver plot.
    """

    def __init__(self, scale=20, **kwargs):
        self.scale = scale

        super().__init__(**kwargs)

    def reset(self, color=None, xlabel=None, ylabel=None):
        # Set global attributes based on what was supplied
        if color:
            self.color = color

        if xlabel:
            self.xlabel = xlabel

        if ylabel:
            self.ylabel = ylabel

    def show(self):
        """
        Display the 2D quiver plot with the loaded data.

        Raises
        ------
        NoDataException
            The requisite data has not been supplied by the user.

        Returns
        -------
        None.

        """

        # Check data exists
        if not hasattr(self, "X") or self.X is None:
            raise ex.NoDataException("Axis X has no data")
        if not hasattr(self, "Y") or self.Y is None:
            raise ex.NoDataException("Axis Y has no data")
        if not hasattr(self, "U") or self.U is None:
            raise ex.NoDataException("Velocities U do not exist")
        if not hasattr(self, "V") or self.V is None:
            raise ex.NoDataException("Velocities V do not exist")

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
            scale=self.scale,
            color=self.color,
        )

        # Axis labels
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)

        # Include axis?
        if self.noaxis:
            ax.set_axis_off()

        plt.show()

        # If this is the first time this method has been called,
        # we want to allow the user to change cosmetic properties
        # and show the plot immediately when those properties are changed.
        # Therefore, if the show_immediately flag has not yet been created,
        # create it here and set it to True.
        if not hasattr(self, "show_immediately"):
            self.show_immediately = True

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

        ##  This method accepts the proportion of the first strategy
        ##  and returns the velocities of the second.
        senders = np.array([x, 1 - x])
        receivers = np.array([y, 1 - y])

        new_pop_vector = self.evo.discrete_replicator_delta_X(
            np.concatenate((senders, receivers))
        )
        new_senders, new_receivers = self.evo.vector_to_populations(new_pop_vector)
        return (1 - x) - new_senders[1], (1 - y) - new_receivers[1]


class Quiver3D(Quiver):
    """
    Plot a 3D quiver plot.
    """

    def __init__(
        self,
        color="k",
        normalize=True,
        length=0.5,
        arrow_length_ratio=0.5,
        pivot="middle",
        **kwargs,
    ):
        self.color = color
        self.normalize = normalize
        self.length = length
        self.arrow_length_ratio = arrow_length_ratio
        self.pivot = pivot

        super().__init__(**kwargs)

    def show(self):
        """
        Display the 3D quiver plot with the loaded data.

        Raises
        ------
        NoDataException
            The requisite data has not been supplied.

        Returns
        -------
        None.

        """

        # Check data exists
        if not hasattr(self, "X") or self.X is None:
            raise ex.NoDataException("Axis X has no data")
        if not hasattr(self, "Y") or self.Y is None:
            raise ex.NoDataException("Axis Y has no data")
        if not hasattr(self, "Z") or self.Z is None:
            raise ex.NoDataException("Axis Z has no data")
        if not hasattr(self, "U") or self.U is None:
            raise ex.NoDataException("Velocities U do not exist")
        if not hasattr(self, "V") or self.V is None:
            raise ex.NoDataException("Velocities V do not exist")
        if not hasattr(self, "W") or self.W is None:
            raise ex.NoDataException("Velocities W do not exist")

        ## Create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ## Parameters at https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.quiver.html#mpl_toolkits.mplot3d.axes3d.Axes3D.quiver
        ax.quiver(
            self.X,
            self.Y,
            self.Z,
            self.U,
            self.V,
            self.W,
            color=self.color,
            normalize=self.normalize,
            length=self.length,
            arrow_length_ratio=self.arrow_length_ratio,
            pivot=self.pivot,
        )

        ax.set_xlim([-0.05, 2.1])
        ax.set_ylim([-0.05, 3.05])
        ax.set_zlim([-0.05, 3.05])

        # Display axis?
        if hasattr(self, "noaxis") and self.noaxis:
            ax.set_axis_off()

            # Camera distance.
            ax.dist = 10

        else:
            # There are axes. Are there axis labels?
            # Axis labels
            if hasattr(self, "xlabel"):
                ax.set_xlabel(self.xlabel)
            if hasattr(self, "ylabel"):
                ax.set_ylabel(self.ylabel)
            if hasattr(self, "zlabel"):
                ax.set_zlabel(self.zlabel)

            # Camera distance.
            ax.dist = 13

        ## Tetrahedron lines
        if hasattr(self, "vertices"):
            lines = combinations(self.vertices, 2)
            i = 0
            for x in lines:
                i += 1
                line = np.transpose(np.array(x))

                ## Make the back line a double dash
                linestyle = "--" if i == 5 else "-"

                ## https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot3D
                ax.plot3D(
                    line[0], line[1], line[2], c="0", linestyle=linestyle, linewidth=0.8
                )

        # Camera properties
        # self.elev=25.
        # self.azim=245
        # self.dist=12

        # # Camera angle.
        # if self.elev and self.azim: ax.view_init(elev=self.elev, azim=self.azim)

        plt.show()

        # If this is the first time this method has been called,
        # we want to allow the user to change cosmetic properties
        # and show the plot immediately when those properties are changed.
        # Therefore, if the show_immediately flag has not yet been created,
        # create it here and set it to True.
        if not hasattr(self, "show_immediately"):
            self.show_immediately = True

    def vector_to_barycentric(self, vector):
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

    """ Property: z-axis label """

    @property
    def zlabel(self):
        # Lazy instantiation: default to None
        if not hasattr(self, "_zlabel"):
            self._zlabel = None

        return self._zlabel

    @zlabel.setter
    def zlabel(self, zlabel):
        self._zlabel = zlabel

        # Update automatically?
        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

    @zlabel.deleter
    def zlabel(self):
        del self._zlabel


class Bar(Figure):
    """
    Bar chart abstract superclass.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(
        self, x, y, xlabel, ylabel, bar_color="w", xlim=None, ylim=None, yscale=None
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
        self.c = bar_color

        ## Limits of axes
        self.xlim = xlim
        self.ylim = ylim

        ## Axes Scaling
        self.yscale = yscale

    def show(self):
        """
        Display the bar chart with the loaded data.

        Raises
        ------
        NoDataException
            The requisite data has not been supplied.
            Bar charts need x-axis values and y-axis values.

        Returns
        -------
        None.

        """

        # Check data exists
        if not hasattr(self, "x") or self.x is None:
            raise ex.NoDataException("Axis X has no data")
        if not hasattr(self, "y") or self.y is None:
            raise ex.NoDataException("Axis Y has no data")

        ## Data
        plt.bar(x=self.x, height=self.y, color=self.c, edgecolor="k")

        ## Labels
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        ## Limits of axes
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.ylim is not None:
            plt.ylim(self.ylim)

        ## Axes Scale
        if self.yscale is not None:
            plt.yscale(self.yscale)

        ## Show plot
        plt.show()

        # If this is the first time this method has been called,
        # we want to allow the user to change cosmetic properties
        # and show the plot immediately when those properties are changed.
        # Therefore, if the show_immediately flag has not yet been created,
        # create it here and set it to True.
        if not hasattr(self, "show_immediately"):
            self.show_immediately = True

    """
        BAR ATTRIBUTES AND ALIASES
    """

    """
        Bar color
    """

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, inp):
        self._c = inp

        if hasattr(self, "show_immediately") and self.show_immediately:
            self.show()

    @c.deleter
    def c(self):
        del self._c

    # Alias
    bar_color = c


class Ternary(Figure):
    """
    Superclass for ternary (2-simplex) plots
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, right_corner_label, top_corner_label, left_corner_label, fontsize):
        ## Update global attributes, which can then be plotted in self.show()

        self.fontsize = fontsize
        self.right_corner_label = right_corner_label
        self.top_corner_label = top_corner_label
        self.left_corner_label = left_corner_label

    def show(self):
        """
        Display the ternary plot with the loaded data.

        Raises
        ------
        NoDataException
            The requisite data was not supplied.
            Ternary plots require an <xyzs> attribute.

        Returns
        -------
        None.

        """

        # Check data exists
        if not hasattr(self, "xyzs") or self.xyzs is None:
            raise ex.NoDataException("Ternary axis values not supplied.")

        # self.xyzs is a list of arrays of dimensions nx3, such that each row is a
        # 3-dimensional stochastic vector.  That is to say, for now, a collection
        # of orbits

        _, tax = figure()

        ## Data
        for xyz in self.xyzs:
            tax.plot(xyz, color="black")

        ## Titles, etc
        tax.right_corner_label(self.right_corner_label, fontsize=self.fontsize)
        tax.top_corner_label(self.top_corner_label, fontsize=self.fontsize)
        tax.left_corner_label(self.left_corner_label, fontsize=self.fontsize)

        ## No ticks or axes
        tax.get_axes().axis("off")
        tax.boundary(linewidth=0.5)
        tax.clear_matplotlib_ticks()

        ## Show plot
        tax.show()


class Surface(Figure):
    """
    Superclass for 3D surface plots (e.g. colormap).
    Uses ax.plot_surface().
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Data parameters
        self.x = None
        self.y = None
        self.z = None

    def reset(
        self,
        x=None,
        y=None,
        z=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        xlim=None,
        ylim=None,
        zlim=None,
        cmap=cm.coolwarm,
        linewidth=1,
        antialiased=False,
        elev=25.0,
        azim=245,
        dist=12,
    ) -> None:
        """
        Update figure parameters, which can then be plotted with self.show().

        Parameters
        ----------
        x : array-like
            x-axis values.
        y : array-like
            y-axis values.
        z : array-like
            z-axis values.
        xlim : array-like, optional
            Minimum and maximum values of x-axis. The default is None.
        ylim : array-like, optional
            Minimum and maximum values of y-axis. The default is None.
        zlim : array-like, optional
            Minimum and maximum values of z-axis. The default is None.
        cmap : matplotlib.colors.LinearSegmentedColormap, optional
            Color mapping. The default is cm.coolwarm.
        linewidth : float(?) or int, optional
            Width of the lines in the surface. The default is 1.
        antialiased : bool, optional
            Whether the figure is antialiased. The default is False.
        elev : float
            camera elevation
        azim : int
            camera azimuth
        dist : int
            camera distance

        Returns
        -------
        None.

        """

        # Data. Only update if supplied.
        if x is not None:
            self.x = np.array(x)
        if y is not None:
            self.y = np.array(y)
        if z is not None:
            self.z = np.array(z)

        # Need to meshgrid it
        self.x, self.y = np.meshgrid(self.x, self.y)

        # Axis labels
        if xlabel:
            self.xlabel = xlabel
        if ylabel:
            self.ylabel = ylabel
        if zlabel:
            self.zlabel = zlabel

        # Axis limits
        if xlim is not None:
            self.xlim = xlim
        if ylim is not None:
            self.ylim = ylim
        if zlim is not None:
            self.zlim = zlim

        # Cosmetic
        self.cmap = cmap
        self.linewidth = linewidth
        self.antialiased = antialiased

        # Camera angle
        self.elev = elev
        self.azim = azim
        self.dist = dist

    def show(self) -> None:
        """
        Show figure with current parameters.

        Returns
        -------
        None.

        """

        # Check data exists
        if not hasattr(self, "x") or self.x is None:
            raise ex.NoDataException("Axis X has no data.")
        if not hasattr(self, "y") or self.y is None:
            raise ex.NoDataException("Axis Y has no data.")
        if not hasattr(self, "z") or self.z is None:
            raise ex.NoDataException("Axis Z has no data.")

        # Create 3D projection
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface.
        surf = ax.plot_surface(
            self.x,
            self.y,
            self.z,
            cmap=self.cmap,
            linewidth=self.linewidth,
            antialiased=self.antialiased,
        )

        # Axis labels
        if hasattr(self, "xlabel"):
            ax.set_xlabel(self.xlabel)
        if hasattr(self, "ylabel"):
            ax.set_ylabel(self.ylabel)
        if hasattr(self, "zlabel"):
            ax.set_zlabel(self.zlabel)

        # Axis limits
        if hasattr(self, "xlim"):
            ax.set_xlim(self.xlim)
        if hasattr(self, "ylim"):
            ax.set_ylim(self.ylim)
        if hasattr(self, "zlim"):
            ax.set_zlim(self.zlim)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Camera angle
        if self.elev and self.azim:
            ax.view_init(elev=self.elev, azim=self.azim)

        # Camera distance
        ax.dist = self.dist

        plt.show()

        # If this is the first time this method has been called,
        # we want to allow the user to change cosmetic properties
        # and show the plot immediately when those properties are changed.
        # Therefore, if the show_immediately flag has not yet been created,
        # create it here and set it to True.
        if not hasattr(self, "show_immediately"):
            self.show_immediately = True
