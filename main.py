"""
Travelling Salesman Using Simulated Annealing
v1.0 25 June 2017 Robert Crowe
Python 3

This is an implementation of a solution to the travelling salesman problem using simulated annealing.  This
was written as an optional exercise for the Udacity Artificial Intelligence Nanodegree, for the unit discussing
local search strategies where the focus is on the problem solution and not the path to the solution.  Algorithms
that are often applied to these problems include various forms of hill-climbing and local beam search.

Simulated annealing is inspired by physical annealing:

"Annealing, in metallurgy and materials science, is a heat treatment that alters the physical and sometimes 
chemical properties of a material to increase its ductility and reduce its hardness, making it more workable. 
It involves heating a material to above its recrystallization temperature, maintaining a suitable temperature, 
and then cooling.
In annealing, atoms migrate in the crystal lattice and the number of dislocations decreases, leading to the 
change in ductility and hardness." (Wikipedia: https://en.wikipedia.org/wiki/Annealing_(metallurgy))

Here we use a simulated temperature to allow a high level of exploration of the local state space at the beginning
of the the process, gradually lowering the temperature and thus the amount of exploration until finally reaching
a solution.  The result is to avoid getting stuck in local minima, and find global minima with a fairly
high degree of reliability, offering both efficiency and completeness (finds a goal if it exists).
"""
import json
import copy

import math
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""Read input data and define helper functions for visualization."""
# Map services and data available from U.S. Geological Survey, National Geospatial Program.
# Please go to http://www.usgs.gov/visual-id/credit_usgs.html for further information
map = mpimg.imread("map.png")  # US States & Capitals map

# List of 30 US state capitals and corresponding coordinates on the map
with open('capitals.json', 'r') as capitals_file:
    capitals = json.load(capitals_file)
capitals_list = list(capitals.items())

def show_path(path, starting_city, title='Map', w=12, h=8):
    """Plot a TSP path overlaid on a map of the US States & their capitals."""
    x, y = list(zip(*path))
    _, (x0, y0) = starting_city
    plt.imshow(map)
    plt.plot(x0, y0, 'y*', markersize=15)  # y* = yellow star for starting point
    plt.plot(x + x[:1], y + y[:1])  # include the starting point at the end of path
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])
    fig.suptitle(title, fontsize=22, fontweight='normal')
    plt.show()

def simulated_annealing(problem, schedule):
    """The simulated annealing algorithm, a version of stochastic hill climbing
    where some downhill moves are allowed. Downhill moves are accepted readily
    early in the annealing schedule and then less often as time goes on. The
    schedule input determines the value of the temperature T as a function of
    time. [Norvig, AIMA Chapter 3]
    
    Parameters
    ----------
    problem : Problem
        An optimization problem, already initialized to a random starting state.
        The Problem class interface must implement a callable method
        "successors()" which returns states in the neighborhood of the current
        state, and a callable function "get_value()" which returns a fitness
        score for the state. (See the `TravelingSalesmanProblem` class below
        for details.)

    schedule : callable
        A function mapping time to "temperature". "Time" is equivalent in this
        case to the number of loop iterations.
    
    Returns
    -------
    Problem
        An approximate solution state of the optimization problem
        
    See Also
    --------
    AIMA simulated_annealing() pseudocode
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Simulated-Annealing.md
    """
    import sys
    current = problem
    for t in range(sys.maxsize):
        temperature = schedule(t)
        if temperature < 1e-10:
            return current
        neighbors = problem.successors()
        if not neighbors:
            return current
        next = random.choice(neighbors)
        delta_e = next.get_value() - current.get_value()
        clear_improvement = delta_e > 0
        choose_with_probability = math.exp(delta_e / temperature) > random.uniform(0.0, 1.0)
        if clear_improvement or choose_with_probability:
            current = next

class TravelingSalesmanProblem:
    """Representation of a traveling salesman optimization problem.  The goal
    is to find the shortest path that visits every city in a closed loop path.
    
    Parameters
    ----------
    cities : list
        A list of cities specified by a tuple containing the name and the x, y
        location of the city on a grid. e.g., ("Atlanta", (585.6, 376.8))
    
    Attributes
    ----------
    names
    coords
    path : list
        The current path between cities as specified by the order of the city
        tuples in the list.
    """
    def __init__(self, cities):
        self.path = copy.deepcopy(cities)
    
    def copy(self):
        """Return a copy of the current board state."""
        new_tsp = TravelingSalesmanProblem(self.path)
        return new_tsp
    
    @property
    def names(self):
        """Strip and return only the city name from each element of the
        path list. For example,
            [("Atlanta", (585.6, 376.8)), ...] -> ["Atlanta", ...]
        """
        names, _ = zip(*self.path)
        return names
    
    @property
    def coords(self):
        """Strip the city name from each element of the path list and return
        a list of tuples containing only pairs of xy coordinates for the
        cities. For example,
            [("Atlanta", (585.6, 376.8)), ...] -> [(585.6, 376.8), ...]
        """
        _, coords = zip(*self.path)
        return coords
    
    def successors(self):
        """Return a list of states in the neighborhood of the current state by
        switching the order in which any adjacent pair of cities is visited.
        
        For example, if the current list of cities (i.e., the path) is [A, B, C, D]
        then the neighbors will include [A, B, D, C], [A, C, B, D], [B, A, C, D],
        and [D, B, C, A]. (The order of successors does not matter.)
        
        In general, a path of N cities will have N neighbors (note that path wraps
        around the end of the list between the first and last cities).

        Returns
        -------
        list<Problem>
            A list of TravelingSalesmanProblem instances initialized with their list
            of cities set to one of the neighboring permutations of cities in the
            present state
        """
        new_problem_list = []
        for i in range(0, len(self.path)-1):
            new_problem = copy.deepcopy(self.path)
            temp = new_problem[i]
            new_problem[i] = new_problem[i+1]
            new_problem[i+1] = temp
            new_problem_list.append(TravelingSalesmanProblem(new_problem))
            
        new_problem = copy.deepcopy(self.path)
        temp = new_problem[0]
        new_problem[0] = new_problem[len(self.path)-1]
        new_problem[len(self.path)-1] = temp
        new_problem_list.append(TravelingSalesmanProblem(new_problem))
        return new_problem_list
            

    def get_value(self):
        """Calculate the total length of the closed-circuit path of the current
        state by summing the distance between every pair of adjacent cities.  Since
        the default simulated annealing algorithm seeks to maximize the objective
        function, return -1x the path length. (Multiplying by -1 makes the smallest
        path the smallest negative number, which is the maximum value.)
        
        Returns
        -------
        float
            A floating point value with the total cost of the path given by visiting
            the cities in the order according to the self.cities list
        """
        dist = 0.0
        for i in range(0, len(self.coords)-1):
            city1, city2 = self.coords[i], self.coords[i+1]
            dist +=(abs(city1[0] - city2[0]) ** 2 + abs(city1[1] - city2[1]) ** 2) ** 0.5
        
        city1, city2 = self.coords[0], self.coords[len(self.coords)-1]
        dist += (abs(city1[0] - city2[0]) ** 2 + abs(city1[1] - city2[1]) ** 2) ** 0.5
        return -dist

# These are presented as globals so that the signature of schedule()
# matches what is shown in the AIMA textbook; you could alternatively
# define them within the schedule function, use a closure to limit
# their scope, or define an object if you would prefer not to use
# global variables
alpha = 0.95
temperature = 1e4

def schedule(time):
    return alpha ** time * temperature

# Create the problem instance and plot the initial state
num_cities = 10
capitals_tsp = TravelingSalesmanProblem(capitals_list[:num_cities])
starting_city = capitals_list[0]
dist = -capitals_tsp.get_value()
print("Initial path value: {:.2f}".format(dist))
print(capitals_list[:num_cities])  # The start/end point is indicated with a yellow star
show_path(capitals_tsp.coords, starting_city, title='Initial Path\n(length = {:.2f})'.format(dist))

print('\nWorking ...\n')

# set the decay rate and initial temperature parameters, then run simulated annealing to solve the TSP
alpha = 0.95
temperature=1e6
result = simulated_annealing(capitals_tsp, schedule)
dist = -result.get_value()
print("Final path length: {:.2f}".format(dist))
print(result.path)
show_path(result.coords, starting_city, title='Final Path\n(length = {:.2f})'.format(dist))
