#############################################
Travelling Salesman Using Simulated Annealing
#############################################

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
of the process, gradually lowering the temperature and thus the amount of exploration until finally reaching
a solution.  The result is that we avoid getting stuck in local minima, and find global minima with a fairly
high degree of reliability.  Simulated annealing offers both efficiency and completeness (finds a goal if it exists).
