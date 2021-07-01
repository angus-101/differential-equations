# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:17:31 2020

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################


def Grid(nodes):                                                                
    
    """
    This function defines the set of nodes that act as the initial guesses
    for the potential at each point. Two arrays are first defined, and old one and
    a new one (that gets updated), which consist of only zeros. Then the outside 
    square of each array is replaced with a non-zero number that acts as the 
    boundary conditions of the system. The function then returns the size of the 
    array (nodes) and the old and new arrays.
    """ 
    
    BC = 1                                                                      # Boundary condition.
    new = np.zeros((nodes,nodes))
    new[:,[0,-1]] = new[[0,-1]] = BC                                            # Changes the values of the outside shell in the array (sets the boundary condition).
    
    return nodes,new,BC


def Potential(nodes,new,convergence_limit):
    
    """
    This function calculates the potential at each node. It cycles through all the
    elements in the array (excluding those in the outer shell) and updates each 
    to be the average of its four neighbours. With each update, it checks the 
    percent change between the previous and current iteration. If it is lower than
    a certain bound, the convergence condition is met and it stops iterating, 
    returning the complete array.
    """
    
    old = np.zeros((nodes,nodes))
    Convergence = False
    
    while Convergence == False: 
        for i in range(1,nodes-1):  
            for j in range(1,nodes-1):      
                new[i][j] = (1/4) * (new[i-1][j] + 
                                     new[i][j-1] + new[i+1][j] + new[i][j+1])   # Calculating the new potential at each node.
        
        middle = int(np.floor(nodes/2))
        centre_new = new[middle][middle]
        centre_old = old[middle][middle]
        
        if centre_old != 0:                                                     # Checking for convergence.
            if 1 - convergence_limit < centre_new / centre_old < 1 + convergence_limit:                
                Convergence = True
        old = np.copy(new)                                                      # Making old = new so the convergence condition can be checked next iteration.
            
    return old

"""
print("This will take about 20 seconds.")

nodes = Grid(100)[0]
new = Grid(nodes)[1]
convergence_limit = 10**-2
potential = Potential(nodes,new,convergence_limit)
          
plt.imshow(potential, cmap='hot')
plt.xlabel("x position")
plt.ylabel("y position")
plt.colorbar(label = "Relative potential")
plt.show()
"""

###############################################################################


def convergence_func(nodes,BC):
    
    """
    This function measures the average ratio of the analytical to the 
    computational solutions for each iteration, with varying convergence
    limit. The analytical solution is the solution as if the computer has an
    infinitesimally small convergence limit, i.e. an array where each element 
    equals the boundary condition. 
    """
    
    analyticalConv = np.full((nodes,nodes),BC)                                  # Defining the analytical solution array.
    diff = []                                                                   # List that will contain the difference between the analytical and computational solutions.
    convergence_list = []                                                       # List that will contain the varying convergence limits.
    convergence_limit = 0.9
    
    while convergence_limit >= 10**-1:
        
        diff_sum = 0
        new = Grid(nodes)[1]
        computationalConv = Potential(nodes,new,convergence_limit)
        
        for i in range(1,nodes-1):
            for j in range(1,nodes-1):
                diff_sum += computationalConv[i][j] / analyticalConv[i][j]      # Difference between each element.
        
        diff.append(diff_sum/(nodes-2)**2)                                      # Taking the average difference over the entire array.
        convergence_list.append(convergence_limit)
        convergence_limit -= 0.1
        
    return convergence_list,diff


def size_func(BC):
    
    """
    This function is very similar to the previous; it compares the average 
    ratio of the analytical solution to the computational solution with a 
    varying grid density. The analytical solution is, as before, an array 
    filled with values equal to the boundary condition.
    """
    
    grid_density = 10 
    diff_grid = []                                                              # List that will contain the difference between thw analytical and computational solutions.
    grid_density_list = []                                                      # List that will contain the varying grid density.
    convergence_limit = 10**-1
    
    while grid_density <= 100:
        
        diff_sum_grid = 0
        new = Grid(grid_density)[1]
        computationalGrid = Potential(grid_density,new,convergence_limit)       # Calculating the computational solution.
        analyticalGrid = np.full((grid_density,grid_density),BC)                # Defining the analytical solution.
        
        for i in range(1,grid_density-1):
            for j in range(1,grid_density-1):
                diff_sum_grid += computationalGrid[i][j] / analyticalGrid[i][j] # Difference between each element.
                
        diff_grid.append(diff_sum_grid/(grid_density-2)**2)                     # Taking the average difference over the entire array.
        grid_density_list.append(grid_density)
        grid_density += 10
        
    return grid_density_list,diff_grid

"""
print("This will take about 20 seconds.")   

nodes = Grid(100)[0]
BC = Grid(nodes)[2]
convergence_list = convergence_func(nodes,BC)[0]
conv_diff = convergence_func(nodes,BC)[1]    
grid_density_list = size_func(BC)[0]
grid_diff = size_func(BC)[1]

plt.plot(convergence_list,conv_diff)
plt.xlabel("Convergence limit")
plt.ylabel("Average ratio")
plt.title("Convergence limit versus average ratio of analytical to computational solutions")
plt.xscale("log")
plt.show()

plt.plot(grid_density_list,grid_diff)
plt.xlabel("Grid density")
plt.ylabel("Average ratio")
plt.title("Grid density versus average ratio of analytical to computational solutions")
plt.xscale("log")
plt.show()
"""
        
###############################################################################


def CapGrid(nodes):                                                                
   
    """
    This function defines the initial guess for the parallel plate capacitor 
    potential. It is an array that consists of zeros everywhere, apart from two 
    parallel lines near the centre. The left of these consists of ones, and the
    right of minus ones, to represent the positive and negative charged plates
    respectively.
    """

    newCap = np.zeros((nodes,nodes))
    separation = 10
    length = 90
    leftPlate = np.floor((nodes/2 - separation/2)*nodes/100)                    # Defining the positions and heights of the capacitor plates.
    rightPlate = np.floor((nodes/2 + separation/2)*nodes/100)
    topPlate = np.floor((nodes/2 - length/2)*nodes/100)
    bottomPlate = np.floor((nodes/2 + length/2)*nodes/100)
    BC = 1
    
    for i in range(nodes):
        for j in range(nodes):
            if topPlate < i < bottomPlate and j == leftPlate:
                newCap[i][j] = BC                                               # Positively charged plate.
            if topPlate < i < bottomPlate and j == rightPlate:
                newCap[i][j] = -BC                                              # Negatively charged plate.
                
    return nodes,newCap,separation,BC


def CapPot(nodes,newCap,convergence_limit,BC):

    """
    This function calculates the potential at each node in the capacitor array.
    It defines the 'old' array as one full of zeros, and makes this equal to the
    updated 'new' array after each iteration. It does the same check for
    convergence as previously, only now the ratio between the old and new values
    can be less than one (due to the negative potential introduced in this part).
    Hence a range either side of 1 is used. 
    """

    old = np.zeros((nodes,nodes))
    Convergence = False
    
    while Convergence == False: 
        
        convergence_count = 0                                                   # Keeps track of the number of nodes that satisfy the convergence condition.
        node_count = 0                                                          # Keeps track of the number of nodes.
        
        for i in range(1,nodes-1):
            for j in range(1,nodes-1):
                
                if not (newCap[i][j] == BC or newCap[i][j] == -BC):             # This if statement keeps the boundary conditions constant.
                    newCap[i][j] = (1/4) * (newCap[i-1][j] + newCap[i][j-1] + 
                          newCap[i+1][j] + newCap[i][j+1])
                    
                    node_count += 1
                    
                    if old[i][j] != 0:                                          # Check to prevent divide by zero error.
                        ratio =  abs(newCap[i][j] / old[i][j])
                        if 1 - convergence_limit < ratio < 1 + convergence_limit:          
                            convergence_count += 1
                            
        if convergence_count == node_count:                                     # If the number of converged nodes = the number of nodes, then every node has converged.
            Convergence = True
                            
        old = np.copy(newCap)                                                    
            
    return old


def CapElec(nodes,old):

    """
    This function calculates the absolute value of the electric field for the 
    parallel plate capacitor defined above. The gradient function returns two 
    arrays - one for each dimension - the gradient in the x and in the y. Hence
    the absolute value of the electric field can be calculated by taking the 
    absolute values of each element in these arrays.
    """ 

    elec = np.zeros((nodes,nodes))
    elec0 = np.gradient(old)[0]                                                 # np.gradient function returns 2 arrays.
    elec1 = np.gradient(old)[1]
    
    for i in range(nodes):
        for j in range(nodes):
            elec[i][j] = np.sqrt(elec0[i][j]**2 + elec1[i][j]**2)               # Calculating the absolute value from each array.
            
    return elec,elec0,elec1

"""
print("This will take about 5 seconds.")

nodes = CapGrid(100)[0]
newCap = CapGrid(nodes)[1]
separation = CapGrid(nodes)[2]
BC = CapGrid(nodes)[3]
convergence_limit = 10**-1
old = CapPot(nodes,newCap,convergence_limit,BC)
elec = CapElec(nodes,old)[0]
elec0 = CapElec(nodes,old)[1]
elec1 = CapElec(nodes,old)[2]

plt.imshow(old, cmap='inferno')
plt.xlabel("x position")
plt.ylabel("y position")
plt.colorbar(label = "Relative potential")
plt.show()

plt.imshow(elec, cmap='inferno')
plt.xlabel("x position")
plt.ylabel("y position")
plt.colorbar(label = "Relative electric field")
plt.show()

plt.quiver(elec0,elec1)
plt.xlabel("x position")
plt.ylabel("y position")
plt.title("Electric field vector plot")
plt.show()
"""

###############################################################################


def rod(nodes):
    
    """
    This function initialises the iron poker, and sets the boundary 
    conditions (1000 C at one end, 0 C at the other). It then defines some
    constants, including h and alpha, and creates the displacment list that
    stores the values of distance along the length of the poker.
    """
    
    rodlist = [293] * nodes                                                     # Initialising the poker.
    rodlist[0] = 273
    rodlist[-1] = 1273
    
    delta_t = 1                                                                 # Defining constants.
    h = 0.5 / nodes
    alpha = 59 / (450 * 7900)
    p = alpha * delta_t / h**2
    
    distance = 0
    displacement = []
    
    for i in range(nodes):                                                      # Creating the displacement list.
        displacement.append(distance)
        distance += h
    
    return nodes,rodlist,p,displacement,delta_t


def matrix(nodes,p):
    
    """
    This function defines the matrix that controls the diffusion of heat 
    through the poker. Only the nodes either side of the current node are 
    affected by the heat transfer, hence the matrix consisting entirely of
    zeros, apart from along the diagonal and elements above and below the 
    diagonal.
    """
    
    heatMatrix = np.zeros((nodes,nodes))
    
    for i in range(nodes):
        for j in range(nodes):
            
            if i == j:                                                          # Defining values on the diagonal.
                heatMatrix[i][j] = 1 + 2*p
            elif j + 1 == i or j - 1 == i:                                      # Defining values above and below the diagonal.
                heatMatrix[i][j] = -p
                
    heatMatrix[0][0] = heatMatrix[-1][-1] = 1                                   # Making the boundary conditions constant.
    heatMatrix[0][1] = heatMatrix[-1][-2] = 0
                
    return heatMatrix


def heatSolve(nodes,heatMatrix,rodlist,delta_t,t_lim):
    
    """
    This function actually solves the diffusion equation. At each time step,
    the temperature at each node along the poker is solved. This repeats until
    the time limit is reached, returning the temperature distribution at that
    time. The whole process repeats for different values of the time limit
    to show convergence.
    """
    
    temp = []
    
    for j in range(len(t_lim)):
        t = 0
        while t <= t_lim[j]:      
            for i in range(nodes):
                rodlist = np.linalg.solve(heatMatrix,rodlist)                   # Using scipy to solve the equation.
                t += delta_t
        temp.append(rodlist)

    return temp


def error(nodes,t_lim,delta_t):
    
    """
    This function compares the computational solution with the analytical
    solution, where the analytical solution has linearly increasing 
    temperature along the length of the poker from 273K to 1273K. The average
    ratio of these values is taken for each time, and then plotted versus 
    time to show how the accuracy of the computational solution converges.
    """
    
    x = np.linspace(0,0.5,nodes)
    analytical = 2000*x + 273                                                   # Defining the analytical solution.
    ratio_list = []
    t = 0
    t_list = []
    rodlist = rod(nodes)[1]                                                     # Re-initialising the poker temperatures.
    
    while t <= t_lim[-1]:
        
        ratio_count = 0
        rodlist = np.linalg.solve(heatMatrix,rodlist)                           # Solving the poker temperature for each time.
        
        for i in range(nodes):
            
            if rodlist[i] <= analytical[i]:
                ratio_count += rodlist[i]/analytical[i]                         # Calculating the ratio of the computational to analytical solutions.
            else:
                ratio_count += analytical[i]/rodlist[i]
            
        ratio_list.append(ratio_count/(nodes))                                  # Finding the average ratio across the entire poker.
        t_list.append(t)
        t += delta_t
        
    return ratio_list,t_list

"""
nodes = rod(20)[0]
rodlist = rod(nodes)[1]
p = rod(nodes)[2]
displacement = rod(nodes)[3]
delta_t = rod(nodes)[4]
heatMatrix = matrix(nodes,p)
t_lim = [125,250,500,1000,2000,4000]
temp = heatSolve(nodes,heatMatrix,rodlist,delta_t,t_lim)

ratio_list = error(nodes,t_lim,delta_t)[0]
t_list = error(nodes,t_lim,delta_t)[1] 


for i in range(len(t_lim)):
    plt.plot(displacement,temp[i])
plt.xlabel("Displacement along rod (m)")
plt.ylabel("Temperature (K)")
plt.legend(t_lim,title="Number of iterations",framealpha = 0.5)
plt.title("Displacement along rod versus temperature (ice bath)")
plt.show()

plt.plot(t_list,ratio_list)
plt.xlabel("Time")
plt.ylabel("Error")
plt.title("Average ratio of computational to analytical solutions versus time (ice bath)")
plt.show()
"""

###############################################################################


def matrixNoCold(nodes,p):
    
    """
    This function has the same purpose as the function 'matrix' previously, 
    but now it takes into account the fact that there is no boundary condition
    on one end (the cold end) of the poker. 
    """
    
    heatMatrix = np.zeros((nodes,nodes))
    
    for i in range(nodes):
        for j in range(nodes):
            
            if i == j:                                                          # Defining values on the diagonal.
                heatMatrix[i][j] = 1 + 2*p
            elif j + 1 == i or j - 1 == i:                                      # Defining values above and below the diagonal.
                heatMatrix[i][j] = -p
                
    heatMatrix[-1][-1] = 1                                                      # Making the boundary conditions constant.
    heatMatrix[0][0] = 1 + p
    heatMatrix[0][1] = -p
    heatMatrix[-1][-2] = 0
                
    return heatMatrix


def errorNoCold(nodes,t_lim,delta_t):
    
    """
    This function has the same purpose as the function 'error' previously, but
    now it calculates the error in the case of no ice bath keeping one end of 
    the poker at 273K.
    """
    
    analytical = [1273] * nodes                                                 # Defining the analytical solution.
    ratio_list = []
    t = 0
    t_list = []
    rodlist = rod(nodes)[1]                                                     # Re-initialising the poker temperatures.
    
    while t <= t_lim[-1]:
        
        ratio_count = 0
        rodlist = np.linalg.solve(heatMatrix,rodlist)                           # Solving the poker temperature for each time.
        
        for i in range(nodes):
            
            if rodlist[i] <= analytical[i]:
                ratio_count += rodlist[i]/analytical[i]                         # Calculating the ratio of the computational to analytical solutions.
            else:
                ratio_count += analytical[i]/rodlist[i]
            
        ratio_list.append(ratio_count/(nodes))                                  # Finding the average ratio across the entire poker.
        t_list.append(t)
        t += delta_t
        
    return ratio_list,t_list

"""
nodes = rod(20)[0]
rodlist = rod(nodes)[1]
rodlist[0] = 293                                                                # Setting the initial poker temperature to 293K rather than 273K.
p = rod(nodes)[2]
displacement = rod(nodes)[3]
delta_t = rod(nodes)[4]
heatMatrix = matrixNoCold(nodes,p)
t_lim = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
temp = heatSolve(nodes,heatMatrix,rodlist,delta_t,t_lim)
ratio_list = errorNoCold(nodes,t_lim,delta_t)[0]
t_list = errorNoCold(nodes,t_lim,delta_t)[1] 


for i in range(len(t_lim)):
    plt.plot(displacement,temp[i])
plt.ylim(273,1300)
plt.xlabel("Displacement along rod (m)")
plt.ylabel("Temperature (K)")
plt.legend(t_lim,title="Number of iterations",framealpha = 0.5)
plt.title("Displacement along rod versus temperature (no ice bath)")
plt.show()

plt.plot(t_list,ratio_list)
plt.xlabel("Time")
plt.ylabel("Error")
plt.title("Average ratio of computational to analytical solutions versus time (no ice bath)")
plt.show()
"""

###############################################################################


def pointCharge(nodes):
    
    """
    This function creates an array and makes the centre node the boundary 
    condition, simulating a point charge.
    """
    
    grid = np.zeros((nodes,nodes))
    BC = 1
    mid = int(nodes/2)
    
    grid[mid][mid] = BC                                                         # Defining the centre node.
    
    return nodes,grid,BC


def circularShell(nodes):
    
    """
    This function creates an array and fills a centre circular shell with
    values equal to the boundary condition, simulating a circular shell
    of charge.
    """
    
    grid_shell = np.zeros((nodes,nodes))
    BC = 1
    radius = nodes/4
    
    for i in range(nodes):
        for j in range(nodes):
            circle = (i-nodes/2)**2 + (j-nodes/2)**2                            # Equation of a circle.
            if circle - 25 < radius**2 < circle + 25:                           # Condition for changing the value of a node.
                grid_shell[i][j] = BC
            
    return nodes,grid_shell,BC


print("This will take about 3 minutes (sorry!).")

nodes = pointCharge(100)[0]
grid_point_charge = pointCharge(nodes)[1]
BC = pointCharge(nodes)[2]
convergence_limit = 10**-5
point_charge = CapPot(nodes,grid_point_charge,convergence_limit,BC)

grid_shell = circularShell(nodes)[1]
circular_shell = CapPot(nodes,grid_shell,convergence_limit,BC)

e_shell = CapElec(nodes,circular_shell)[0]                                      # Calculating the electric field for the case of a circular shell of charge.


plt.imshow(point_charge, cmap='inferno')
plt.xlabel("x position")
plt.ylabel("y position")
plt.title("Potential for a point charge")
plt.colorbar(label = "Relative potential")
plt.show()

plt.imshow(grid_shell, cmap='inferno')
plt.xlabel("x position")
plt.ylabel("y position")
plt.title("Potential for a circular shell of charge")
plt.colorbar(label = "Relative potential")
plt.show()

plt.imshow(e_shell, cmap='inferno')
plt.xlabel("x position")
plt.ylabel("y position")
plt.title("Electric field for a circular shell of charge")
plt.colorbar(label = "Relative electric field")
plt.show()










































        
        
        
        
