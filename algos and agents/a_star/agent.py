#!/usr/bin/python3
'''
Very simple example how to use gym_wrapper and BasedAgent class for state space search 
@author: Zdeněk Rozsypálek, and the KUI-2019 team
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018, 2019
'''

import time
import kuimaze
import os
import random 
import nodes as nod
#import numpy as np

class Agent(kuimaze.BaseAgent):
    '''
    Simple example of agent class that inherits kuimaze.BaseAgent class 
    '''
    def __init__(self, environment):
        self.environment = environment

    def find_path(self):
        #functions start
        def g_function(a, b, c):#unnecessary, but i will leave it here for the time being and educational purposes
            a = np.array(a)
            b = np.array(b)
            if(c < 2 ):
                c = 1
            return np.linalg.norm(a-b)*c

        def h_function(a, b): #simple heurestic with manhattan disctance
            dx = abs(a[0]-b[0])
            dy = abs(a[1]-b[1])
            D = 1
            D2 = 1.4
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

        observation = self.environment.reset() # must be called first, it is necessary for maze initialization
        goal = observation[1][0:2]
        position = observation[0][0:2] 
        start = observation[0][0:2]                              # initial state (x, y)     
        
        close_array = []                                         # some arrs and vars and yarrs
        open_array = []
        path = []
        goal_reached = False
        
        start_node = nod.Node(None, start)                       #generate start and end nodes
        start_node.g = start_node.h = start_node.f = 0
        goal_node = nod.Node(None, goal)
        goal_node.g = goal_node.h = goal_node.f = 0
        
        open_array.append(start_node)                            

        print("Starting not random searching")

        while(len(open_array) > 0):  #LOOP
            infinity = 1000000       #infinity
            index = 0
            for i in range (len(open_array)):   #identifing smallest
                if(open_array[i].f < infinity):
                    infinity = open_array[i].f
                    index = i
                      
            position_node = open_array.pop(index)               
            
            position = position_node.position

            if position == goal:                     # break the loop when the goal position is reached 
                current = position_node
                while current is not None:           # path create
                    path.append(current.position)
                    current = current.parent
                goal_reached = True
                break

            new_positions = self.environment.expand(position)         # [[(x1, y1), cost], [(x2, y2), cost], ... ]

            for i in range (len(new_positions)):     #children already generated, now we deciding what to do with them
                child_node = nod.Node(position_node, new_positions[i][0], new_positions[i][1])
                if(child_node in close_array):       #ignoring visited
                    continue
                child_node.g = position_node.g + (0.9 * child_node.cost) 
                child_node.h = h_function(child_node.position, goal_node.position)
                child_node.f = child_node.g + child_node.h
               
                dont_add = True 
                for j in range(len(open_array)):     #deciding if to upgrade g cost
                    if(child_node == open_array[j]): 
                        if(child_node.g > open_array[j].g):
                            dont_add = False
                            continue
                if(dont_add):
                    open_array.append(child_node)
    
            close_array.append(position_node)        #CLOSE
            
        print("path", path)
        if(goal_reached):
            return path[::-1] # cool trick everybody knows to reverse list
        else:
            return None

if __name__ == '__main__':

    MAP = 'maps/normal/normal11.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False

    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env) 

    path = agent.find_path()
    print(path)
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    time.sleep(4)
