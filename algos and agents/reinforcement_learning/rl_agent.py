#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
A sandbox for playing with the HardMaze
@author: Tomas Svoboda
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018
'''


import kuimaze
import numpy as np
import sys
import os
import gym
import time
import random


# MAP = 'maps/normal/normal3.bmp'
MAP = 'maps/normal/normal9.bmp'
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
# PROBS = [0.8, 0.1, 0.1, 0]
PROBS = [1, 0, 0, 0]
GRAD = (0, 0)
SKIP = False
VERBOSITY = 2

GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 255, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

# MAP = GRID_WORLD3


def wait_n_or_s():

    def wait_key():
        """
        returns key pressed ... works only in terminal! NOT in IDE!
        """
        result = None
        if os.name == 'nt':
            import msvcrt
            # https://cw.felk.cvut.cz/forum/thread-3766-post-14959.html#pid14959
            result = chr(msvcrt.getch()[0])
        else:
            import termios
            fd = sys.stdin.fileno()

            oldterm = termios.tcgetattr(fd)
            newattr = termios.tcgetattr(fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, newattr)
            try:
                result = sys.stdin.read(1)
            except IOError:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        return result

    '''
    press n - next, s - skip to end ... write into terminal
    '''
    global SKIP
    x = SKIP
    while not x:
        key = wait_key()
        x = key == 'n'
        if key == 's':
            SKIP = True
            break


def get_visualisation(table):
    ret = []
    for i in range(len(table[0])):
        for j in range(len(table)):
            ret.append({'x': j, 'y': i, 'value': [table[j][i][0], table[j][i][1], table[j][i][2], table[j][i][3]]})
    return ret

def learn_policy(env):
    stoptime = time.time()
    # Number of discrete actions
    num_actions = env.action_space.n
    x_dims = env.observation_space.spaces[0].n
    y_dims = env.observation_space.spaces[1].n
    # Q-table:
    global q_table
    q_table = np.zeros([x_dims, y_dims, num_actions], dtype=float)

    while (time.time() - stoptime) < 18.5:
        policy = walk_randomly(env)
    #print("fin: ", time.time() - stoptime)
    return policy
    
def walk_randomly(used_env):
    # used_env.action_space.np_random.seed(123) if you want to fix the randomness between experiments
    
    
    policy = dict()
    used_env.action_space.np_random.seed()
    obv = used_env.reset()
    state = obv[0:2]
    
    is_done = False
    q_table1 = q_table
    x_dims = used_env.observation_space.spaces[0].n
    y_dims = used_env.observation_space.spaces[1].n
    terminal_states = []
    MAX_T = 1000  # max trials (for one episode)
    t = 0
    for x in range(x_dims):
        for y in range(y_dims):
            policy[x, y] = np.argmax(q_table1[x][y])
    
    while not is_done and t < MAX_T:
        t += 1
        if 0.6 > random.random():
            action = policy[obv[0:2]]
        else:
            action = used_env.action_space.sample()

        obv, reward, is_done, _ = used_env.step(action)
        next_state = obv[0:2]
        
        q_table[state[0]][state[1]][action] = reward +  max(q_table1[next_state[0]][next_state[1]])
        
        if is_done:
            terminal_states.append(obv[0:2])
        state = next_state

    used_env.visualise(get_visualisation(q_table))
    used_env.render()
    '''
    if not is_done:
        print('Timed out')
    '''
    return policy

if __name__ == "__main__":
    # Initialize the maze environment
    env = kuimaze.HardMaze(map_image=MAP, probs=PROBS, grad=GRAD)

    if VERBOSITY > 0:
        print('====================')
        print('works only in terminal! NOT in IDE!')
        print('press n - next')
        print('press s - skip to end')
        print('====================')
    
    '''
    Define constants:
    '''
    # Maze size
    x_dims = env.observation_space.spaces[0].n
    y_dims = env.observation_space.spaces[1].n
   

    '''
    if VERBOSITY > 0:
        env.visualise(get_visualisation(q_table))
        env.render()
    '''
    policy = learn_policy(env)
    print(policy)
    print(type(policy))
    env.visualise(get_visualisation(q_table))
    env.render()
    time.sleep(20)
    #visualise_policy(policy)
    if VERBOSITY > 0:
        SKIP = False
        env.visualise(get_visualisation(q_table))
        env.render()
        wait_n_or_s()

        env.save_path()
        env.save_eps()
