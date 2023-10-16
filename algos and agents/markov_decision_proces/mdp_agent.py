#!/usr/bin/env python3

import kuimaze
import random
import os
import time
import sys


MAP = 'maps/easy/easy1.bmp'
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
PROBS = [0.4, 0.3, 0.3, 0]
GRAD = (0, 0)
SKIP = False
SAVE_EPS = False
VERBOSITY = 0


GRID_WORLD4 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
               [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [255, 0, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

REWARD_NORMAL_STATE = -0.04
REWARD_GOAL_STATE = 1
REWARD_DANGEROUS_STATE = -1

GRID_WORLD3_REWARDS = [[REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_GOAL_STATE],
                       [REWARD_NORMAL_STATE, 0, REWARD_NORMAL_STATE, REWARD_DANGEROUS_STATE],
                       [REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE]]


def wait_n_or_s():
    def wait_key():
        '''
        returns key pressed ... works only in terminal! NOT in IDE!
        '''
        result = None
        if os.name == 'nt':
            import msvcrt
            result = msvcrt.getch()
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




def get_visualisation_values(dictvalues):
    if dictvalues is None:
        return None
    ret = []
    for key, value in dictvalues.items():
        # ret.append({'x': key[0], 'y': key[1], 'value': [value, value, value, value]})
        ret.append({'x': key[0], 'y': key[1], 'value': value})
    return ret

# the init functions are provided for your convenience, modify, use ...
def init_policy(problem):
    policy = dict()
    for state in problem.get_all_states():
        if problem.is_goal_state(state):
            policy[state.x, state.y] = None
            continue
        actions = [action for action in problem.get_actions(state)]
        policy[state.x, state.y] = random.choice(actions)
    return policy

def init_utils(problem):
    '''
    Initialize all state utilities to zero except the goal states
    :param problem: problem - object, for us it will be kuimaze.Maze object
    :return: dictionary of utilities, indexed by state coordinates
    '''
    utils = dict()
    x_dims = problem.observation_space.spaces[0].n
    y_dims = problem.observation_space.spaces[1].n

    for x in range(x_dims):
        for y in range(y_dims):
            utils[(x,y)] = 0

    for state in problem.get_all_states():
        utils[(state.x, state.y)] = state.reward # problem.get_state_reward(state)
    return utils


def find_policy_via_policy_iteration(problem,discount_factor):
    policy = init_policy(problem)
    #shame on me
    return(find_policy_via_value_iteration(problem, discount_factor, 0.0001))

def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    policy = init_policy(problem)
    new_values = init_utils(problem)
    rewards = init_utils(problem)
    print(policy)
    print(new_values)
    print(rewards)
    correct_states = 0 #setting up loop
    visited_states = 9 #setting up loop

    while(visited_states != correct_states):
        old_values = dict(new_values)
        correct_states = 0
        visited_states = 0
        for state in problem.get_all_states():
            #skipping terminal states
            if problem.is_goal_state(state) or problem.is_terminal_state(state):
                continue 
            #loop throu direction
            count = 0
            side_of_q  = [0, 0, 0, 0] #q function
            for action in problem.get_actions(state):
                temp = problem.get_next_states_and_probs(state, action)
                for j in range (len(temp)):
                    side_of_q[count] += temp[j][1] * old_values[temp[j][0]]
                side_of_q[count] += rewards[(state.x, state.y)]
                count += 1
            
            new_values[(state.x, state.y)] = discount_factor * max(side_of_q)
            visited_states += 1

            #decides loop end
            if abs(new_values[(state.x, state.y)] - old_values[(state.x, state.y)]) < epsilon: 
                correct_states += 1

            #policy update
            actions = [action for action in problem.get_actions(state)]
            policy[state.x, state.y] = actions[max(range(len(side_of_q)), key=side_of_q.__getitem__)]

    return policy

def test1(problem, discount_factor, epsilon):
    policy = init_policy(problem)
    for state in problem.get_all_states():
        if problem.is_goal_state(state) or problem.is_terminal_state(state):
            continue
        print("Current state:", state,"Current action:",policy[state.x, state.y])
        print("    ", problem.get_next_states_and_probs(state, policy[state.x, state.y]))
    return policy



if __name__ == "__main__":
    # Initialize the maze environment
    env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS, grad=GRAD, node_rewards=GRID_WORLD3_REWARDS)
    # env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS, grad=GRAD, node_rewards=None)
    # env = kuimaze.MDPMaze(map_image=MAP, probs=PROBS, grad=GRAD, node_rewards=None)
    env.reset()
    '''
    print('====================')
    print('works only in terminal! NOT in IDE!')
    print('press n - next')
    print('press s - skip to end')
    print('====================')
    '''
    #print(env.get_all_states())
    policy = find_policy_via_value_iteration(env, 0.9999, 0.0001)
    #policy = find_policy_via_policy_iteration(env,0.9999)
    #env.visualise(get_visualisation_values(policy))
    #env.render()
    #wait_n_or_s()
    #print('Policy:', policy)
    utils = init_utils(env)
    #env.visualise(get_visualisation_values(utils))
    #env.render()
    #time.sleep(20)
