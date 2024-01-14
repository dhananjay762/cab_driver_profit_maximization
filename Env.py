# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = [(i,j,k) for i in range(m) for j in range(t) for k in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


        
    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0]*(m+t+d)
        state_encod[state[0]] = 1       #get location
        state_encod[m+state[1]] = 1     #get time
        state_encod[m+t+state[2]] = 1   #get day
        return state_encod

    

    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        state_encod = [0]*(m+t+d+m+m)
        state_encod[state[0]] = 1       #get location
        state_encod[m+state[1]] = 1     #get time
        state_encod[m+t+state[2]] = 1   #get day
        if(action[0]!=0):
            state_encod[m+t+d+action[0]] = 1        #get pickup location
        if(action[1]!=0):
            state_encod[m+t+d+m+action[1]] = 1      #get drop location
        return state_encod


    
    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions   



#     def reward_func(self, state, action, Time_matrix):
#         """Takes in state, action and Time-matrix and returns the reward"""
        
#         return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []
        
        wait_time = 0
        curr_pickup_time = 0
        pickup_drop_time = 0
        
        # Get current location, time and day
        curr_loc = int(state[0])
        curr_time = int(state[1])
        curr_day = int(state[2])
        
        # Get requested location
        pickup_loc = int(action[0])
        drop_loc = int(action[1])
        
        # When request is refused - wait time is 1 
        if((pickup_loc==0) and (drop_loc==0)):
            wait_time = 1
            next_loc = curr_loc
        # When driver is at pickup point already, only ride time (from pickup to drop) is considered
        elif(pickup_loc==curr_loc):
            pickup_drop_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            next_loc = drop_loc
        # When driver is not at pickup point 
        else:
            # Time to reach pickup point
            curr_pickup_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            updated_time, updated_day = self.update_time(curr_time, curr_day, int(curr_pickup_time))
            # Time taken to drop the passenger
            pickup_drop_time = Time_matrix[pickup_loc][drop_loc][updated_time][updated_day]
            next_loc = drop_loc
        
        # Total time taken for a ride to complete
        total_time = wait_time + int(curr_pickup_time) + int(pickup_drop_time)
        next_time, next_day = self.update_time(curr_time, curr_day, total_time)
        # Calculate next_state based on new location and time details
        next_state = [next_loc, next_time, next_day]
        
        # Calculate Reward
        ride_time = pickup_drop_time
        idle_time = wait_time + curr_pickup_time
        
        reward = R*ride_time -C*(ride_time + idle_time)
        
        return next_state, reward, total_time

    
    def step(self, state, action, Time_matrix):
        # Get next state, rewards and total time
        next_state, rewards, total_time = self.next_state_func(state, action, Time_matrix)
        return next_state, rewards, total_time


    
    def update_time(self, time, day, transit_duration):
        updated_time = (time + transit_duration)%24
        new_day = (time + transit_duration)//24
        updated_day = (day + new_day)%7
        return int(updated_time), int(updated_day)
    

    def reset(self):
        return self.action_space, self.state_space, self.state_init
