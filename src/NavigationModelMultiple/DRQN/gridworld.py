import os, sys
sys.path.append('../')

import cv2
import copy
import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
from Solvers.SequenceSolver import SequenceSolver
import Constants as const
from Core.Object import Object
from Utilities.DiscoveryRate import DiscoveryRate
        
class gameEnv():
    def __init__(   self,
                    partial,
                    size,
                    num_targets        ):

        # Epsidoe solver method
        self._solver = SequenceSolver()

        # Square dimensions
        self._sizeX = size
        self._sizeY = size

        # POMDP
        self._partial = partial

        # Number of targets to visit
        self._num_targets = num_targets

        # Penalty per-move
        self._move_penalty = -0.00

        # For handling target discovery rate statistics
        self._dr = DiscoveryRate()

        a = self.reset()
        #plt.imshow(a,interpolation="nearest")
        
    def reset(self):
        # Unique identifier counter
        id_ctr = 0

        # Add the agent
        rand_agent_x = random.randint(0, self._sizeX-1)
        rand_agent_y = random.randint(0, self._sizeY-1)
        self._agent = Object(id_ctr, True, x=rand_agent_x, y=rand_agent_y)

        # Increment unique identifier counter
        id_ctr += 1

        # Reset and add targets
        self._targets = []
        for i in range(self._num_targets):
            t_x, t_y = self.newPosition()
            target = Object(id_ctr, False, x=t_x, y=t_y)
            self._targets.append(target)
            id_ctr += 1

        # Solve the episode using the globally-optimal sequence solver
        self._solver.reset(self._agent, self._targets)
        min_actions, _ = self._solver.solve()

        # Create the current state
        self._state = self.renderEnv()

        # Reset the DT metric
        self._dr.reset()

        return self._state, min_actions

    def moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        heroX = self._agent._x
        heroY = self._agent._y
        penalize = 0.

        if direction == 0 and self._agent._y >= 1:
            self._agent._y -= 1
        if direction == 1 and self._agent._y <= self._sizeY-2:
            self._agent._y += 1
        if direction == 2 and self._agent._x >= 1:
            self._agent._x -= 1
        if direction == 3 and self._agent._x <= self._sizeX-2:
            self._agent._x += 1

        # Do we penalise the agent for making a non-move
        # (e.g. moving up or left when it's in the top-left corner)
        if self._agent._x == heroX and self._agent._y == heroY:
            penalize = 0.0

        return penalize + self._move_penalty
    
    def newPosition(self):
        # List of occupied positions
        occupied = []

        # Combine all generated positions up to this point
        occupied.append((self._agent._x, self._agent._y))
        if self._targets is not None:
            for target in self._targets:
                occupied.append((target._x, target._y))

        # Loop until we've generated a valid position
        while True:
            # Generate a position within bounds
            rand_x = random.randint(0, self._sizeX-1)
            rand_y = random.randint(0, self._sizeY-1)

            ok = True

            # Check the generated position isn't already in use
            for pos in occupied:
                if rand_x == pos[0] and rand_y == pos[1]:
                    ok = False
                    break

            if ok: return (rand_x, rand_y)

    def checkGoal(self):
        reward = 0

        self._dr.iterate()

        # Iterate over all targets
        for target in self._targets:
            # If the agent is over an unvisited target
            if self._agent._x == target._x and self._agent._y == target._y and not target._visited:
                reward = 1
                target.setVisited(True)
                self._dr.discovery()

        # Check whether all targets have now been visited
        for target in self._targets:
            if not target._visited:
                return reward, False, self._dr.finish()

        # If we're here, all targets have now been visited
        return reward, True, self._dr.finish()

    def renderEnv(self):
        # Create the image with a one pixel border
        a = np.zeros((self._sizeY+2, self._sizeX+2, 3), np.uint8)

        # Set the border colour to white
        a[:,:,:] = 255

        # Set the background colour
        a[1:-1,1:-1,:] = const.BACKGROUND_COLOUR

        # Render targets
        for t in self._targets:
            if not t._visited:
                a[t._y+1:t._y+2, t._x+1:t._x+2, :] = t._colour
            else:
                a[t._y+1:t._y+2, t._x+1:t._x+2, :] = const.VISITED_COLOUR

        # Get the agent position
        a_x = self._agent._x
        a_y = self._agent._y

        # Render the agent
        a[a_y+1:a_y+2, a_x+1:a_x+2, :] = self._agent._colour

        # If we're only supposed to have a partial view of the environment
        if self._partial == True:
            a = a[a_y:a_y+3, a_x:a_x+3,:]

        # Scale the image up to 84x84 pixels
        scaled = cv2.resize(a, (84, 84), interpolation=cv2.INTER_NEAREST)

        return scaled

    def step(self, action):
        penalty = self.moveChar(action)
        reward,done,dr = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done,dr
        else:
            return state,(reward+penalty),done,dr

# Entry method/unit testing
if __name__ == '__main__':
    env = gameEnv(partial=False,size=10,num_targets=5)
    render = env.renderEnv()
    plt.show(render)
