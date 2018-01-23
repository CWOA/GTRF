import os, sys
const_path = os.path.abspath(os.path.join(__file__,'..','..'))
solver_path = os.path.abspath(os.path.join(__file__,'..','..','Solvers'))
sys.path.append(const_path)
sys.path.append(solver_path)

import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import SequenceSolver
import Constants as const

class gameOb():
    def __init__(   self,
                    coordinates,
                    color,
                    is_agent,
                    reward=1.0            ):
        self._x = coordinates[0]
        self._y = coordinates[1]
        self._color = color

        if is_agent:
            self._is_agent = True
        else:
            self._is_agent = False
            self._visited = False
            self._reward = reward
        
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
        self._move_penalty = -0.01

        a = self.reset()
        plt.imshow(a,interpolation="nearest")
        
    def reset(self):
        # Add the agent
        rand_agent_x = random.randint(0, self._sizeX)
        rand_agent_y = random.randint(0, self._sizeY)
        self._agent = gameOb((rand_agent_x, rand_agent_y), constants.AGENT_COLOUR, True)

        # Reset and add targets
        self._targets = []
        for i in range(self._num_targets):
            target = gameOb(self.newPosition(), constants.TARGET_COLOUR, False)
            self._targets.append(target)

        # Solve the episode using the globally-optimal sequence solver
        self.solver.reset(self._agent, self._targets)
        min_actions = self.solver.solve()

        # Create the current state
        self._state = self.renderEnv()

        return self._state

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
            rand_x = random.randint(0, self._sizeX)
            rand_y = random.randint(0, self._sizeY)

            ok = True

            # Check the generated position isn't already in use
            for pos in occupied:
                if rand_x == pos[0] and rand_y == pos[1]:
                    ok = False
                    break

            if ok: return (rand_x, rand_y)

    def checkGoal(self):
        # Get game objects that aren't the agent (just targets)
        targets = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                targets.append(obj)

        reward = 0

        # Iterate over all targets
        for target in self._targets:
            # If the agent is over an unvisited target
            if self._agent._x == target._x and self._agent._y == target._y and not target._visited:
                reward = target._reward
                target._visited = True

        # Check whether all targets have now been visited
        for target in self._targets:
            if not target._visited:
                return reward, False

        # If we're here, all targets have now been visited
        return reward, True

    def renderEnv(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self._sizeY+2, self.size_X+2, 3])
        a[1:-1,1:-1,:] = 0
    
        for item in self._targets:
            a[  item.y+1:item.y+item.size+1,
                item.x+1:item.x+item.size+1,
                :                               ] = item._color

        # Get the agent position
        a_x = self._agent._x
        a_y = self._agent._y

        if self._partial == True:
            a = a[a_y:a_y+3, a_x:a_x+3,:]

        b = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest')
        c = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        d = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')

        a = np.stack([b,c,d], axis=2)
        return a

    def step(self, action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done
        else:
            return state,(reward+penalty),done