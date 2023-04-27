#   o   o   o   o   o
#   o   o   x   x   o   
#   o   x   v   o   o
#   o   x   o   o   o
#   R   o   o   o   o
# R: robot, x: trap, reward=-1, v: heaven, reward=1 o: places to walk

# agent: for example, a neural network. agent has inputs as State and outputs as Policy
# Actions: left, right, up, down
# State: the inputs of the agent, for example the locations of the agent
# Environment: The maze that can take some actions from the agent and return state and rewards
# Observation: The agent has to observe the environment to make actions. By observing the environment, the agent get state
# s1, based on s1, the agent makes an action a1, then this action acts on the environment and the agent can get a reward
# r_{t+1}, and the agent moves to state s2, 

import numpy as np
import time, sys
import tkinter as tk 

UNIT = 40   # pixels
MAZE_H = 5
MAZE_W = 5  # maze h, w

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']    # all the actions
        self.n_actions = len(self.action_space)     # total # of actions
        self.title('maze')
        self.height = MAZE_H * UNIT
        self.width = MAZE_W * UNIT
        self.geometry('{0}x{1}'.format(self.height, self.width))
        self._build_maze()      # build maze

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=self.height, width=self.width)
        # create grids
        for c in range(0, self.width, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.height
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.height, UNIT):
            x0, y0, x1, y1 = 0, r, self.width, r 
            self.canvas.create_line(x0, y0, x1, y1)
        # create orgin
        origin = np.array([20, 20])
        # hell
        hell1_center = origin + np.array([UNIT * 3, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        # hell
        hell3_center = origin + np.array([UNIT, UNIT * 3])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
        # hell
        hell4_center = origin + np.array([UNIT * 2, UNIT])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')
        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # left
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # right
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3), self.canvas.coords(self.hell4)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

# if __name__ == '__main__':
#     env = Maze()
#     env.after(100, update)
#     env.mainloop()

import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

from study.RLearning.maze_env import Maze
from study.RLearning.RL_brain import QLearningTable
import pandas as pd

def update():
    # 跟着行为轨迹
    df = pd.DataFrame(columns=('state','action_space','reward','Q','action'))
    # 转换为迷宫坐标（x,y）
    def set_state(observation):
        p = []
        p.append(int((observation[0]-5)/40))
        p.append(int((observation[1]-5)/40))
        return p
    for episode in range(100):
        # initial observation
        observation = env.reset()
        observation = set_state(observation)

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            if observation_ != 'terminal':
                observation_ = set_state(observation_)                                    
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))
            q = RL.q_table.loc[str(observation),action]
            df = df.append(pd.DataFrame({'state':[observation],'action_space':[env.action_space[action]],'reward':[reward],'Q':[q],'action':action}), ignore_index=True)
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    df.to_csv('action.csv')
    RL.q_table.to_csv('q_table.csv')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()