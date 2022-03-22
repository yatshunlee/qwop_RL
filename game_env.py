import pyautogui
import numpy as np
from time import sleep, time
from selenium import webdriver
from gym import Env, spaces

class qwopEnv(Env):
    """
    Custom Environment that follows gym interface.
    """

    # Game settings
    PRESS_DURATION = 0.1 # action duration
    MAX_DURATION = 90 # max seconds per one round
    NUM_STATES = 71 # total num of body states
    ACTIONS_SPACE = {
        0: 'Q', 1: 'W', 2: 'O', 3: 'P',
        4: 'QW', 5: 'QO', 6: 'QP', 7: 'WO',
        8: 'WP', 9: 'OP', 10: ''
    }

    def __init__(self):
        super(qwopEnv, self).__init__()

        # Define a discrete action space ranging from 0 to 10
        self.action_space = spaces.Discrete(len(self.ACTIONS_SPACE))
        # Define a 1-D observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[self.NUM_STATES], dtype=np.float32
        )

        # Initialize (temp)
        self.gameover = False
        self.score = 0.
        self.scoreTime = 0.

        self.game_start()

    def game_start(self):
        """
        Start the game on a browser
        :return:
        """

        self.driver = webdriver.Chrome()
        self.driver.get('http://localhost:8000/Athletics.html')
        sleep(3)

        # locate the game screen and auto click
        x, y, w, h = pyautogui.locateOnScreen(
            'game/start_screen.png',
            confidence=0.7
        )
        pyautogui.click(x+w//2, y+h//2)
        sleep(.2)

    def get_variable(self,var_name):
        """
        Retrieve variables from localhost (game)
        :return: specific variable
        """
        return self.driver.execute_script(f'return {var_name};')

    def get_state(self):
        """
        Retrieve game states as well as body states from JS adaptor.
        :return: obs, reward, done, info
        """
        game_state = self.get_variable('globalgamestate')
        body_state = self.get_variable('globalbodystate')

        reward = game_state['score']
        time = game_state['scoreTime']

        if (game_state['gameEnded'] > 0) or (game_state['gameOver'] > 0) or (time > self.MAX_DURATION):
            self.gameover = done = True
        else:
            self.gameover = done = False

        states = []
        for body_part in body_state.values():
            for v in body_part.values():
                states.append(v)
        states = np.array(states)

        return states, reward, done, []

    def step(self, i):
        """
        Executes a step in the environment by applying an action.
        Returns the new observation, reward, completion status, and other info.
        :return: self.get_state()
        """
        # execute action
        for key in self.ACTIONS_SPACE[i]:
            pyautogui.keyDown(key)
        sleep(self.PRESS_DURATION)
        for key in self.ACTIONS_SPACE[i]:
            pyautogui.keyUp(key)

        return self.get_state()

    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observation.
        :return: states in self.get_state() only
        """
        # restart the game
        pyautogui.keyDown('space')
        sleep(0.1)
        pyautogui.keyUp('space')

        # Initialize
        self.gameover = False
        self.score = 0.
        self.scoreTime = 0.

        # return states only
        return self.get_state()[0]

    def render(self, mode="human"):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    env = qwopEnv()

    while True:
        if env.gameover:
            env.reset()
        else:
            s = time()
            # return obs, reward, done, info from step function
            env.step(env.action_space.sample())
            e = time()
            print('time for one iter:', e-s)