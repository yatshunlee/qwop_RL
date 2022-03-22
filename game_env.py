import pyautogui
import numpy as np
from time import sleep, time
from selenium import webdriver
from gym import Env, spaces

class qwopEnv(Env):
    """
        Custom Environment that follows gym interface.
    """

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
        # define a 1-D observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[self.NUM_STATES], dtype=np.float32
        )

        # Initialize
        self.gameover = False
        self.score = 0.
        self.scoreTime = 0.

        self.game_start()

    def game_start(self):
        # turn on the game on a browser
        self.driver = webdriver.Chrome()
        self.driver.get('http://localhost:8000/Athletics.html')
        sleep(3)

        # # Since the pyautogui.locateOnScreen can only get
        # # the value of picture comes from pyautogui.screenshot,
        # # so add below code to capture the game screen
        # cap = pyautogui.screenshot(region=(26, 142, 640, 400))
        # cap.save('game/start_screen.png')

        # locate the game screen
        x, y, w, h = pyautogui.locateOnScreen(
            'game/start_screen.png',
            confidence=0.7
        )

        # initiate the game
        pyautogui.click(x+w//2, y+h//2)
        sleep(.2)

    def get_variable(self,var_name):
        # JS adaptor
        return self.driver.execute_script(f'return {var_name};')

    def get_state(self):
        game_state = self.get_variable('globalgamestate')
        body_state = self.get_variable('globalbodystate')

        reward = game_state['score']
        time = game_state['scoreTime']

        if (game_state['gameEnded'] > 0) or (game_state['gameOver'] > 0) or (time > self.MAX_DURATION):
            self.gameover = True
        else:
            self.gameover = False

        states = []
        for body_part in body_state.values():
            for v in body_part.values():
                states.append(v)
        states = np.array(states)

        return states, reward

    def step(self, i):
        """
        Executes a step in the environment by applying an action.
        Returns the new observation, reward, completion status, and other info.
        :return:
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
        :return:
        """
        # restart the game
        pyautogui.keyDown('space')
        sleep(0.1)
        pyautogui.keyUp('space')

        # Initialize
        self.gameover = False
        self.score = 0.
        self.scoreTime = 0.

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
            env.step(env.action_space.sample())
            e = time()
            print('time for one iter:', e-s)