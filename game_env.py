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
        self.previous_score = 0.
        self.previous_head_y = 0.
        self.no_moving_count = 0

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
            confidence=0.2
        )
        pyautogui.click(x+w//2, y+h//2)
        sleep(.2)

    def terminate(self):
        """
        When stuck in certain position for an infinitely long period
        (> PRESS_DURATION second), terminate it.
        :return: True if terminate
        """

        isEnd = False

        while not isEnd:
            body_state = self.get_variable('globalbodystate')
            left = body_state['leftFoot']['position_x']
            right = body_state['rightFoot']['position_x']

            action = self.ACTIONS_SPACE[1] if left < right else self.ACTIONS_SPACE[0]

            self.press_key(action, 2)
            isEnd = pyautogui.locateOnScreen('game/end_screen.png', confidence=0.2)

        return True

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

        # progress in x direction
        current_score = body_state['torso']['position_x']

        if (current_score == self.previous_score):
            self.no_moving_count += 1
        else:
            self.no_moving_count = 0

        r1 = 3 * (current_score - self.previous_score) if self.scoreTime > 0 else 0
        r1 -= 0.01 * (self.no_moving_count-3) if self.no_moving_count >= 5 else 0
        self.previous_score = current_score

        # speed of head drop
        current_head_y = body_state['head']['position_y']
        r2 = -3 * (current_head_y - self.previous_head_y) if self.scoreTime > 0 else 0
        r2 = min(r2, 0)
        self.previous_head_y = current_head_y

        # foot in correct position
        if body_state['rightFoot']['position_x'] > body_state['leftFoot']['position_x']:
            condition1 = (body_state['rightFoot']['position_y'] >= 8)
            condition2 = (abs(body_state['rightFoot']['angle']) < 0.02)
        else:
            condition1 = (body_state['leftFoot']['position_y'] >= 8)
            condition2 = (abs(body_state['leftFoot']['angle']) < 0.02)
        r3 = condition1 * condition2 * 0.05

        info = {} # {'r':body_state['rightFoot'], 'l':body_state['leftFoot']}

        self.scoreTime = game_state['scoreTime']

        if self.scoreTime > self.MAX_DURATION:
            if self.terminate():
                self.gameover = done = True
        elif (game_state['gameEnded'] > 0) or (game_state['gameOver']) > 0:
            self.gameover = done = True
        else:
            self.gameover = done = False

        r4 = self.gameover * -0.01 * (100 - current_score)**2

        # print('r1:',np.round(r1,2),'r2:',np.round(r2,2),'r3:',np.round(r3,2),'r4:',np.round(r4,2))
        reward = r1 + r2 + r3 + r4
        print(reward)

        states = []
        for body_part in body_state.values():
            for v in body_part.values():
                states.append(v)
        states = np.array(states)

        return states, reward, done, info # {}

    def step(self, i):
        """
        Executes a step in the environment by applying an action.
        Returns the new observation, reward, completion status, and other info.
        :return: self.get_state()
        """

        self.press_key(self.ACTIONS_SPACE[i],self.PRESS_DURATION)
        return self.get_state()

    def press_key(self, actions, duration):
        """
        press actions key(s) for duration seconds to execute action
        """
        for key in actions:
            pyautogui.keyDown(key)
        sleep(duration)
        for key in actions:
            pyautogui.keyUp(key)

    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observation.
        :return: states in self.get_state() only
        """

        self.press_key(['space'], self.PRESS_DURATION)

        # Initialize
        self.gameover = False
        self.score = 0.
        self.scoreTime = 0.
        self.previous_score = 0.
        self.previous_head_y = 0.
        self.no_moving_count = 0

        # return states only
        return self.get_state()[0]

    def render(self, mode="human"):
        pass

    def close(self):
        self.driver.close()

if __name__ == '__main__':
    env = qwopEnv()
    s = time()
    count = 0
    while True:
        if env.gameover:
            env.reset()
            s = time()
        else:
            # return obs, reward, done, info from step function
            obs, reward, done, _ = env.step(10) # env.action_space.sample())
            e = time()
            if count%10==0:
                print('time for one iter:', e-s)
                # print('reward:', reward)
            count+=1