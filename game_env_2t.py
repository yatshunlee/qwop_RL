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
    
    # MAX_DURATION = 1.6 # max seconds per one round
    NUM_STATES = 71 # total num of body states
    key_space = {
        0: 'q', 1: 'w', 2: 'o', 3: 'p',
        4: 'qw', 5: 'qo', 6: 'qp', 7: 'wo',
        8: 'wp', 9: 'op', 10: ''
    }
    time_space = { 0 : 0.15, 1: 0.4}
    k=0
    l = dict()
    for i in key_space:
        for j in time_space:
            l2 ={ k: [key_space[i] , time_space[j]]}
            l[k] = l2
            k=k+1
            
 
    ACTIONS_SPACE = l
    
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
        self.previous_score = 0
        self.mean_speed = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self.game_start()
        self.MAX_DURATION = 80 # max seconds per one round
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
        :return: obs, reward, done, infoQ
        """
        
       
        alpha = 0.1 # weigh for velocity
        self.MAX_DURATION = 60
        game_state = self.get_variable('globalgamestate')
        body_state = self.get_variable('globalbodystate')
        torso_x = body_state['torso']['position_x']
        torso_y = body_state['torso']['position_y']   
        torso_a = body_state['torso']['angle']
        time = game_state['scoreTime']
        if (game_state['gameEnded'] > 0) or (game_state['gameOver'] > 0) or (time > self.MAX_DURATION):
            self.gameover = done = True
        else:
            self.gameover = done = False
        
        self.mean_speed = alpha * (game_state['score'] - self.previous_score) + (1-alpha ) * self.mean_speed
        
        states = []
        for body_part in body_state.values():
            for v in body_part.values():
                states.append(v)
        states = np.array(states)
        #g = game_state
        #print(body_state)
        #print(self.mean_speed)
        #self.PRESS_DURATION = 0.15 # action duration
        # Get reward

        r1 =  4 * max(game_state['score'] +0.4,0) ** 2  # initial distance travelled
       # r2 = 0 #max( time,10) /10     # penalize for staying
       # r3 =  0.5 * max(game_state['score'] - self.previous_score, 0)   ## foward
        r4 = -torso_y/6  ## reward for standing up
        r5 =  max(abs((body_state['joints']['leftKnee'] + body_state['joints']['rightKnee'])/2) - 0.6,0) ## penalty for knee angle
        if self.mean_speed>0 :
            r6 =  3* self.mean_speed ** 1  #aveage velocity
        else :
            r6 = -3* (-self.mean_speed) ** 1
        r7 = max(abs((torso_a+1.45))-0.7,0)  #penalty for too big angle
        
        if (game_state['gameEnded'] == 0) :
            reward =  r1 * (r6+0.00) * ( 1 + max(min(r4,0.6),0)) * (1-min(r5*0.3,1))*(1-min(r7*2,1)) 
        else :
            reward = -2
       # print("angle:{}".format(torso_a))
        #print("reward:{}, r1:{} , r5: {:4f}, r7: {:4f}".format(reward,r1,r5,r7))
       # print("reward:{:4f}, angle:{:4f} ".format(reward,torso_a))
        #print("reward:{:4f} ".format(reward))
        #print("left: {:4f}".format(body_state['joints']['leftKnee']))
       # print("right: {:4f}".format(body_state['joints']['rightKnee']))
        self.previous_torso_x = torso_x
        self.previous_torso_y = torso_y
        self.previous_score = game_state['score']
        self.previous_time = game_state['scoreTime']
        
        return states, reward, done, {}

    def step(self, i):
        """
        Executes a step in the environment by applying an action.
        Returns the new observation, reward, completion status, and other info.
        :return: self.get_state()
        """
        # execute action
        for key in self.ACTIONS_SPACE[i].items():
           # print(key[1])
            for x in key[1][0]:
                pyautogui.keyDown(x)
            sleep(key[1][1])
        for key in self.ACTIONS_SPACE[i].items():
            for x in key[1][0]:
                pyautogui.keyUp(x)
            print("action : {}".format((key[1])))
        return self.get_state()

        
    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observation.
        :return: states in self.get_state() only
        """
        # restart the game
        pyautogui.keyDown('r')
        sleep(0.1)
        pyautogui.keyUp('r')

        # Initialize
        self.gameover = False
        self.score = 0.
        self.scoreTime = 0.
        self.previous_score = 0.
        self.mean_speed = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        # return states only
        sleep(0.7)
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