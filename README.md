# qwop_RL
One of the hardest game in the world. You can try to play by yourself on http://www.foddy.net/Athletics.html.

# Setup procedures:
1) Run `pip install -r requirements.txt`
2) Make sure you have installed chromedriver. If no, https://chromedriver.chromium.org/downloads. Store it into the ~/qwop_RL dir.
3) Create a terminal to host game:
    `python host_game.py`
4) Create another terminal to train the agent. You can train either on CPU or GPU environment.

# Flowchart of Training by DQN with Target Network as well as DDQN
Our approach is through DQN and DDQN to train our agent to complete the 100m racing game.
![image](https://user-images.githubusercontent.com/69416199/161551941-4612d814-ab15-4d43-97c1-3109cd0eca6a.png)

# Training
Before training the agent, you can configure the training parameters as well as the action/state/reward design. You can choose to run the training by:
1) Deep Q Network with Target Network: `python dqn_main.py --train`, or
2) Double Deep Q Network: `python ddqn_train.py`

Credit to @Wesleyliao for the game environment settings: https://github.com/Wesleyliao/QWOP-RL.
