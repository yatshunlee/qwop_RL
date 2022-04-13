# qwop_RL
It is one of the hardest games in the world. You can try on http://www.foddy.net/Athletics.html.
Our objective is to:
1) train up a model to complete 100m in a reasonable time
2) improve the speed of the ragdoll to a competitive level
3) converged the model to a stable running pattern
4) to create the best record on our own

# Setup Procedures
1) Run `pip install -r requirements.txt`
2) Make sure you have installed chromedriver. If no, https://chromedriver.chromium.org/downloads. Store it into the ~/qwop_RL dir.
3) Create a terminal to host game:
    `python host_game.py`
4) Create another terminal to train the agent. You can train either on CPU or GPU environment.

# Proposed Methodology
![image](https://user-images.githubusercontent.com/69416199/163233261-24721c28-1641-4fdf-bd45-5d97c5a6f57d.png)

# Reward Design
![image](https://user-images.githubusercontent.com/69416199/163232728-0fad0aa0-7d4d-484c-9199-214eb28e1348.png)

# Training Flowchart
Our approach is through DQN and DDQN to train our agent to complete the 100m racing game. Following diagram illustrates the entire process of updating Q network for approximating the value function by DQN with Target Network as well as DDQN. They share the same updating approach except the way they calculate for TD target.

![image](https://user-images.githubusercontent.com/69416199/161551941-4612d814-ab15-4d43-97c1-3109cd0eca6a.png)

# Training
Before training the agent, you can configure the training parameters as well as the action/state/reward design. You can choose to run the training by:
1) Deep Q Network with Target Network: `python dqn_main.py --train`, or `python dqn_main.py --retrain`
2) Double Deep Q Network: `python ddqn_train.py`

# Testing
1) Deep Q Network with Target Network: `python dqn_main.py --test`
2) Double Deep Q Network: `python ddqn_test.py`

# Videos of Performance
- Training Process: https://drive.google.com/file/d/15IC227quOzvnp3kPCHc8gTxv3fjIBRTY/view?usp=sharing
- Avg. performance (5~10 min): https://drive.google.com/file/d/1g4-nemSH48i-Dl_-h3mAmzq_HFhBOK1b/view?usp=sharing
- Best (3 min 15 sec): https://drive.google.com/file/d/122VpHFDevAKw39ia4rBXRtUYgZRyO8WB/view?usp=sharing

# Credits
- to the author of QWOP http://www.foddy.net/Athletics.html
- to @Wesleyliao for the game engine and help: https://github.com/Wesleyliao/QWOP-RL
