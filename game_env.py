import pyautogui
import numpy as np
from time import sleep, time
from selenium import webdriver

# turn on the game on a browser
driver = webdriver.Chrome()
driver.get('http://localhost:8000/Athletics.html')
sleep(3)
#Since the pyautogui.locateOnScreen can only get the value of picture comes from pyautogui.screenshot, So add below code to capture the game screen
cap = pyautogui.screenshot(region=(26, 142, 640, 400))
cap.save('game/start.png')
# initiate the game
# locate the game screen
x, y, w, h = pyautogui.locateOnScreen(
    'game/start_screen.png',
    confidence=0.7
)
# initiate the game
pyautogui.click(x+w//2, y+h//2)
sleep(.2)
time_choice = 0.1 # Press Duration
reset = True # Check if the pressing time > time_choice

def get_variable(var_name):
    global driver
    return driver.execute_script(f'return {var_name};')

def get_state():
    game_state = get_variable('globalgamestate')
    body_state = get_variable('globalbodystate')
    # print(game_state)
    # print(body_state)
    gameover = True if (game_state['gameEnded'] > 0) or (game_state['gameOver'] > 0) else False

    return gameover, game_state, body_state

while True:
    s = time()

    # get state
    # maybe have to create a memory if not detected
    gameover, game_state, body_state = get_state()

    # restart the game if lose
    if gameover:
        pyautogui.keyDown('space')
        sleep(0.1)
        pyautogui.keyUp('space')

    # action space
    kb_choice = np.random.choice([
            'Q', 'W', 'O', 'P',
            'QW', 'QO', 'QP',
            'WO', 'WP', ''
    ])

    # execute action method 1
    for key in kb_choice:
        pyautogui.keyDown(key)
    sleep(time_choice)
    for key in kb_choice:
        pyautogui.keyUp(key)

    e = time()
    print('time:', e-s)
