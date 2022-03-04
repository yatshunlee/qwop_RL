import cv2
import numpy as np
import pyautogui, webbrowser, pytesseract

from time import sleep

# you have to download from https://github.com/UB-Mannheim/tesseract/wiki first
# https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i
pytesseract.pytesseract.tesseract_cmd = 'C:/Users/User/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

# turn on the game on a browser
webbrowser.open('http://www.foddy.net/Athletics.html')
sleep(5)

# locate the game screen
x, y, w, h = pyautogui.locateOnScreen(
    'start_screen.png',
    confidence=0.7
)
# initiate the game
pyautogui.click(x+w//2, y+h//2)
sleep(.2)

def get_reward():
    global x, y, w, h
    # corner of reward box in the progress
    im = pyautogui.screenshot(
        region=(x+w//4, y, w//2, h//6)
    )
    # OCR the msg of the reward
    msg = pytesseract.image_to_string(im)
    reward = msg.split()[-2]
    print(reward)
    return float(reward)

def lose():
    # corner of msg box in the terminal state
    end = pyautogui.locateOnScreen(
        'end_screen.png',
        confidence=0.7)
    if end:
        return True
    else:
        False

while True:
    # action = 'Q', 'W', 'O', 'P'
    # time of control in range of (0,1]

    # restart the game if lose
    if lose():
        last_reward = get_reward()
        pyautogui.keyDown('space')
        sleep(0.1)
        pyautogui.keyUp('space')

    # read reward
    reward = get_reward()

    # sample from action space
    kb_choice = np.random.choice(['Q','W','O','P'])
    time_choice = np.random.random()

    # execute action
    pyautogui.keyDown(kb_choice)
    sleep(time_choice)
    pyautogui.keyUp(kb_choice)