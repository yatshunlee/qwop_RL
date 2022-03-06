import cv2
import numpy as np
import pyautogui, webbrowser, pytesseract

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from time import sleep, time

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

time_choice = 0.1 # Press Duration
reset = True # Check if the pressing time > time_choice
last_keypoints = [] # temp solution for undetectable case

def get_reward():
    global x, y, w, h
    # corner of reward box in the progress
    im = pyautogui.screenshot(
        region=(x+w//4, y, w//2, h//6)
    )

    # OCR the msg of the reward
    msg = pytesseract.image_to_string(im)
    reward = msg.split()[-2]
    print('reward:', reward)
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


def get_state(last_keypoints, draw=False):
    # tutorial of pose estimation
    # https://google.github.io/mediapipe/solutions/pose.html#output

    # get game screen
    img = pyautogui.screenshot(
        region=(x, y, w, h)
    )

    # process game screenshot
    img = np.array(img)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    results = pose.process(img)

    # temporarily
    if not results.pose_landmarks:
        print('Cant find state')
        cv2.imshow('img',img)
        cv2.waitKey(0)
        return last_keypoints

    # get specific joint
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            results.pose_landmarks.landmark[0],  # nose
            results.pose_landmarks.landmark[11],  # left shoulder
            results.pose_landmarks.landmark[13],  # left elbow
            results.pose_landmarks.landmark[15],  # left hand
            results.pose_landmarks.landmark[12],  # right shoulder
            results.pose_landmarks.landmark[14],  # right elbow
            results.pose_landmarks.landmark[16],  # right hand
            results.pose_landmarks.landmark[23],  # left hip
            results.pose_landmarks.landmark[25],  # left knee
            results.pose_landmarks.landmark[27],  # left ankle
            results.pose_landmarks.landmark[24],  # right hip
            results.pose_landmarks.landmark[26],  # right knee
            results.pose_landmarks.landmark[28],  # right ankle
        ]
    )

    # see the joint on the video capture
    if draw:
        mp_drawing.draw_landmarks(img, landmark_subset)  # mp_pose.POSE_CONNECTIONS
        cv2.imshow('output', img)
        cv2.waitKey(10)

    # reformat
    keypoints = []
    for data_point in landmark_subset.landmark:
        keypoints.append({
            'X': data_point.x,
            'Y': data_point.y,
            'Z': data_point.z,
            'Visibility': data_point.visibility,
        })

    return keypoints

while True:
    s = time()

    # restart the game if lose
    if lose():
        last_reward = get_reward()
        pyautogui.keyDown('space')
        sleep(0.1)
        pyautogui.keyUp('space')

    # read reward
    reward = get_reward()

    # get state
    # maybe have to create a memory if not detected
    keypoints = get_state(last_keypoints, draw=False)
    last_keypoints = keypoints

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

    # execute action method 2
    # if reset:
    #     # sample from action space
    #     kb_choice = np.random.choice([
    #         'Q', 'W', 'O', 'P',
    #         'QW', 'QO', 'QP',
    #         'WO', 'WP', ''
    #     ])
    #
    #     # press timer
    #     s = time()
    #     for key in kb_choice:
    #         pyautogui.keyDown(key)
    #     # not allow new sample action
    #     reset = False
    #
    # e = time()
    # print('time:', e-s)
    # if e-s > time_choice:
    #     for key in kb_choice:
    #         pyautogui.keyUp(key)
    #     # allow new sample action
    #     reset = True