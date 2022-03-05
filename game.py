import cv2
import numpy as np
import pyautogui, webbrowser, pytesseract

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

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


def get_state(draw=False):
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

    # get state
    keypoints = get_state(draw=False)
    print(keypoints)

    # sample from action space
    kb_choice = np.random.choice(['Q','W','O','P'])
    time_choice = 0.1 # np.random.random()

    # execute action
    pyautogui.keyDown(kb_choice)
    sleep(time_choice)
    pyautogui.keyUp(kb_choice)