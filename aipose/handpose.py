import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class HandPoseAnalyzer:
    landmark_names = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]

    def __init__(self):
        self.model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        self.model_path = 'hand_landmarker.task'
        self.download_model(self.model_url, self.model_path)
        self.setup_detector()

    def setup_detector(self):
        # Setup the hand landmark detector with the model file
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def download_model(self, url, save_path):
        # Download the model file if it's not already present
        if not os.path.exists(save_path):
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                print(f"Model downloaded and saved as {save_path}")
            except requests.RequestException as e:
                print(f"An error occurred while downloading the model: {e}")

    def analyze_hand_pose(self, image_path):
        # Load the input image and detect hand landmarks
        image = mp.Image.create_from_file(image_path)
        detection_result = self.detector.detect(image)
        if not detection_result.hand_landmarks:
            return "No hands detected. Please take another picture."

        return self.get_landmarks_string(detection_result)

    def get_landmarks_string(self, detection_result):
        results = ""
        for i, handedness_list in enumerate(detection_result.handedness):
            for handedness in handedness_list:
                # Analyze hand pose
                landmarks = detection_result.hand_landmarks[i]
                results += self.analyze_hand_bend(landmarks)
                results += self.analyze_wrist_flexion(landmarks)
                results += self.analyze_claw_grip(landmarks)
                results += self.analyze_finger_extension(landmarks)
        return results

    def analyze_hand_bend(self, landmarks):
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]

        if middle_tip.y < middle_mcp.y and middle_tip.y < wrist.y:
            return "  Hand is bent inwards.\n"
        elif middle_tip.y > middle_mcp.y and middle_tip.y > wrist.y:
            return "  Hand is bent outwards.\n"
        else:
            return "  Hand is not bent inwards or outwards.\n"

    def analyze_wrist_flexion(self, landmarks):
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        if index_mcp.y < wrist.y and pinky_mcp.y < wrist.y:
            return "  Wrist is flexed upwards.\n"
        elif index_mcp.y > wrist.y and pinky_mcp.y > wrist.y:
            return "  Wrist is flexed downwards.\n"
        else:
            return "  Wrist is not flexed upwards or downwards.\n"

    def analyze_claw_grip(self, landmarks):
        threshold = 0.1
        bent_fingers = sum(
            np.linalg.norm(np.array([landmarks[tip_index].x, landmarks[tip_index].y]) - 
                           np.array([landmarks[tip_index - 2].x, landmarks[tip_index - 2].y])) < threshold
            for tip_index in [8, 12, 16, 20]
        )

        return "  Claw grip detected.\n" if bent_fingers >= 3 else "  Claw grip not detected.\n"

    def analyze_finger_extension(self, landmarks):
        extended_fingers = sum(
            landmarks[tip_index].y < landmarks[tip_index - 1].y < landmarks[tip_index - 2].y
            for tip_index in [8, 12, 16, 20]
        )

        return "  Fingers are extended.\n" if extended_fingers >= 3 else "  Fingers are not extended.\n"

