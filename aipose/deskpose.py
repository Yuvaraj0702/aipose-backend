import mediapipe as mp
import cv2
import numpy as np

class DeskPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)
        self.CONFIDENCE_THRESHOLD = 0.2
        self.ANGLE_THRESHOLD_LOW = 50
        self.ANGLE_THRESHOLD_HIGH = 130
        self.BODY_TOLERANCE = 0.15
        self.NECK_ANGLE_TOLERANCE = 5
        self.WRIST_ELBOW_VERTICAL_TOLERANCE = 0.1
        self.SHOULDER_HIP_ANGLE_THRESHOLD = 160
        self.SHOULDER_BALANCE_TOLERANCE = 0.1

    @staticmethod
    def calculate_angle(point1, point2, point3):
        a = point1 - point2
        b = point3 - point2
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        angle = np.arccos(dot_product / (magnitude_a * magnitude_b))
        return np.degrees(angle)

    @staticmethod
    def calculate_horizontal_angle(point1, point2):
        vector = point2 - point1
        horizontal = np.array([1, 0])
        dot_product = np.dot(vector, horizontal)
        magnitude_vector = np.linalg.norm(vector)
        if magnitude_vector == 0:
            return 0
        angle = np.arccos(dot_product / magnitude_vector)
        return np.degrees(angle) - 90

    @staticmethod
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image file. Please check the image path and format.")
        return image

    def analyze_pose(self, image_path):
        image = self.preprocess_image(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self.mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                return "Improper picture. Please take a better picture.", None, None

            keypoints_with_scores = np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark])
            keypoints = keypoints_with_scores[:, :2]
            scores = keypoints_with_scores[:, 2]

            keypoints_of_interest_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            low_confidence_points = np.sum(scores[keypoints_of_interest_indices] < self.CONFIDENCE_THRESHOLD)
            if low_confidence_points / len(keypoints_of_interest_indices) > 0.75:
                return "Improper picture. Please take a better picture.", keypoints_with_scores, scores

            nose, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist = keypoints[[0, 3, 4, 5, 6, 7, 8, 9, 10]]

            facing_side = "left" if abs(nose[0] - left_ear[0]) < abs(nose[0] - right_ear[0]) else "right" if abs(nose[0] - left_ear[0]) > abs(nose[0] - right_ear[0]) else "ambiguous"

            results_text = ""

            if facing_side != "ambiguous":
                shoulder, elbow, wrist = (right_shoulder, right_elbow, right_wrist) if facing_side == "left" else (left_shoulder, left_elbow, left_wrist)

                shoulder_elbow_wrist_angle = self.calculate_angle(shoulder, elbow, wrist)
                neck = (left_shoulder + right_shoulder) / 2
                neck_angle = self.calculate_horizontal_angle(neck, nose)

                if shoulder_elbow_wrist_angle < self.ANGLE_THRESHOLD_LOW:
                    results_text += "positive\n"
                elif shoulder_elbow_wrist_angle > self.ANGLE_THRESHOLD_HIGH:
                    results_text += "Negative\n"
                else:
                    results_text += "Neutral.\n"

                shoulder_wrist_distance = np.linalg.norm(shoulder - wrist) * 2
                if shoulder_wrist_distance > self.BODY_TOLERANCE:
                    results_text += "Negative\n"
                elif shoulder_wrist_distance < self.BODY_TOLERANCE / 2:
                    results_text += "Positive\n"
                else:
                    results_text += "Neutral\n"

                if neck_angle > self.NECK_ANGLE_TOLERANCE:
                    results_text += "Positive\n"
                elif neck_angle < -self.NECK_ANGLE_TOLERANCE:
                    results_text += "Negative\n"
                else:
                    results_text += "Neutral\n"

                # if abs(wrist[1] - elbow[1]) > self.WRIST_ELBOW_VERTICAL_TOLERANCE:
                #     results_text += "Wrist higher than elbow.\n" if wrist[1] > elbow[1] else "Wrist lower than elbow.\n"

                # shoulder_hip_angle = self.calculate_angle(left_shoulder, neck, right_shoulder)
                # results_text += "Back is not straight.\n" if shoulder_hip_angle < self.SHOULDER_HIP_ANGLE_THRESHOLD else "Back is straight.\n"

                # if abs(left_shoulder[1] - right_shoulder[1]) > self.SHOULDER_BALANCE_TOLERANCE:
                #     results_text += "Leaning to the right.\n" if left_shoulder[1] > right_shoulder[1] else "Leaning to the left.\n"
                # else:
                #     results_text += "Body is well balanced.\n"

            return results_text
