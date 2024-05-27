import mediapipe as mp
import cv2
import numpy as np

class DeskPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    @staticmethod
    def calculate_angle(point1, point2, point3):
        a = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        b = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        angle = np.arccos(dot_product / (magnitude_a * magnitude_b))
        return np.degrees(angle)

    @staticmethod
    def calculate_horizontal_angle(point1, point2):
        vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
        horizontal = np.array([1, 0])
        dot_product = np.dot(vector, horizontal)
        magnitude_vector = np.linalg.norm(vector)
        if magnitude_vector == 0:
            return 0
        angle = np.arccos(dot_product / magnitude_vector)
        return np.degrees(angle) - 90

    @staticmethod
    def preprocess_image(image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Invalid image file. Please check the image path and format.")
        except Exception as e:
            raise ValueError("Error reading the image.") from e
        
        return image

    def analyze_pose(self, image_path):
        image = self.preprocess_image(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return "Improper picture. Please take a better picture.", None, None

        keypoints_with_scores = []
        for landmark in results.pose_landmarks.landmark:
            keypoints_with_scores.append([landmark.x, landmark.y, landmark.visibility])
        keypoints_with_scores = np.array(keypoints_with_scores)

        keypoints = keypoints_with_scores[:, :2]
        scores = keypoints_with_scores[:, 2]

        keypoints_of_interest_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        low_confidence_points = np.sum(scores[keypoints_of_interest_indices] < 0.2)
        if low_confidence_points / len(keypoints_of_interest_indices) > 0.75:
            return "Improper picture. Please take a better picture.", keypoints_with_scores, scores

        nose = keypoints[0]
        left_ear = keypoints[3]
        right_ear = keypoints[4]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        results_text = ""

        if abs(nose[0] - left_ear[0]) < abs(nose[0] - right_ear[0]):
            facing_side = "left"
            shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
        elif abs(nose[0] - left_ear[0]) > abs(nose[0] - right_ear[0]):
            facing_side = "right"
            shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
        else:
            # results_text += "Facing direction is ambiguous or frontal.\n"
            facing_side = "ambiguous"

        if facing_side != "ambiguous":
            # results_text += f"The {facing_side} side of the person is facing the camera.\n"
            shoulder_elbow_wrist_angle = self.calculate_angle(shoulder, elbow, wrist)
            
            neck = (left_shoulder + right_shoulder) / 2
            neck_angle = self.calculate_horizontal_angle(neck, nose)

            if shoulder_elbow_wrist_angle*2 < 100:
                print(shoulder_elbow_wrist_angle)
                results_text += "The desk is too high.\n"
            elif shoulder_elbow_wrist_angle > 130:
                results_text += "Table too low.\n"
            else:
                results_text += "Correct table height.\n"

            shoulder_wrist_distance = np.linalg.norm(np.array(shoulder) - np.array(wrist))*2
            body_tolerance = 0.15
            if shoulder_wrist_distance > body_tolerance:
                results_text += "Table too far.\n"
            elif shoulder_wrist_distance < body_tolerance / 2:
                results_text += "Table too close.\n"
            else:
                results_text += "Table at a good distance.\n"

            if neck_angle > 5:
                results_text += "Looking upwards.\n"
            elif neck_angle < -5:
                results_text += "Looking downwards.\n"
            else:
                results_text += "Good neck position.\n"

            if abs(wrist[1] - elbow[1]) > 0.1:
                if wrist[1] > elbow[1]:
                    results_text += "Wrist higher than elbow.\n"
                else:
                    results_text += "Wrist lower than elbow.\n"

            shoulder_hip_angle = self.calculate_angle(left_shoulder, (left_shoulder + right_shoulder) / 2, right_shoulder)
            if shoulder_hip_angle < 160:
                results_text += "Back is not straight.\n"
            else:
                results_text += "Back is straight.\n"

            if abs(left_shoulder[1] - right_shoulder[1]) > 0.1:
                if left_shoulder[1] > right_shoulder[1]:
                    results_text += "Leaning to the right.\n"
                else:
                    results_text += "Leaning to the left.\n"
            else:
                results_text += "Body is well balanced.\n"

        return results_text
