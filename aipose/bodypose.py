import cv2
import mediapipe as mp
import numpy as np

class PoseAnalyzer:
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
        angle = np.arccos(dot_product / (magnitude_a * magnitude_b))
        return np.degrees(angle)

    @staticmethod
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image file. Please check the image path and format.")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    @staticmethod
    def check_legs_crossed(left_knee, right_knee, left_ankle, right_ankle):
        knees_crossed = abs(left_knee[0] - right_knee[0]) < 0.1
        ankles_crossed = abs(left_ankle[0] - right_ankle[0]) < 0.1
        return knees_crossed or ankles_crossed

    def analyze_pose(self, image_path):
        image = self.preprocess_image(image_path)
        results = self.pose.process(image)

        if not results.pose_landmarks:
            return "Improper picture. Please provide a clearer image.", None, None

        keypoints = []
        scores = []

        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y])
            scores.append(landmark.visibility)

        keypoints = np.array(keypoints)
        scores = np.array(scores)

        # Check confidence levels
        low_confidence_points = np.sum(scores < 0.2)
        if low_confidence_points / len(scores) > 0.75:
            return "Improper picture. Please provide a clearer image.", keypoints, scores

        nose = keypoints[self.mp_pose.PoseLandmark.NOSE.value]
        left_ear = keypoints[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = keypoints[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_shoulder = keypoints[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = keypoints[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = keypoints[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = keypoints[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = keypoints[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = keypoints[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = keypoints[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = keypoints[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        analysis_results = ""

        # Determine facing direction by comparing the horizontal positions of the nose and ears
        if abs(nose[0] - left_ear[0]) < abs(nose[0] - right_ear[0]):
            facing_side = "left"
            shoulder, hip, knee, ankle = right_shoulder, right_hip, right_knee, right_ankle
        elif abs(nose[0] - left_ear[0]) > abs(nose[0] - right_ear[0]):
            facing_side = "right"
            shoulder, hip, knee, ankle = left_shoulder, left_hip, left_knee, left_ankle
        else:
            # analysis_results += "Facing direction is ambiguous or frontal.\n"
            facing_side = "ambiguous"

        if facing_side != "ambiguous":
            # analysis_results += f"The {facing_side} side of the person is facing the camera.\n"
            shoulder_hip_knee_angle = self.calculate_angle(shoulder, hip, knee)
            hip_knee_ankle_angle = self.calculate_angle(hip, knee, ankle)
            print(shoulder_hip_knee_angle)
            print(hip_knee_ankle_angle)
            if 85 <= shoulder_hip_knee_angle <= 115:
                analysis_results += "Correct sitting posture.\n"
            elif shoulder_hip_knee_angle < 85:
                analysis_results += "Leaning forward.\n"
            else:
                analysis_results += "Leaning backward.\n"

            if 90 <= hip_knee_ankle_angle <= 110:
                analysis_results += "Hip in line with legs.\n"
            elif hip_knee_ankle_angle < 90:
                analysis_results += "Hip lower than knees.\n"
            else:
                analysis_results += "Hip higher than knees.\n"

            # Back Position
            vertical = np.array([0, 1])
            shoulder_hip_vector = np.array([right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]])
            shoulder_hip_angle = np.arccos(np.dot(shoulder_hip_vector, vertical) / np.linalg.norm(shoulder_hip_vector))
            shoulder_hip_angle = np.degrees(shoulder_hip_angle)

            if shoulder_hip_angle < 20:
                analysis_results += "Back is not straight.\n"
            else:
                analysis_results += "Back is straight.\n"

            # Overall Balance
            if abs(left_shoulder[1] - right_shoulder[1]) > 0.1:
                if left_shoulder[1] > right_shoulder[1]:
                    analysis_results += "Leaning to the right.\n"
                else:
                    analysis_results += "Leaning to the left.\n"
            else:
                analysis_results += "Body is well balanced.\n"

        feet_on_ground_tolerance = 0.05
        if abs(left_ankle[1] - right_ankle[1]) < feet_on_ground_tolerance:
            analysis_results += "Both feet are placed on the ground.\n"
        else:
            analysis_results += "Feet are not evenly placed on the ground or at least one foot is not on the ground.\n"

        if self.check_legs_crossed(left_knee, right_knee, left_ankle, right_ankle):
            analysis_results += "The legs are not crossed.\n"
        else:
            analysis_results += "The legs are crossed.\n"

        return analysis_results
