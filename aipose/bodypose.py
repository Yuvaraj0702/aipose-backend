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

        keypoints = np.array([[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark])
        scores = np.array([landmark.visibility for landmark in results.pose_landmarks.landmark])

        # Check confidence levels
        low_confidence_points = np.sum(scores < 0.2)
        if low_confidence_points / len(scores) > 0.75:
            return "Improper picture. Please provide a clearer image.", keypoints, scores

        landmarks = {landmark: keypoints[self.mp_pose.PoseLandmark[landmark].value]
                     for landmark in self.mp_pose.PoseLandmark.__members__.keys()}

        nose = landmarks['NOSE']
        left_ear = landmarks['LEFT_EAR']
        right_ear = landmarks['RIGHT_EAR']
        left_shoulder = landmarks['LEFT_SHOULDER']
        right_shoulder = landmarks['RIGHT_SHOULDER']
        left_hip = landmarks['LEFT_HIP']
        right_hip = landmarks['RIGHT_HIP']
        left_knee = landmarks['LEFT_KNEE']
        right_knee = landmarks['RIGHT_KNEE']
        left_ankle = landmarks['LEFT_ANKLE']
        right_ankle = landmarks['RIGHT_ANKLE']

        analysis_results = ""

        # Determine facing direction by comparing the horizontal positions of the nose and ears
        if abs(nose[0] - left_ear[0]) < abs(nose[0] - right_ear[0]):
            facing_side = "left"
            shoulder, hip, knee, ankle = right_shoulder, right_hip, right_knee, right_ankle
        elif abs(nose[0] - left_ear[0]) > abs(nose[0] - right_ear[0]):
            facing_side = "right"
            shoulder, hip, knee, ankle = left_shoulder, left_hip, left_knee, left_ankle
        else:
            facing_side = "ambiguous"

        if facing_side != "ambiguous":
            shoulder_hip_knee_angle = self.calculate_angle(shoulder, hip, knee)
            hip_knee_ankle_angle = self.calculate_angle(hip, knee, ankle)

            if 85 <= shoulder_hip_knee_angle <= 115:
                analysis_results += "Neutral.\n"
            elif shoulder_hip_knee_angle < 85:
                analysis_results += "Positive\n"
            else:
                analysis_results += "Negative\n"

            if 90 <= hip_knee_ankle_angle <= 110:
                analysis_results += "Neutral\n"
            elif hip_knee_ankle_angle < 90:
                analysis_results += "Positive\n"
            else:
                analysis_results += "Negative\n"

            # # Back Position
            # vertical = np.array([0, 1])
            # shoulder_hip_vector = right_shoulder - left_shoulder
            # shoulder_hip_angle = np.degrees(np.arccos(np.dot(shoulder_hip_vector, vertical) / np.linalg.norm(shoulder_hip_vector)))

            # if shoulder_hip_angle < 20:
            #     analysis_results += "Negative\n"
            # else:
            #     analysis_results += "Positive.\n"

            # # Overall Balance
            # shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            # if shoulder_diff > 0.1:
            #     analysis_results += "Positive\n" if left_shoulder[1] > right_shoulder[1] else "Negative\n"
            # else:
            #     analysis_results += "Neutral\n"

        # if abs(left_ankle[1] - right_ankle[1]) < 0.05:
        #     analysis_results += "Positive\n"
        # else:
        #     analysis_results += "Negative\n"

        # if self.check_legs_crossed(left_knee, right_knee, left_ankle, right_ankle):
        #     analysis_results += "Positive\n"
        # else:
        #     analysis_results += "Negative\n"

        return analysis_results

