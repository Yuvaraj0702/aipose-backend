import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class DeskPoseAnalyzer:
    def __init__(self):
        try:
            self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.movenet = self.model.signatures['serving_default']  # type: ignore
        except Exception as e:
            raise RuntimeError("Failed to load the MoveNet model from TensorFlow Hub.") from e

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
    def calculate_horizontal_angle(point1, point2):
        vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
        horizontal = np.array([1, 0])
        dot_product = np.dot(vector, horizontal)
        magnitude_vector = np.linalg.norm(vector)
        angle = np.arccos(dot_product / magnitude_vector)
        return np.degrees(angle) - 90

    @staticmethod
    def preprocess_image(image_path):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        except tf.errors.InvalidArgumentError:
            raise ValueError("Invalid image file. Please check the image path and format.")
        
        image = tf.image.resize_with_pad(image, target_height=192, target_width=192)
        image = tf.image.adjust_contrast(image, 300.0)
        image = tf.cast(image, dtype=tf.int32)
        return tf.expand_dims(image, axis=0)

    def analyze_pose(self, image_path):
        image = self.preprocess_image(image_path)

        outputs = self.movenet(image)
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0]

        keypoints = keypoints_with_scores[:, :2]
        scores = keypoints_with_scores[:, 2]

        # List of indices for the keypoints of interest
        keypoints_of_interest_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        # Check confidence levels for the keypoints of interest
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

        results = ""

        # Determine facing direction by comparing the horizontal positions of the nose and ears
        if abs(nose[0] - left_ear[0]) < abs(nose[0] - right_ear[0]):
            facing_side = "left"
            shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
        elif abs(nose[0] - left_ear[0]) > abs(nose[0] - right_ear[0]):
            facing_side = "right"
            shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
        else:
            results += "Facing direction is ambiguous or frontal.\n"
            facing_side = "ambiguous"

        if facing_side != "ambiguous":
            results += f"The {facing_side} side of the person is facing the camera.\n"
            shoulder_elbow_wrist_angle = self.calculate_angle(shoulder, elbow, wrist)
            
            neck = (left_shoulder + right_shoulder) / 2
            neck_angle = self.calculate_horizontal_angle(neck, nose)

            if shoulder_elbow_wrist_angle < 90:
                results += "The desk is too high.\n"
            elif shoulder_elbow_wrist_angle > 120:
                results += "Table too low.\n"
            else:
                results += "Correct table height.\n"

            shoulder_wrist_distance = np.linalg.norm(np.array(shoulder) - np.array(wrist))
            body_tolerance = 0.15  # Adjust this value as needed
            if shoulder_wrist_distance > body_tolerance:
                results += "Table too far.\n"
            elif shoulder_wrist_distance < body_tolerance / 2:
                results += "Table too close.\n"
            else:
                results += "Table at a good distance.\n"

            if neck_angle > 5:
                results += "Looking upwards.\n"
            elif neck_angle < -5:
                results += "Looking downwards.\n"
            else:
                results += "Good neck position.\n"

            # Arm and Wrist Position
            if abs(wrist[1] - elbow[1]) > 0.1:
                if wrist[1] > elbow[1]:
                    results += "Wrist higher than elbow.\n"
                else:
                    results += "Wrist lower than elbow.\n"

            # Back Position
            shoulder_hip_angle = self.calculate_angle(left_shoulder, (left_shoulder + right_shoulder) / 2, right_shoulder)
            if shoulder_hip_angle < 160:
                results += "Back is not straight.\n"
            else:
                results += "Back is straight.\n"

            # Overall Balance
            if abs(left_shoulder[1] - right_shoulder[1]) > 0.1:
                if left_shoulder[1] > right_shoulder[1]:
                    results += "Leaning to the right.\n"
                else:
                    results += "Leaning to the left.\n"
            else:
                results += "Body is well balanced.\n"

        return results, keypoints_with_scores, scores
