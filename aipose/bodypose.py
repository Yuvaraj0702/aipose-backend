import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class PoseAnalyzer:
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

    @staticmethod
    def check_legs_crossed(left_knee, right_knee, left_ankle, right_ankle):
        knees_crossed = abs(left_knee[1] - right_knee[1]) < 0.1
        ankles_crossed = abs(left_ankle[1] - right_ankle[1]) < 0.1
        return knees_crossed or ankles_crossed

    def analyze_pose(self, image_path):
        image = self.preprocess_image(image_path)

        outputs = self.movenet(image)
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0]

        keypoints = keypoints_with_scores[:, :2]
        scores = keypoints_with_scores[:, 2]

        # Check confidence levels
        low_confidence_points = np.sum(scores < 0.2)
        if low_confidence_points / len(scores) > 0.75:
            return "Improper picture. Please provide a clearer image.", keypoints_with_scores, scores

        nose = keypoints[0]
        left_ear = keypoints[3]
        right_ear = keypoints[4]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]

        results = ""

        # Determine facing direction by comparing the horizontal positions of the nose and ears
        if abs(nose[0] - left_ear[0]) < abs(nose[0] - right_ear[0]):
            facing_side = "left"
            shoulder, hip, knee, ankle = right_shoulder, right_hip, right_knee, right_ankle
        elif abs(nose[0] - left_ear[0]) > abs(nose[0] - right_ear[0]):
            facing_side = "right"
            shoulder, hip, knee, ankle = left_shoulder, left_hip, left_knee, left_ankle
        else:
            results += "Facing direction is ambiguous or frontal.\n"
            facing_side = "ambiguous"

        if facing_side != "ambiguous":
            results += f"The {facing_side} side of the person is facing the camera.\n"
            shoulder_hip_knee_angle = self.calculate_angle(shoulder, hip, knee)
            hip_knee_ankle_angle = self.calculate_angle(hip, knee, ankle)

            if 85 <= shoulder_hip_knee_angle <= 115:
                results += "Correct sitting posture.\n"
            elif shoulder_hip_knee_angle < 85:
                results += "Leaning forward.\n"
            else:
                results += "Leaning backward.\n"

            if 90 <= hip_knee_ankle_angle <= 110:
                results += "Hip in line with legs.\n"
            elif hip_knee_ankle_angle < 90:
                results += "Hip lower than knees.\n"
            else:
                results += "Hip higher than knees.\n"

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

        feet_on_ground_tolerance = 0.05
        if abs(left_ankle[0] - right_ankle[0]) < feet_on_ground_tolerance:
            results += "Both feet are placed on the ground.\n"
        else:
            results += "Feet are not evenly placed on the ground or at least one foot is not on the ground.\n"

        if self.check_legs_crossed(left_knee, right_knee, left_ankle, right_ankle):
            results += "The legs are not crossed.\n"
        else:
            results += "The legs are crossed.\n"

        return results, keypoints_with_scores, scores
