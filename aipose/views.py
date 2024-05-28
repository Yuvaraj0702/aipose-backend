from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image as PILImage
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import logging
from concurrent.futures import ThreadPoolExecutor

from .models import Image
from .serializers import ImageSerializer
from .bodypose import PoseAnalyzer
from .handpose import HandPoseAnalyzer
from .deskpose import DeskPoseAnalyzer

logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_with_pad(image, target_height=192, target_width=192)
        image = tf.image.adjust_contrast(image, 300.0)
        image = tf.cast(image, dtype=tf.int32)
        return tf.expand_dims(image, axis=0)
    except tf.errors.InvalidArgumentError:
        raise ValueError("Invalid image file. Please check the image path and format.")

def process_pose_image(file_path, analyzer_class):
    try:
        image_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        analyzer = analyzer_class()
        results = analyzer.pose.process(image_rgb)
        analysis_results = analyzer.analyze_pose(file_path)

        if not results.pose_landmarks:
            raise ValueError("No landmarks detected. Please provide a clearer image.")

        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image_rgb.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            analyzer.mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        return annotated_image, analysis_results
    except Exception as e:
        logger.error("Error during image processing: %s", str(e))
        raise

def process_hand_image(file_path):
    try:
        analyzer = HandPoseAnalyzer()
        hand_results = analyzer.analyze_hand_pose(file_path)

        image_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        annotated_image = image_rgb.copy()  # This part can be expanded to draw landmarks if needed.

        return annotated_image, hand_results
    except Exception as e:
        logger.error("Error during hand image processing: %s", str(e))
        raise

class BasePoseAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    analyzer_class = None

    def get(self, request, format=None):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        logger.info("Request data: %s", request.data)
        logger.info("Request FILES: %s", request.FILES)

        image_file = request.FILES.get('image_file', None)
        if not image_file:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            temp_image_path = default_storage.save('tmp/' + image_file.name, image_file)
            temp_image_full_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

            with ThreadPoolExecutor() as executor:
                if self.analyzer_class is not None:
                    future = executor.submit(process_pose_image, temp_image_full_path, self.analyzer_class)
                else:
                    future = executor.submit(process_hand_image, temp_image_full_path)
                annotated_image, analysis_results = future.result()

            annotated_image_path = 'annotated_' + image_file.name
            annotated_image_full_path = os.path.join(settings.MEDIA_ROOT, 'images', annotated_image_path)
            cv2.imwrite(annotated_image_full_path, annotated_image)

            with open(annotated_image_full_path, 'rb') as f:
                annotated_image_file_name = default_storage.save('images/' + annotated_image_path, f)

            annotated_image_url = default_storage.url(annotated_image_file_name)
            os.remove(temp_image_full_path)

            result = {
                'analysis': analysis_results,
                'annotated_image_url': annotated_image_url
            }

            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error("Error during file processing: %s", str(e))
            return Response({"error": "An error occurred while processing the file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SeatedPosture(BasePoseAPIView):
    analyzer_class = PoseAnalyzer

class HandPosition(BasePoseAPIView):
    analyzer_class = None

class DeskPosition(BasePoseAPIView):
    analyzer_class = DeskPoseAnalyzer
