import time
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
import os
import cv2
import mediapipe as mp
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

def analyze_pose_image(file_path, analyzer_class):
    try:
        image_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        analyzer = analyzer_class()
        analysis_results = analyzer.analyze_pose(file_path)
        return analysis_results
    except Exception as e:
        logger.error("Error during image processing: %s", str(e))
        raise

def analyze_hand_image(file_path):
    try:
        analyzer = HandPoseAnalyzer()
        hand_results = analyzer.analyze_hand_pose(file_path)
        return hand_results
    except Exception as e:
        logger.error("Error during hand image processing: %s", str(e))
        raise

def annotate_image(file_path, analyzer_class):
    try:
        # Read and process the image
        image_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        analyzer = analyzer_class()
        results = analyzer.pose.process(image_rgb)

        if not results.pose_landmarks:
            raise ValueError("No landmarks detected. Please provide a clearer image.")

        # Draw landmarks on the image
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

        return annotated_image
    except Exception as e:
        logger.error("Error during image annotation: %s", str(e))
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
                    future = executor.submit(analyze_pose_image, temp_image_full_path, self.analyzer_class)
                else:
                    future = executor.submit(analyze_hand_image, temp_image_full_path)
                analysis_results = future.result()
                os.remove(temp_image_full_path)

                result = {
                    'analysis': analysis_results,
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

class Annotation(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        logger.info("Request data: %s", request.data)
        logger.info("Request FILES: %s", request.FILES)

        image_file = request.FILES.get('image_file', None)
        analyzer_type = request.data.get('analyzer_type', None)
        if not image_file or not analyzer_type:
            return Response({"error": "No image file or analyzer type provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            temp_image_path = default_storage.save('tmp/' + image_file.name, image_file)
            temp_image_full_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

            analyzer_class = None
            if analyzer_type == 'seated':
                analyzer_class = PoseAnalyzer
            elif analyzer_type == 'desk':
                analyzer_class = DeskPoseAnalyzer
            elif analyzer_type == 'hand':
                analyzer_class = HandPoseAnalyzer

            if not analyzer_class:
                return Response({"error": "Invalid analyzer type"}, status=status.HTTP_400_BAD_REQUEST)

            annotated_image = annotate_image(temp_image_full_path, analyzer_class)

            os.remove(temp_image_full_path)

            _, annotated_image_encoded = cv2.imencode('.jpg', annotated_image)
            response = HttpResponse(annotated_image_encoded.tobytes(), content_type='image/jpeg')
            response['Content-Disposition'] = 'inline; filename="annotated_image.jpg"'

            return response
        except Exception as e:
            logger.error("Error during file processing: %s", str(e))
            return Response({"error": "An error occurred while processing the file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
