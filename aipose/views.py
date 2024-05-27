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

from .models import Image
from .serializers import ImageSerializer
from .bodypose import PoseAnalyzer
from .handpose import HandPoseAnalyzer
from .deskpose import DeskPoseAnalyzer

class SeatedPosture(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, format=None):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        print("Request data:", request.data)
        print("Request FILES:", request.FILES)

        # Access the uploaded image file
        image_file = request.FILES.get('image_file', None)
        if not image_file:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Save the uploaded image temporarily
            temp_image_path = default_storage.save('tmp/' + image_file.name, image_file)
            temp_image_full_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

            # Load the original image to get its dimensions
            with PILImage.open(temp_image_full_path) as img:
                original_width, original_height = img.size

            # Preprocess the image
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

            preprocessed_image = preprocess_image(temp_image_full_path)

            # Analyze the pose using MediaPipe Pose
            pose_analyzer = PoseAnalyzer()

            # Convert the image to RGB and save it back to the same path
            image_rgb = cv2.cvtColor(cv2.imread(temp_image_full_path), cv2.COLOR_BGR2RGB)
            cv2.imwrite(temp_image_full_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            results = pose_analyzer.pose.process(image_rgb)
            pose_results = pose_analyzer.analyze_pose(temp_image_full_path)

            if not results.pose_landmarks:
                return Response({"error": "No landmarks detected. Please provide a clearer image."}, status=status.HTTP_400_BAD_REQUEST)

            # Draw landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            annotated_image = image_rgb.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                pose_analyzer.mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Convert the annotated image back to BGR for saving
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            annotated_image_path = 'annotated_' + image_file.name
            annotated_image_full_path = os.path.join(settings.MEDIA_ROOT, 'images', annotated_image_path)
            cv2.imwrite(annotated_image_full_path, annotated_image)

            # Save the annotated image to the model
            with open(annotated_image_full_path, 'rb') as f:
                annotated_image_file_name = default_storage.save('images/' + annotated_image_path, f)
            
            annotated_image_url = default_storage.url(annotated_image_file_name)

            # Clean up temporary files
            os.remove(temp_image_full_path)

            # Combine analysis results
            analysis_results = {
                'pose_analysis': pose_results,
                'annotated_image_url': annotated_image_url
            }

            return Response(analysis_results, status=status.HTTP_201_CREATED)

        except Exception as e:
            print("Error during file processing:", str(e))
            return Response({"error": "An error occurred while processing the file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HandPosition(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, format=None):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        print("Request data:", request.data)
        print("Request FILES:", request.FILES)

        # Access the uploaded image file
        image_file = request.FILES.get('image_file', None)
        if not image_file:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Save the uploaded image temporarily
            temp_image_path = default_storage.save('tmp/' + image_file.name, image_file)
            temp_image_full_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

            # Load the original image to get its dimensions
            with PILImage.open(temp_image_full_path) as img:
                original_width, original_height = img.size

            # Preprocess the image
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

            preprocessed_image = preprocess_image(temp_image_full_path)

            # Save the preprocessed image temporarily for analysis
            preprocessed_image_path = 'preprocessed_' + image_file.name
            preprocessed_image_full_path = os.path.join(settings.MEDIA_ROOT, 'tmp', preprocessed_image_path)
            preprocessed_image_np = tf.squeeze(preprocessed_image).numpy()
            preprocessed_image_pil = PILImage.fromarray(preprocessed_image_np.astype(np.uint8))
            preprocessed_image_pil.save(preprocessed_image_full_path)

            # Analyze the hand pose using HandPoseAnalyzer
            hand_pose_analyzer = HandPoseAnalyzer()
            hand_results = hand_pose_analyzer.analyze_hand_pose(temp_image_full_path)

            # Draw keypoints and lines on the original image
            with PILImage.open(temp_image_full_path) as img:
                
                # Save the annotated image
                annotated_image_path = 'annotated_' + image_file.name
                annotated_image_full_path = os.path.join(settings.MEDIA_ROOT, 'images', annotated_image_path)
                img.save(annotated_image_full_path)

            # Save the annotated image to the model
            with open(annotated_image_full_path, 'rb') as f:
                annotated_image_file = default_storage.save('images/' + annotated_image_path, f)

            # Clean up temporary files
            os.remove(temp_image_full_path)
            os.remove(preprocessed_image_full_path)

            # Combine analysis results
            analysis_results = {
                'hand_pose_analysis': hand_results
            }

            # Save image instance with annotated image
            serializer = ImageSerializer(data={'title': request.data.get('title', ''),
                                               'image_file': annotated_image_file})
            if serializer.is_valid():
                serializer.save()

                # Include analysis results in the response
                return Response(analysis_results, status=status.HTTP_201_CREATED)
            else:
                return Response(analysis_results, status=status.HTTP_201_CREATED)
        except Exception as e:
            print("Error during file processing:", str(e))
            return Response({"error": "An error occurred while processing the file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DeskPosition(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, format=None):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        print("Request data:", request.data)
        print("Request FILES:", request.FILES)

        # Access the uploaded image file
        image_file = request.FILES.get('image_file', None)
        if not image_file:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Save the uploaded image temporarily
            temp_image_path = default_storage.save('tmp/' + image_file.name, image_file)
            temp_image_full_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

            # Load the original image to get its dimensions
            with PILImage.open(temp_image_full_path) as img:
                original_width, original_height = img.size

            # Preprocess the image
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

            preprocessed_image = preprocess_image(temp_image_full_path)

            # Analyze the pose using MediaPipe Pose
            pose_analyzer = DeskPoseAnalyzer()

            # Convert the image to RGB and save it back to the same path
            image_rgb = cv2.cvtColor(cv2.imread(temp_image_full_path), cv2.COLOR_BGR2RGB)
            cv2.imwrite(temp_image_full_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            results = pose_analyzer.pose.process(image_rgb)
            pose_results = pose_analyzer.analyze_pose(temp_image_full_path)

            if not results.pose_landmarks:
                return Response({"error": "No landmarks detected. Please provide a clearer image."}, status=status.HTTP_400_BAD_REQUEST)

            # Draw landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            annotated_image = image_rgb.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                pose_analyzer.mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Convert the annotated image back to BGR for saving
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            annotated_image_path = 'annotated_' + image_file.name
            annotated_image_full_path = os.path.join(settings.MEDIA_ROOT, 'images', annotated_image_path)
            cv2.imwrite(annotated_image_full_path, annotated_image)

            # Save the annotated image to the model
            with open(annotated_image_full_path, 'rb') as f:
                annotated_image_file_name = default_storage.save('images/' + annotated_image_path, f)
            
            annotated_image_url = default_storage.url(annotated_image_file_name)

            # Clean up temporary files
            os.remove(temp_image_full_path)

            # Combine analysis results
            analysis_results = {
                'pose_analysis': pose_results,
                'annotated_image_url': annotated_image_url
            }

            return Response(analysis_results, status=status.HTTP_201_CREATED)

        except Exception as e:
            print("Error during file processing:", str(e))
            return Response({"error": "An error occurred while processing the file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
