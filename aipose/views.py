from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image as PILImage, ImageDraw
import os
import tensorflow as tf
import numpy as np

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

            # Save the preprocessed image temporarily for analysis
            preprocessed_image_path = 'preprocessed_' + image_file.name
            preprocessed_image_full_path = os.path.join(settings.MEDIA_ROOT, 'tmp', preprocessed_image_path)
            preprocessed_image_np = tf.squeeze(preprocessed_image).numpy()
            preprocessed_image_pil = PILImage.fromarray(preprocessed_image_np.astype(np.uint8))
            preprocessed_image_pil.save(preprocessed_image_full_path)

            # Analyze the pose using PoseAnalyzer
            pose_analyzer = PoseAnalyzer()
            pose_results, keypoints_with_scores, scores = pose_analyzer.analyze_pose(temp_image_full_path)


            # Adjust keypoints to the original image dimensions
            def adjust_keypoints(keypoints, original_width, original_height, target_width=192, target_height=192):
                width_ratio = original_width / target_width
                height_ratio = original_height / target_height
                adjusted_keypoints = []
                for keypoint in keypoints:
                    x, y = keypoint[1] * target_width, keypoint[0] * target_height
                    adjusted_keypoints.append((y * height_ratio, x * width_ratio))
                return adjusted_keypoints

            adjusted_keypoints = adjust_keypoints(keypoints_with_scores, original_width, original_height)

            # Example skeleton structure for connecting keypoints
            skeleton = [
                (3, 5), (5, 7), (7, 9), (2, 4),
                (4, 6), (6, 8), (5, 6), (5, 11),
                (6, 12), (11, 12), (11, 13), (13, 15),
                (12, 14), (14, 16), (1, 3), (2, 4), (0, 1),
                (0, 2), (0, 3), (0, 4), (8, 10)
            ]

            # Draw keypoints and lines on the original image
            with PILImage.open(temp_image_full_path) as img:
                draw = ImageDraw.Draw(img)
                for i, keypoint in enumerate(adjusted_keypoints):
                    x, y = keypoint[1], keypoint[0]
                    color = 'green' if scores[i] > 0.3 else 'red'
                    draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline=color)
                
                # Draw lines based on the skeleton structure
                for start, end in skeleton:
                    if start < len(adjusted_keypoints) and end < len(adjusted_keypoints):
                        start_x, start_y = adjusted_keypoints[start][1], adjusted_keypoints[start][0]
                        end_x, end_y = adjusted_keypoints[end][1], adjusted_keypoints[end][0]
                        line_color = 'green' if scores[start] > 0.3 and scores[end] > 0.3 else 'red'
                        draw.line((start_x, start_y, end_x, end_y), fill=line_color, width=3)
                
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
                'pose_analysis': pose_results
            }

            # Save image instance with annotated image
            serializer = ImageSerializer(data={'title': request.data.get('title', ''),
                                               'image_file': annotated_image_file})
            if serializer.is_valid():
                serializer.save()
                image_instance = serializer.instance

                # Include analysis results in the response
                return Response(analysis_results, status=status.HTTP_201_CREATED)
            else:
                return Response(analysis_results, status=status.HTTP_201_CREATED)
        except Exception as e:
            print("Error during file processing:", str(e))
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

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

            # Save the preprocessed image temporarily for analysis
            preprocessed_image_path = 'preprocessed_' + image_file.name
            preprocessed_image_full_path = os.path.join(settings.MEDIA_ROOT, 'tmp', preprocessed_image_path)
            preprocessed_image_np = tf.squeeze(preprocessed_image).numpy()
            preprocessed_image_pil = PILImage.fromarray(preprocessed_image_np.astype(np.uint8))
            preprocessed_image_pil.save(preprocessed_image_full_path)

            # Analyze the pose using PoseAnalyzer
            pose_analyzer = DeskPoseAnalyzer()
            pose_results, keypoints_with_scores, scores = pose_analyzer.analyze_pose(temp_image_full_path)

           

            # Adjust keypoints to the original image dimensions
            def adjust_keypoints(keypoints, original_width, original_height, target_width=192, target_height=192):
                width_ratio = original_width / target_width
                height_ratio = original_height / target_height
                adjusted_keypoints = []
                for keypoint in keypoints:
                    x, y = keypoint[1] * target_width, keypoint[0] * target_height
                    adjusted_keypoints.append((y * height_ratio, x * width_ratio))
                return adjusted_keypoints

            adjusted_keypoints = adjust_keypoints(keypoints_with_scores, original_width, original_height)

            # Example skeleton structure for connecting keypoints
            skeleton = [
                (3, 5), (5, 7), (7, 9), (2, 4),
                (4, 6), (6, 8), (5, 6), (5, 11),
                (6, 12), (11, 12), (11, 13), (13, 15),
                (12, 14), (14, 16), (1, 3), (2, 4), (0, 1),
                (0, 2), (0, 3), (0, 4), (8, 10)
            ]

            # Draw keypoints and lines on the original image
            with PILImage.open(temp_image_full_path) as img:
                draw = ImageDraw.Draw(img)
                for i, keypoint in enumerate(adjusted_keypoints):
                    x, y = keypoint[1], keypoint[0]
                    color = 'green' if scores[i] > 0.3 else 'red'
                    draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline=color)
                
                # Draw lines based on the skeleton structure
                for start, end in skeleton:
                    if start < len(adjusted_keypoints) and end < len(adjusted_keypoints):
                        start_x, start_y = adjusted_keypoints[start][1], adjusted_keypoints[start][0]
                        end_x, end_y = adjusted_keypoints[end][1], adjusted_keypoints[end][0]
                        line_color = 'green' if scores[start] > 0.3 and scores[end] > 0.3 else 'red'
                        draw.line((start_x, start_y, end_x, end_y), fill=line_color, width=3)
                
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
                'pose_analysis': pose_results,
            }

            # Save image instance with annotated image
            serializer = ImageSerializer(data={'title': request.data.get('title', ''),
                                               'image_file': annotated_image_file})
            if serializer.is_valid():
                serializer.save()
                image_instance = serializer.instance

                # Include analysis results in the response
                return Response(analysis_results, status=status.HTTP_201_CREATED)
            else:
                return Response(analysis_results, status=status.HTTP_201_CREATED)
        except Exception as e:
            print("Error during file processing:", str(e))
            return Response({"error": "An error occurred while processing the file."+ str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
