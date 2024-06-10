import os
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import mediapipe as mp
import tensorflow as tf
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem, Image as RLImage
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
import base64

from .models import Images
from .serializers import ImageSerializer
from .bodypose import PoseAnalyzer
from .handpose import HandPoseAnalyzer
from .deskpose import DeskPoseAnalyzer
from aipose.datajson import mappings

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
        image_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        analyzer = analyzer_class()
        results = analyzer.pose.process(image_rgb)

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

        # Add current date to the image
        current_date = datetime.now().strftime("%Y-%m-%d")
        cv2.putText(annotated_image, current_date, (10, annotated_image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated_image
    except Exception as e:
        logger.error("Error during image annotation: %s", str(e))
        raise

def create_report(filename, received_data, mapping_data):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    custom_styles = {
        'Title': ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.HexColor('#333333')
        ),
        'Heading2': ParagraphStyle(
            'Heading2',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=15,
            textColor=colors.HexColor('#333333')
        ),
        'Heading3': ParagraphStyle(
            'Heading3',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.HexColor('#555555'),
            leftIndent=10
        ),
        'Normal': ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=5,
            textColor=colors.HexColor('#333333')
        ),
        'Bullet': ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            bulletIndent=20,
            bulletFontName='Helvetica',
            bulletFontSize=10
        )
    }
    elements = []

    # Main Title
    elements.append(Paragraph("<b>Your Personalized Self-Assessment Report</b>", custom_styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))

    # Process images
    image_mappings = {
        'deskposition': 'deskpositionImage',
        'handposition': 'handpositionImage',
        'seatedposture': 'seatedpostureImage'
    }
    image_paths = {}

    for key, image_key in image_mappings.items():
        image_data = received_data.get(image_key, '')
        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                image_path = os.path.join(os.getcwd(), f"{key}.png")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                image_paths[key] = image_path
            except (IndexError, ValueError) as e:
                logger.error(f"Error decoding image data for {key}: {e}")
            except Exception as e:
                logger.error(f"Error saving or loading image for {key}: {e}")

    risk_groups = {'high': {}, 'medium': {}}
    recommendations = []

    # Collect items into risk groups
    for question in mapping_data.get('questions', []):
        category_key = question['category'].replace('_', '')
        category_name = question['category'].replace('_', ' ').title()
        items = received_data.get(category_key, [])
        for item, map_item in zip(items, question['items']):
            key = item.lower().replace(".", "").strip()
            try:
                scenario = map_item['scenarios'][key]
            except KeyError:
                logger.error(f"KeyError: '{key}' not found in scenarios.")
                continue

            risk = scenario['risk'].lower()
            conditions = scenario.get('conditions', [])
            condition = ', '.join(conditions) if conditions and conditions != [""] else 'None'
            
            if risk in risk_groups:
                if condition not in risk_groups[risk]:
                    risk_groups[risk][condition] = []
                risk_groups[risk][condition].append((category_key, category_name, item, scenario))
            else:
                logger.warning(f"Unexpected risk level: {risk}")

    # Add items to the report based on risk groups
    for risk_level, color in [('high', colors.red), ('medium', colors.yellow)]:
        if risk_groups[risk_level]:
            table_data = [[Paragraph(f"{risk_level.capitalize()} Risk", custom_styles['Heading2'])]]
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), color),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white if risk_level == 'high' else colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, 0), 16),
                ('TOPPADDING', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ])
            risk_table = Table(table_data, colWidths=[doc.width])
            risk_table.setStyle(table_style)
            elements.append(risk_table)
            elements.append(Spacer(1, 0.2 * inch))
            
            for condition, items in risk_groups[risk_level].items():
                condition_text = Paragraph(f"Condition: {condition}", custom_styles['Heading3'])
                item_details = []
                for category_key, category_name, item, scenario in items:
                    details = []
                    if scenario['affected_parts']:
                        details.append(Paragraph(f"<b>Affected parts:</b> {', '.join(scenario['affected_parts'])}", custom_styles['Normal']))
                    if scenario['current']:
                        details.append(Spacer(1, 0.1 * inch))
                        details.append(Paragraph(f"<b>Current:</b> {scenario['current']}", custom_styles['Normal']))
                    if details:
                        item_details.extend(details)
                    else:
                        item_details.append(Paragraph("No specific details", custom_styles['Normal']))
                    
                    # Collect recommendations
                    if scenario['recommendation']:
                        recommendations.append((category_name, item, scenario['recommendation']))

                    # Append image
                    if category_key in image_paths:
                        try:
                            image = RLImage(image_paths[category_key])
                            image.drawWidth = 100
                            image.drawHeight = 80
                            item_details.append(Spacer(1, 0.1 * inch))
                            item_details.append(image)
                        except Exception as e:
                            logger.error(f"Error processing image for {category_name}: {e}")
                
                # Create a single row table with two columns for condition and details
                details_flowables = []
                for detail in item_details:
                    details_flowables.append(detail)
                
                table_data = [[condition_text, details_flowables]]
                
                table_style = TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LINEABOVE', (0, 0), (-1, 0), 1, colors.grey),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ])
                condition_table = Table(table_data, colWidths=[2 * inch, 4.5 * inch])
                condition_table.setStyle(table_style)
                elements.append(condition_table)
                elements.append(Spacer(1, 0.035 * inch))

    # Add all recommendations at the end
    if recommendations:
        elements.append(PageBreak())
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("<b>Recommendations</b>", custom_styles['Heading2']))

        # Create a list of bullet points for recommendations
        bullet_points = []
        for category_name, item, recommendation in recommendations:
            bullet_points.append(ListItem(Paragraph(recommendation, custom_styles['Normal']), bulletText='•'))
        
        recommendation_list = ListFlowable(bullet_points, bulletType='bullet', start='•', bulletFontSize=12, bulletOffsetY=5)
        elements.append(recommendation_list)
        elements.append(Spacer(1, 0.1 * inch))

    doc.build(elements)

    # Clean up the saved images after building the report
    for path in image_paths.values():
        if os.path.exists(path):
            os.remove(path)

class BasePoseAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    analyzer_class = None

    def get(self, request, format=None):
        images = Images.objects.all()
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
                    analysis_results = future.result()

                    # Annotate the image
                    annotated_image = annotate_image(temp_image_full_path, self.analyzer_class)

                    _, annotated_image_encoded = cv2.imencode('.jpg', annotated_image)
                    result = {
                        'analysis': analysis_results,
                        'annotated_image': base64.b64encode(annotated_image_encoded).decode('utf-8')
                    }
                else:
                    future = executor.submit(analyze_hand_image, temp_image_full_path)
                    analysis_results = future.result()
                    # Annotate the image
                    annotated_image = annotate_image(temp_image_full_path, PoseAnalyzer)

                    _, annotated_image_encoded = cv2.imencode('.jpg', annotated_image)
                    result = {
                        'analysis': analysis_results,
                        'annotated_image': base64.b64encode(annotated_image_encoded).decode('utf-8')
                    }
                os.remove(temp_image_full_path)

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

class GenerateImage(APIView):
    parser_classes = [JSONParser]

    def post(self, request, *args, **kwargs):
        # Log the incoming JSON data
        logger.info("Incoming JSON data: %s", request.data)
        report_path = os.path.join(settings.MEDIA_ROOT, 'assessment_report.pdf')
        create_report(report_path, request.data, mappings)
        with open(report_path, 'rb') as report_file:
            response = HttpResponse(report_file.read(), content_type='application/pdf')
            response['Content-Disposition'] = f'inline; filename="assessment_report.pdf"'
        os.remove(report_path)
        return response
