# models.py in the images app
from django.db import models

class Image(models.Model):
    title = models.CharField(max_length=255, blank=True)
    image_file = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
