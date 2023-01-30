from django.db import models

# Create your models here.


# Model for image uploading
class UploadImage(models.Model):  

    image = models.ImageField(upload_to='images/')  
  
      

