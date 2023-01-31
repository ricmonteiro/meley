from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
from .diagnostic import extract_features, diagnose
from .forms import ImageForm
import numpy as np



def index(request):
  if request.method == 'POST':
    form = ImageForm(request.POST, request.FILES)
    
    if form.is_valid():
      form.save()
      image = str(request.FILES['image'])
      features = extract_features(image)
      diagnosis = diagnose(features)
      
      img_obj = form.instance

      template = loader.get_template('class.html')     

      if diagnosis[0][0]>0.70:
        result = 'queratose'
        probability = round(diagnosis[0][0]*100,1)
      elif diagnosis[0][1]>0.70:
        result = 'melanoma'
        probability = round(diagnosis[0][1]*100,1)
      else:
        result = 'bening mark'
        probability = round(100-(np.abs(diagnosis[0][0]-diagnosis[0][1]))*100,2)
      

      context = {
          'result' : result,
          'probability' : str(probability),
          'img_obj' : img_obj,
        }


        
      return HttpResponse(template.render(context, request)) # methods must return HttpResponse
    else:
        return HttpResponse('File is not an image. Refresh and upload an image.')

  else:
    form = ImageForm()
    template = loader.get_template('index.html')
    return render(request,'index.html', {'form': form})