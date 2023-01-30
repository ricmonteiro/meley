from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
from .diagnostic import extract_features, diagnose
from .forms import ImageForm




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

      context = {
          'diagnostic' : diagnosis,
          'img_obj' : img_obj,
        }


        
      return HttpResponse(template.render(context, request)) # methods must return HttpResponse
    else:
        return HttpResponse('File is not an image. Refresh and upload an image.')

  else:
    form = ImageForm()
    template = loader.get_template('index.html')
    return render(request,'index.html', {'form': form})