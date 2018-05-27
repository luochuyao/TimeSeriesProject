
import os
import datetime

from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.generic import View

from analyse.tools import SPD_classification, SPD_algorithm
from analyse.tools import utils

from analyse.models import spd_task
from analyse.tools import mysql_op



def index_view(request):
    return render(request, 'index.html')

def products_view(request):
    return render(request, 'products.html')

def dataset_view(request):
    return render(request, 'dataset.html')

def contact_view(request):
    return render(request, 'contact.html')

def register(request):
    from analyse.models import User
    if request.method == "POST":
        user = User(request.POST)
    return render(request,'register.html')
def spd_view(request):

    if request.POST:
        files_selected = request.POST.getlist("checkbox_files")
        models_selected = request.POST.getlist("checkbox_models")
        upload_file = request.FILES.get('upload_file')
        upload_model = request.FILES.get('upload_model')
        if upload_file is not None:
            utils.upload_file(upload_file)
        if upload_model is not None:
            utils.upload_file_to_media(upload_model)



        results = []
        for model in models_selected:
            for file in files_selected:
                temp = [model]
                temp += SPD_algorithm.execute_data(file, model)
                results.append(temp)

        results = tuple(results)

        return render(request, 'spd_result.html', {'content': results})

    return render(request, 'singlepoint.html')
