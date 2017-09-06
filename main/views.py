import logging

from django.contrib import messages
from django.http.response import HttpResponseForbidden
from django.shortcuts import redirect
from django.shortcuts import render

from main import licenseplatenumberdetection
from main.models import *

logger = logging.getLogger(__name__)


def index(request):
    msgs = ''
    req_msgs = messages.get_messages(request)

    detected = False

    for msg in req_msgs:
        msgs += str(msg)

    lp_number = ''
    if len(msgs) != 0:
        lp_number = licenseplatenumberdetection.get_lp_number('static/upload/image.jpg')

    if lp_number != '':
        detected = True

    cars2 = []
    accurate = False
    found = False

    if detected is True:
        cars = Car.objects.all()

        for car in cars:
            if lp_number in car.nmb or car.nmb in lp_number:
                cars2.append(car)
                accurate = True
                found = True

        if accurate is False:
            # replace similar chars
            print 'asdfasdf'
            lp_number = lp_number.replace('o', '0') \
                .replace('O', '0') \
                .replace('B', '8') \
                .replace('G', 'C') \
                .replace('S', '5')

            for car in cars:
                car.nmb = car.nmb.replace('O', '0') \
                    .replace('o', '0') \
                    .replace('B', '8') \
                    .replace('G', 'C') \
                    .replace('S', '5')

            for car in cars:
                if car.nmb in lp_number or lp_number in car.nmb:
                    cars2.append(car)
                    found = True

            if found is False:
                for car in cars:
                    if lp_number[:-1] in car.nmb or car.nmb[:-1] in lp_number \
                            or lp_number[1:] in car.nmb or car.nmb[1:] in lp_number:
                        cars2.append(car)
                        found = True
    # endif

    car_count = len(cars2)

    for car in cars2:
        shtrafs = Shtraf.objects.filter(car=car)
        car.shtrafs = shtrafs

    return render(request, 'vehicle/index.html', locals())


def cars(request):
    nmb = request.GET.get('nmb', '')

    if len(nmb) == 0:
        cars = Car.objects.all()
    else:
        cars = Car.objects.filter(nmb__contains=nmb)

    for car in cars:
        shtrafs = Shtraf.objects.filter(car=car)
        car.shtrafs = shtrafs

    return render(request, 'vehicle/cars.html', locals())


def upload(request):
    img_path = 'static/upload/image.jpg'

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            with open(img_path, 'wb+') as destination:
                for chunk in request.FILES['image'].chunks():
                    destination.write(chunk)

            messages.add_message(request, messages.INFO, 'uploaded')
            return redirect('/')
            # return HttpResponse('image upload success')
    return HttpResponseForbidden('allowed only via POST')
