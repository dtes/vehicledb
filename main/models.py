from __future__ import unicode_literals

from django.db import models
from django import forms


class Car(models.Model):
    name = models.CharField(max_length=100)
    img = models.CharField(max_length=200)
    year = models.IntegerField()
    nmb = models.CharField(max_length=20)
    owner = models.CharField(max_length=300)


class Shtraf(models.Model):
    sum = models.FloatField(default=0)
    pay_before = models.DateTimeField()
    status = models.IntegerField()
    car = models.ForeignKey(Car)


class ImageUploadForm(forms.Form):
    image = forms.ImageField()
