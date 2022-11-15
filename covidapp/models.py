from django.db import models
from django.forms import ModelForm
import datetime
from django.utils import timezone

now = timezone.now()

class tweet(models.Model):
    tweet_id = models.CharField(max_length=100, primary_key=True, blank=True, null=False)
    tweet_text = models.CharField(max_length=2000)
    created_at = models.DateField()
    username = models.CharField(max_length=200, default="user")

class tokenize(models.Model):
    tweet_id = models.CharField(max_length=100, primary_key=True, blank=True, null=False)
    tweet_text = models.CharField(max_length=2000, default="0")

class stopworded(models.Model):
    tweet_id = models.CharField(max_length=100, primary_key=True, blank=True, null=False)
    tweet_text = models.CharField(max_length=2000, default="0")

class normalized(models.Model):
    tweet_id = models.CharField(max_length=100, primary_key=True, blank=True, null=False)
    tweet_text = models.CharField(max_length=2000, default="0")

class stemmed(models.Model):
    tweet_id = models.CharField(max_length=100, primary_key=True, blank=True, null=False)
    tweet_text = models.CharField(max_length=2000, default="0")
    created_at = models.DateField(default=now)

class sentimen(models.Model):
    tweet_id = models.CharField(max_length=100, primary_key=True, blank=True, null=False)
    tweet_text = models.CharField(max_length=2000, default="0")
    created_at = models.DateField()
    label = models.CharField(max_length=2000, default="0")

class latih(models.Model):
    tweet_id = models.CharField(max_length=100, primary_key=True, blank=True, null=False)
    tweet_text = models.CharField(max_length=2000, default="0")
    created_at = models.DateField(default=now)
    label = models.CharField(max_length=2000, default="0")

class uji(models.Model):
    tweet_id = models.CharField(max_length=100, primary_key=True, blank=True, null=False)
    tweet_text = models.CharField(max_length=2000, default="0")
    created_at = models.DateField(default=now)
    label = models.CharField(max_length=2000, default="0")