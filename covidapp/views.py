from urllib import response
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from covidapp.isian import InputForm
import datetime
import time
from django.template import loader
from .models import normalized, sentimen, stemmed, stopworded, tokenize, tweet, latih, uji
import re
from . import preprocessing
from .tfidf import *
import pandas as pd
from requests import request
import tweepy
import psycopg2
from tweepy import OAuthHandler
from textblob import TextBlob

conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
cur = conn.cursor()

def sentim(request):
    return render(request, 'covidapp/sentimen.html')

def prep(request):
    return render(request, 'covidapp/prep.html')

def tfidf(request):
    postf = pos()
    negtf = nega()
    frqc = freq()
    postr = positerm()
    negtr = negaterm()
    context = {
        'posi' : postf,
        'neg' : negtf,
        'freq' : frqc,
        'postr' : postr,
        'negtr' : negtr
    }
    return render(request, 'covidapp/tfidf.html', context)

def svm(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
    cur = conn.cursor()
    label = []
    comy2 = "select tweet_id from covidapp_tweet"
    cur.execute(comy2)
    rowsy = cur.fetchall()
    for i in rowsy:
        label.append(i)
    hhh = len(label)
    context = {
        'leng' : hhh,
    }

    return render(request, 'covidapp/svm.html', context)

def home_view(request):
    golek = InputForm()
    return render(request, 'covidapp/home.html', {'form': golek})  

def keyword(request):
    if request.method == 'POST':
        form = request.POST["kata"]
        dict = {
                'cari': form
            }
        return render(request, 'covidapp/keyword.html', dict)
    else:  
        return redirect('')
        
def show_tokenize(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
    cur = conn.cursor()
    if request.method == 'POST':
        golek = tokenize.objects.all()
        context = {
            'tabel': golek
        }
        return render(request, 'covidapp/prep.html', context)

def show_stopremoved(request):
    if request.method == 'POST':
        golek = stopworded.objects.all()
        context = {
            'tabel': golek
        }
        return render(request, 'covidapp/prep.html', context)

def show_normalized(request):
    if request.method == 'POST':
        golek = normalized.objects.all()
        context = {
            'tabel': golek
        }
        return render(request, 'covidapp/prep.html', context)
            
def show_stemmed(request):
    if request.method == 'POST':
        golek = stemmed.objects.all()
        context = {
            'tabel': golek
        }
        return render(request, 'covidapp/prep.html', context)

def unlabeled(request):
    golek = sentimen.objects.filter(label="0")
    template = loader.get_template('covidapp/labeling.html')
    context = {
        'form': golek,
    }
    return HttpResponse(template.render(context, request))  

def labeled(request):
    golek = sentimen.objects.exclude(label="0")
    template = loader.get_template('covidapp/labeling.html')
    context = {
        'form': golek,
    }
    return HttpResponse(template.render(context, request))  

def lu(request):
    golek = uji.objects.all()
    template = loader.get_template('covidapp/lu.html')
    jml = len(golek)
    ent = 'Data Uji'
    context = {
        'form': golek,
        'leng': jml,
        'enti' : ent,
    }
    return HttpResponse(template.render(context, request))  

def tr(request):
    golek = latih.objects.all()
    template = loader.get_template('covidapp/lu.html')
    jml = len(golek)
    ent = 'Data Latih'
    context = {
        'form': golek,
        'leng': jml,
        'enti' : ent,
    }
    return HttpResponse(template.render(context, request))  

def te(request):
    golek = uji.objects.all()
    template = loader.get_template('covidapp/lu.html')
    jml = len(golek)
    ent = 'Data Uji'
    context = {
        'form': golek,
        'leng': jml,
        'enti' : ent,
    }
    return HttpResponse(template.render(context, request))  

def savelabel(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
    cur = conn.cursor()
    if request.method == 'POST':
        id = request.POST["tweet_id"]
        label = request.POST["sentimen"]
        commande = "UPDATE covidapp_sentimen set label = %s where tweet_id = %s;"
        cur.execute(commande,(label, id))
        conn.commit()
        
        golek = sentimen.objects.filter(label="0")
        template = loader.get_template('covidapp/labeling.html')
        context = {
            'form': golek,
        }
        return HttpResponse(template.render(context, request))
    else:
        pass
    cur.close()
    conn.close()

def delete(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
    cur = conn.cursor()
    if request.method == 'POST':
        id = request.POST["tweet_id"]
        commande = "DELETE FROM covidapp_sentimen WHERE tweet_id = %s;"
        cur.execute(commande,(id,))
        commandj = "DELETE FROM covidapp_tweet WHERE tweet_id = %s;"
        cur.execute(commandj,(id,))
        commandt = "DELETE FROM covidapp_tokenize WHERE tweet_id = %s;"
        cur.execute(commandt,(id,))
        commandn = "DELETE FROM covidapp_normalized WHERE tweet_id = %s;"
        cur.execute(commandn,(id,))
        commandst = "DELETE FROM covidapp_stopwords WHERE tweet_id = %s;"
        cur.execute(commandst,(id,))
        commandse = "DELETE FROM covidapp_stemmed WHERE tweet_id = %s;"
        cur.execute(commandse,(id,))
        conn.commit()
        
        golek = sentimen.objects.filter(label="0")
        template = loader.get_template('covidapp/labeling.html')
        context = {
            'form': golek,
        }
        return HttpResponse(template.render(context, request))
    else:
        pass
    cur.close()
    conn.close()

def ambil_data(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
    cur = conn.cursor()
    kod = []
    commandrr = "SELECT max(sesi_id) FROM covidapp_sesi"
    cur.execute(commandrr)
    rowsr = cur.fetchall()

    for s in rowsr:
        p = s[0]
        kod.append(p)

    kodd = kod[0]
    golek = tweet.objects.filter(taken=kodd)
    template = loader.get_template('covidapp/home.html')
    context = {
        'form': golek,
    }
    cur.close()
    conn.close()
    return HttpResponse(template.render(context, request)) 

def ambil_all(request):
    golek = tweet.objects.all()
    template = loader.get_template('covidapp/home.html')
    context = {
        'form': golek,
    }
    cur.close()
    conn.close()
    return HttpResponse(template.render(context, request)) 

cur.close()
conn.close()