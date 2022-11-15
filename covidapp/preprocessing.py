from http.client import HTTPResponse
import pandas as pd 
import psycopg2
import numpy as np
import string
import re
import nltk
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from django.shortcuts import render, redirect
from .models import tokenize, stopworded, normalized, stemmed

conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
cur = conn.cursor()
def remove_tweet_special(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    return text.replace("http://", " ").replace("https://", " ")
                
def remove_number(text):
    return  re.sub(r"\d+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

def remove_whitespace_LT(text):
    return text.strip()

def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def word_tokenize_wrapper(text):
    return word_tokenize(text)

def tokenizing(request):
    

    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
    cur = conn.cursor()
    if request.method == 'POST':
        tokenize.objects.all().delete()

        commandd = "Select tweet_id from covidapp_tweet"
        cur.execute(commandd)
        rows2 = cur.fetchall()

        command = "Select tweet_id from covidapp_latih"
        cur.execute(command)
        rows = cur.fetchall()

        commands = "Select * from covidapp_tweet"
        cur.execute(commands)
        rows3 = cur.fetchall() 
        
        def ins(u):
            for i in rows3:
                tweet_id = i[0]
                tweet_text = i[1]
                tweet_text0 = tweet_text.lower()
                tweet_text1 = remove_tweet_special(tweet_text0)
                tweet_text2 = remove_number(tweet_text1)
                tweet_text3 = remove_punctuation(tweet_text2)
                tweet_text4 = remove_whitespace_LT(tweet_text3)
                tweet_text5 = remove_whitespace_multiple(tweet_text4)
                tweet_text6 = remove_singl_char(tweet_text5)
                kl = word_tokenize_wrapper(tweet_text6)
                tweet_token = " ".join(kl)
                if tweet_id == u:
                    commande = "INSERT INTO covidapp_tokenize VALUES (%s,%s);"
                    cur.execute(commande,(tweet_id, tweet_token))
                    conn.commit()
                else:
                    pass
                    
        for i in rows2:
            if i in rows:
                pass
            elif i not in rows:
                m = i[0]
                ins(m)
                
            else :
                pass
        data = "tokenizing berhasil"
    else:
        data = "tokenizing gagal"
    return render(request, 'covidapp/prep.html', {'data':data})

list_stopwords = stopwords.words('indonesian')
stop_factory = StopWordRemoverFactory()

more_stopwords = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'hiv']

txt_stopword = stop_factory.get_stop_words()+more_stopwords+list_stopwords

def stopwords_removal(words):
    return [word for word in words if word not in txt_stopword]

def stopwording(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
    cur = conn.cursor()
    if request.method == 'POST':
        stopworded.objects.all().delete()

        commandd = "Select tweet_id from covidapp_normalized"
        cur.execute(commandd)
        rows2 = cur.fetchall()

        command = "Select tweet_id from covidapp_stopworded"
        cur.execute(command)
        rows = cur.fetchall()

        commands = "Select * from covidapp_normalized"
        cur.execute(commands)
        rows3 = cur.fetchall() 
        
        def ins(u):
            for i in rows3:
                tweet_id = i[0]
                tweet_text = i[1]
                
                daftar = tweet_text.split(" ")
                l = stopwords_removal(daftar)
                tweet_stop = " ".join(l)
                if tweet_id == u:
                    commande = "INSERT INTO covidapp_stopworded VALUES (%s,%s);"
                    cur.execute(commande,(tweet_id, tweet_stop))
                    conn.commit()
                else:
                    pass
                    
        for i in rows2:
            if i in rows:
                pass
            elif i not in rows:
                m = i[0]
                ins(m)
                
            else :
                pass
        data = "stopword removal berhasil"
    else:
        data = "stopword removal gagal"
    return render(request, 'covidapp/prep.html', {'data':data})

normalizad_word = pd.read_excel("normalisasi_february.xlsx")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

def normalizing(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
    cur = conn.cursor()
    if request.method == 'POST':
        normalized.objects.all().delete()

        commandd = "Select tweet_id from covidapp_tokenize"
        cur.execute(commandd)
        rows2 = cur.fetchall()

        command = "Select tweet_id from covidapp_normalized"
        cur.execute(command)
        rows = cur.fetchall()

        commands = "Select * from covidapp_tokenize"
        cur.execute(commands)
        rows3 = cur.fetchall() 
        def ins(u):
            for i in rows3:
                tweet_id = i[0]
                tweet_text = i[1]
                
                daftar = tweet_text.split(" ")
                l = normalized_term(daftar)
                tweet_normal = " ".join(l)
                if tweet_id == u:
                    commande = "INSERT INTO covidapp_normalized VALUES (%s,%s);"
                    cur.execute(commande,(tweet_id, tweet_normal))
                    conn.commit()
                else:
                    pass
                    
        for i in rows2:
            if i in rows:
                pass
            elif i not in rows:
                m = i[0]
                ins(m)
                
            else :
                pass
        data = "normalisasi berhasil"
    else:
        data = "normalisasi gagal"
    return render(request, 'covidapp/prep.html', {'data':data})

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmed_wrapper(term):
    return stemmer.stem(term)

def stemming(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
    cur = conn.cursor()
    if request.method == 'POST':
        stemmed.objects.all().delete()
        
        commandd = "Select tweet_id from covidapp_stopworded"
        cur.execute(commandd)
        rows2 = cur.fetchall()

        command = "Select tweet_id from covidapp_stemmed"
        cur.execute(command)
        rows = cur.fetchall()

        commands = "Select * from covidapp_stopworded"
        cur.execute(commands)
        rows3 = cur.fetchall() 
        def ins(u):
            for i in rows3:
                tweet_id = i[0]
                tweet_text = i[1]
                tweet_stem = stemmed_wrapper(tweet_text)
                if tweet_id == u:
                    commandh = "Select created_at, label from covidapp_sentimen where tweet_id = %s"
                    cur.execute(commandh, (tweet_id,))
                    rowsh = cur.fetchall()
                    for i in rowsh:
                        tgl = i[0]
                        labeell = i[1]
                        commande = "INSERT INTO covidapp_stemmed VALUES (%s,%s,%s);"
                        cur.execute(commande,(tweet_id, tweet_stem, tgl))
                        commande = "INSERT INTO covidapp_uji VALUES (%s,%s,%s,%s);"
                        cur.execute(commande,(tweet_id, tweet_stem, tgl, labeell))
                        conn.commit()
                else:
                    pass
                    
        for i in rows2:
            if i in rows:
                pass
            elif i not in rows:
                m = i[0]
                ins(m)
                
            else :
                pass
        data = "stemming berhasil"
    else:
        data = "stemming gagal"
    return render(request, 'covidapp/prep.html', {'data':data})
cur.close()
conn.close()