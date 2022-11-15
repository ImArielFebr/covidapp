from ast import keyword
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import HttpResponseRedirect
import datetime
import time
import re
import json
from requests import request
import tweepy
import psycopg2
from tweepy import OAuthHandler
from textblob import TextBlob

API_key = "cJFB9Z7rnYyGBW9eVOWInF4Ae"
API_secret = "8Znjhs7YdVvdBwH1Al8ofwBDJUo0uRMhGkaMB4lpk0cfYysYHk"
access_token = "228668907-VnqrZnQs49UkYsY2tqoiRYgP47Za3lL5ZXPvcsOr"
access_token_secret = "HFity9Gdpse9qoe0Oaj5qrO92F8Px6Sfn9DQ5vRPUOhE5"
authentication = tweepy.OAuthHandler(API_key, API_secret)
authentication.set_access_token(access_token, access_token_secret)

api = tweepy.API(authentication)

def scrapgw(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")
        
    cur = conn.cursor()

    hasil = []
    rowo = []
    commandd = "Select tweet_id from covidapp_latih"
    cur.execute(commandd)
    rows = cur.fetchall()

    for k in rows:
        p = k[0]
        rowo.append(p)

    if request.method == 'POST':
        data = request.POST["kata"]
        jumlah = request.POST["jumlah"]
        angka = int(jumlah)
        adding =" pemerintah -filter:media -filter:links -filter:retweets lang:id"
        query = data + adding
        tweets_list = tweepy.Cursor(api.search_tweets,  q=query, include_entities=False,  tweet_mode='extended').items(angka)
        def insig(u):
            for line in hasil:
                tweet__id = str(line["tweet_id"])
                tweet__text = line["tweet_text"]
                created__at = line["created_at"]
                usernames = line["username"]
                label = "0"
                if tweet__id == u:
                    command = "INSERT INTO covidapp_tweet VALUES (%s,%s,%s,%s);"
                    cur.execute(command,(tweet__id, tweet__text, created__at, usernames))
                    command2 = "INSERT INTO covidapp_sentimen VALUES (%s,%s,%s,%s);"
                    cur.execute(command2,(tweet__id, tweet__text, label, created__at))
                    conn.commit()
                else:
                    pass

        for tweet in tweets_list:
            tweet_id = tweet.id
            tweet_text = tweet.full_text
            created_at = tweet.created_at
            username = tweet.user.screen_name
            line = {
                'tweet_id' : tweet_id,
                'tweet_text' : tweet_text,
                'created_at' : str(created_at),
                'username' : username
                }
            hasil.append(line)
        u = len(rowo)
        for j in hasil:
            i = str(j['tweet_id'])
            if u == 0:
                insig(i)
            elif i in rowo:
                pass
            elif i not in rowo:
                insig(i)
            else:
                pass
        data = "scrapping berhasil"
        cur.close()
        conn.close()                        
    else:  
        data = "scrapping gagal"
    return render(request, 'covidapp/home.html', {'data':data})