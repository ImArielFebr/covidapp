import ast
from random import shuffle
import re
from symbol import comp_for
import numpy as np
import psycopg2
from wordcloud import WordCloud
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import matplotlib
import matplotlib.pyplot as plt
import uuid, base64
from .models import *
from io import BytesIO
from matplotlib import pyplot
import seaborn as sn
color = sn.color_palette()
from sklearn import svm as sevem
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render, redirect

conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
cur = conn.cursor()

command = "Select * from covidapp_latih"
cur.execute(command)
rows = cur.fetchall()
latihd = []
no = 0
for i in rows:
    no += 1
    tweet_id = i[0]
    tweet_text = i[1]
    label = i[3]
    line = {
        'no' : no,
        'text' : tweet_text,
        'label' : label
    }
    latihd.append(line)
    
command = "Select * from covidapp_uji"
cur.execute(command)
rows = cur.fetchall()
ujid = []
for i in rows:
    tweet_id = i[0]
    tweet_text = i[1]
    label = i[3]
    line = {
        'tweet_id' : tweet_id,
        'text' : tweet_text,
        'label' : label
    }
    ujid.append(line)

trainData = pd.DataFrame(latihd, columns=["no", "text", "label"])
testData = pd.DataFrame(ujid, columns=["id", "text", "label"])
vectorizer = TfidfVectorizer(use_idf = True,
                             smooth_idf = True)
train_vectors = vectorizer.fit_transform(trainData["text"])
test_vectors = vectorizer.transform(testData["text"])

classifier_linear = sevem.SVC(kernel='linear', C=1)
classifier_poly = sevem.SVC(kernel='poly', C=1)
classifier_rbf = sevem.SVC(kernel='rbf', C=1)
classifier_sigmoid = sevem.SVC(kernel='sigmoid', C=1)


def dnsv(request):
    if request.method == 'POST':
        classifier_linear.fit(train_vectors, trainData["label"])
        supv = classifier_linear.support_vectors_
        dnsv = supv.todok()
        wkeys = np.array(list(dnsv.keys()))
        wval = np.array(list(dnsv.values()))
        sv_doc = wkeys[:, 0]
        svt = wkeys[:, 1]
        svds = np.stack((sv_doc, svt, wval), axis=1)
        td = trainData.values.tolist()
        sy = []
        for d in svds:
            for t in td:
                if d[0] == t[0]:
                    r1 = d[0]
                    r2 = d[1]
                    r3 = d[2]
                    r4 = t[2]
                    fff = {
                        'document index' : r1,
                        'feature index' : r2,
                        'values' : r3,
                        'label' : r4
                    }
                    sy.append(fff)
        
        dndd = pd.DataFrame(sy)
        denvec = dndd.to_html(classes='table table-dark table-bordered table-hover')
        template = loader.get_template('covidapp/svm.html')
        context = {
            'denvec' : denvec,
        }
        return HttpResponse(template.render(context, request))

def prefor(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
    cur = conn.cursor()
    if request.method == 'POST':
        classifier_linear.fit(train_vectors, trainData["label"])
        commandd = "Select * from covidapp_uji"
        cur.execute(commandd)
        golek = cur.fetchall()
        template = loader.get_template('covidapp/svm.html')
        list = []
        for i in golek:
            ids = i[0]
            text = i[1]
            tgl = i[2]
            lbl = i[3]
            review_vector = vectorizer.transform([text])
            preds = classifier_linear.predict(review_vector)
            data = {
                'id' : ids,
                'tweet' : text,
                'tanggal' : tgl,
                'label' : lbl,
                'prediksi svm' : preds,
            }
            list.append(data)
        hn = pd.DataFrame(list)
        tblll = hn.to_html(classes='table table-dark table-bordered table-hover')
        context = {
            'predf' : tblll,
        }
        return HttpResponse(template.render(context, request))

def svm(request):
    if request.method == 'POST':
        
        t0 = time.time()
        classifier_linear.fit(train_vectors, trainData["label"])
        classifier_poly.fit(train_vectors, trainData["label"])
        classifier_rbf.fit(train_vectors, trainData["label"])
        classifier_sigmoid.fit(train_vectors, trainData["label"])
        t1 = time.time()
        prediction_linear = classifier_linear.predict(test_vectors)
        prediction_poly = classifier_poly.predict(test_vectors)
        prediction_rbf = classifier_rbf.predict(test_vectors)
        prediction_sigmoid = classifier_sigmoid.predict(test_vectors)
        t2 = time.time()
        time_linear_train = t1-t0
        time_linear_predict = t2-t1
        
        train = len(trainData["label"])
        test = len(testData["label"])
        a_linear = "Results for SVM(kernel=Linear)"
        b_linear = "Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict)
        report_linear = classification_report(testData["label"], prediction_linear, output_dict=True)
        akurasi_linear = accuracy_score(testData["label"], prediction_linear)*100
        c_linear = akurasi_linear
        percaya_linear_precision = report_linear['1']['precision']*100
        percaya_linear_recall = report_linear['1']['recall']*100
        percaya_linear_f = report_linear['1']['f1-score']*100
        percaya_linear_sup = report_linear['1']['support']
        tidak_linear_precision = report_linear['-1']['precision']*100
        tidak_linear_recall = report_linear['-1']['recall']*100
        tidak_linear_f = report_linear['-1']['f1-score']*100
        tidak_linear_sup = report_linear['-1']['support']
        a_poly = "Results for SVM(kernel=Polynomial)"
        b_poly = "Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict)
        report_poly = classification_report(testData["label"], prediction_poly, output_dict=True)
        akurasi_poly = accuracy_score(testData["label"], prediction_poly)*100
        c_poly = akurasi_poly
        percaya_poly_precision = report_poly['1']['precision']*100
        percaya_poly_recall = report_poly['1']['recall']*100
        percaya_poly_f = report_poly['1']['f1-score']*100
        percaya_poly_sup = report_poly['1']['support']
        tidak_poly_precision = report_poly['-1']['precision']*100
        tidak_poly_recall = report_poly['-1']['recall']*100
        tidak_poly_f = report_poly['-1']['f1-score']*100
        tidak_poly_sup = report_poly['-1']['support']
        a_rbf = "Results for SVM(kernel=Gaussian)"
        b_rbf = "Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict)
        report_rbf = classification_report(testData["label"], prediction_rbf, output_dict=True)
        akurasi_rbf = accuracy_score(testData["label"], prediction_rbf)*100
        c_rbf = akurasi_rbf
        percaya_rbf_precision = report_rbf['1']['precision']*100
        percaya_rbf_recall = report_rbf['1']['recall']*100
        percaya_rbf_f = report_rbf['1']['f1-score']*100
        percaya_rbf_sup = report_rbf['1']['support']
        tidak_rbf_precision = report_rbf['-1']['precision']*100
        tidak_rbf_recall = report_rbf['-1']['recall']*100
        tidak_rbf_f = report_rbf['-1']['f1-score']*100
        tidak_rbf_sup = report_rbf['-1']['support']
        a_sigmoid = "Results for SVM(kernel=Sigmoid)"
        b_sigmoid = "Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict)
        report_sigmoid = classification_report(testData["label"], prediction_sigmoid, output_dict=True)
        akurasi_sigmoid = accuracy_score(testData["label"], prediction_sigmoid)*100
        c_sigmoid = akurasi_sigmoid
        percaya_sigmoid_precision = report_sigmoid['1']['precision']*100
        percaya_sigmoid_recall = report_sigmoid['1']['recall']*100
        percaya_sigmoid_f = report_sigmoid['1']['f1-score']*100
        percaya_sigmoid_sup = report_sigmoid['1']['support']
        tidak_sigmoid_precision = report_sigmoid['-1']['precision']*100
        tidak_sigmoid_recall = report_sigmoid['-1']['recall']*100
        tidak_sigmoid_f = report_sigmoid['-1']['f1-score']*100
        tidak_sigmoid_sup = report_sigmoid['-1']['support']
        template = loader.get_template('covidapp/svm.html')
        gralin = get_gra_lin(testData["label"], prediction_linear)
        grapoly = get_gra_poly(testData["label"], prediction_poly)
        grarbf = get_gra_rbf(testData["label"], prediction_rbf)
        grasig = get_gra_sig(testData["label"], prediction_sigmoid)

        context = {
            'train': train,
            'test': test,
            'a_linear': a_linear,
            'b_linear': b_linear,
            'c_linear': c_linear,
            'percaya_linear_p': percaya_linear_precision,
            'percaya_linear_r': percaya_linear_recall,
            'percaya_linear_f': percaya_linear_f,
            'percaya_linear_sup': percaya_linear_sup,
            'tidak_linear_p': tidak_linear_precision,
            'tidak_linear_r': tidak_linear_recall,
            'tidak_linear_f': tidak_linear_f,
            'tidak_linear_sup': tidak_linear_sup,
            'a_poly': a_poly,
            'b_poly': b_poly,
            'c_poly': c_poly,
            'percaya_poly_p': percaya_poly_precision,
            'percaya_poly_r': percaya_poly_recall,
            'percaya_poly_f': percaya_poly_f,
            'percaya_poly_sup': percaya_poly_sup,
            'tidak_poly_p': tidak_poly_precision,
            'tidak_poly_r': tidak_poly_recall,
            'tidak_poly_f': tidak_poly_f,
            'tidak_poly_sup': tidak_poly_sup,
            'a_rbf': a_rbf,
            'b_rbf': b_rbf,
            'c_rbf': c_rbf,
            'percaya_rbf_p': percaya_rbf_precision,
            'percaya_rbf_r': percaya_rbf_recall,
            'percaya_rbf_f': percaya_rbf_f,
            'percaya_rbf_sup': percaya_rbf_sup,
            'tidak_rbf_p': tidak_rbf_precision,
            'tidak_rbf_r': tidak_rbf_recall,
            'tidak_rbf_f': tidak_rbf_f,
            'tidak_rbf_sup': tidak_rbf_sup,
            'a_sigmoid': a_sigmoid,
            'b_sigmoid': b_sigmoid,
            'c_sigmoid': c_sigmoid,
            'percaya_sigmoid_p': percaya_sigmoid_precision,
            'percaya_sigmoid_r': percaya_sigmoid_recall,
            'percaya_sigmoid_f': percaya_sigmoid_f,
            'percaya_sigmoid_sup': percaya_sigmoid_sup,
            'tidak_sigmoid_p': tidak_sigmoid_precision,
            'tidak_sigmoid_r': tidak_sigmoid_recall,
            'tidak_sigmoid_f': tidak_sigmoid_f,
            'tidak_sigmoid_sup': tidak_sigmoid_sup,
            'gralin' : gralin,
            'grapoly' : grapoly,
            'grarbf' : grarbf,
            'grasig' : grasig
        }

        return HttpResponse(template.render(context, request))

 
def delete(request):
    commandd = "Select tweet_id from covidapp_uji"
    cur.execute(commandd)
    rows2 = cur.fetchall()

    command = "Select tweet_id from covidapp_latih"
    cur.execute(command)
    rows = cur.fetchall()

    commands = "Select * from covidapp_uji"
    cur.execute(commands)
    rows3 = cur.fetchall() 
    def ins(u):
        for i in rows3:
            tweet_id = i[0]
            tweet_text = i[1]
            tgl = i[2]
            label = i[3]
            if tweet_id == u:
                commande = "INSERT INTO covidapp_latih VALUES (%s,%s,%s,%s);"
                cur.execute(commande,(tweet_id, tweet_text, tgl, label))
                conn.commit()
                commands = "delete from covidapp_uji where tweet_id = %s;"
                cur.execute(commands,(tweet_id,))
                conn.commit()
                commandt = "delete from covidapp_sentimen where tweet_id = %s;"
                cur.execute(commandt,(tweet_id,))
                conn.commit()
                commandp = "delete from covidapp_stemmed where tweet_id = %s;"
                cur.execute(commandp,(tweet_id,))
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
    
def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_gra_lin(v, b):
    ciem = confusion_matrix(v, b)
    plt.figure(figsize=(6,4))
    sn.heatmap(ciem, annot=True, fmt="d")
    plt.xlabel('Hasil Prediksi')
    plt.ylabel('Data Aktual')
    plt.title("Linear", fontsize=16)
    gralin = get_graph()
    return gralin

def get_gra_poly(v, b):
    ciem = confusion_matrix(v, b)
    plt.figure(figsize=(6,4))
    sn.heatmap(ciem, annot=True, fmt="d")
    plt.xlabel('Hasil Prediksi')
    plt.ylabel('Data Aktual')
    plt.title("Polinomial", fontsize=16)
    grapoly = get_graph()
    return grapoly

def get_gra_rbf(v, b):
    ciem = confusion_matrix(v, b)
    plt.figure(figsize=(6,4))
    sn.heatmap(ciem, annot=True, fmt="d")
    plt.xlabel('Hasil Prediksi')
    plt.ylabel('Data Aktual')
    plt.title("RBF", fontsize=16)
    grarbf = get_graph()
    return grarbf

def get_gra_sig(v, b):
    ciem = confusion_matrix(v, b)
    plt.figure(figsize=(6,4))
    sn.heatmap(ciem, annot=True, fmt="d")
    plt.xlabel('Hasil Prediksi')
    plt.ylabel('Data Aktual')
    plt.title("Sigmoid", fontsize=16)
    grasig = get_graph()
    return grasig

cur.close()
conn.close()
