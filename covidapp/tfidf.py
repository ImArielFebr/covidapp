import ast
import re
from wordcloud import WordCloud
import psycopg2
import time
import numpy as np
import matplotlib
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import pylab as pl
from io import BytesIO
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.offline as pyoff
import pandas as pd
from matplotlib.colors import ListedColormap
from collections import Counter
import uuid, base64
from io import BytesIO
import pandas as pd
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render, redirect
from collections import Counter
from .models import stemmed
 
conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
cur = conn.cursor()

cnt = Counter()

hasil = []
commandsd = "Select * from covidapp_uji"
cur.execute(commandsd)
rowsa = cur.fetchall()

for i in rowsa:
    tweet_id = i[0]
    tweet_text = i[1]
    daftar = tweet_text.split(" ")
    line = {
        'tweet_id' : tweet_id,
        'tweet_text' : daftar
    }
    hasil.append(line)
    
def calc_TF(document):
    # Counts the number of times the word appears in review
    TF_dict = {}
    for term in document:
        if term in TF_dict:
            TF_dict[term] += 1
        else:
            TF_dict[term] = 1
    # Computes tf for each word
    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(document)
    return TF_dict

list1 = []
daftar = []
for item in hasil:
    ids = item['tweet_id']
    tf = calc_TF(item['tweet_text'])
    
    line = {
        'tweet_id' : ids,
        'term_frequency' : tf
    }
    daftar.append(tf)
    list1.append(line)
df_tf = pd.DataFrame(list1, columns=['tweet_id', 'term_frequency'])
        
def ambil_tf(request):
    if request.method == 'POST':    
        m = df_tf.to_html(classes='table table-dark table-hover table-bordered')
        template = loader.get_template('covidapp/tfidf.html')
        postf = pos()
        negtf = nega()
        frqc = freq()
        postr = positerm()
        negtr = negaterm()
        context = {
            'data': m,
            'posi' : postf,
            'neg' : negtf,
            'freq' : frqc,
            'postr' : postr,
            'negtr' : negtr
        }
        return HttpResponse(template.render(context, request))
    else: pass

    
def calc_DF(tfDict):
    count_DF = {}
    for document in tfDict:
        for term in document:
            if term in count_DF:
                count_DF[term] += 1
            else:
                count_DF[term] = 1
    return count_DF

doc_fr = calc_DF(daftar)
df_df = pd.DataFrame.from_dict(doc_fr, orient='index', dtype=None, columns=['document_frequency'])
def ambil_df(request):
    if request.method == 'POST':    
        m = df_df.to_html(classes='table table-dark table-bordered table-hover')
        template = loader.get_template('covidapp/tfidf.html')
        postf = pos()
        negtf = nega()
        frqc = freq()
        postr = positerm()
        negtr = negaterm()
        context = {
            'data': m,
            'posi' : postf,
            'neg' : negtf,
            'freq' : frqc,
            'postr' : postr,
            'negtr' : negtr
        }
        
        return HttpResponse(template.render(context, request))
    else: pass

n_document = len(daftar)

def calc_IDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
    return IDF_Dict
  
IDF = calc_IDF(n_document, doc_fr)
df_idf = pd.DataFrame.from_dict(IDF, orient='index', dtype=None, columns=['invers document_frequency'])        
def ambil_idf(request):
    if request.method == 'POST':    
        m = df_idf.to_html(classes='table table-dark table-bordered table-hover')
        template = loader.get_template('covidapp/tfidf.html')
        postf = pos()
        negtf = nega()
        frqc = freq()
        postr = positerm()
        negtr = negaterm()
        context = {
            'data': m,
            'posi' : postf,
            'neg' : negtf,
            'freq' : frqc,
            'postr' : postr,
            'negtr' : negtr
        }
        return HttpResponse(template.render(context, request))
    else: pass


def calc_TF_IDF(TF):
    TF_IDF_Dict = {}
    #For each word in the review, we multiply its tf and its idf.
    for key in TF:
        TF_IDF_Dict[key] = TF[key] * IDF[key]
    return TF_IDF_Dict

list2 = []
daftar2 = []
for item in list1:
    ids = item['tweet_id']
    tfidf = calc_TF_IDF(item['term_frequency'])
    line = {
        'tweet_id' : ids,
        'tfidf' : tfidf
    }
    list2.append(line)
    daftar2.append(tfidf)

tefidef = pd.DataFrame(list2, columns=['tweet_id', 'tfidf'])
def ambil_tfidf(request):
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
    cur = conn.cursor()
    if request.method == 'POST':    
        m = tefidef.to_html(classes='table table-dark table-bordered table-hover')
        
        template = loader.get_template('covidapp/tfidf.html')
        postf = pos()
        negtf = nega()
        frqc = freq()
        postr = positerm()
        negtr = negaterm()
        context = {
            'data': m,
            'posi' : postf,
            'neg' : negtf,
            'freq' : frqc,
            'postr' : postr,
            'negtr' : negtr
        }
        return HttpResponse(template.render(context, request))
    else: pass

sorted_DF = sorted(doc_fr.items(), key=lambda kv: kv[1], reverse=True)

unique_term = [item[0] for item in sorted_DF]

def calc_TF_IDF_Vec(__TF_IDF_Dict):
    TF_IDF_vector = [0.0] * len(unique_term)
    # For each unique word, if it is in the review, store its TF-IDF value.
    for i, term in enumerate(unique_term):
        if term in __TF_IDF_Dict:
            TF_IDF_vector[i] = __TF_IDF_Dict[term]
    return TF_IDF_vector

matriks = []
label = []
for item in list2:
    tweet_id = item['tweet_id']
    tweet_vector = calc_TF_IDF_Vec(item['tfidf'])
    comy2 = "select label from covidapp_sentimen where tweet_id = %s"
    cur.execute(comy2,(tweet_id,))
    rowsy = cur.fetchall()
    for i in rowsy:
        label.append(i)
    matriks.append(tweet_vector)

TF_IDF_Vec_List = np.array(matriks)
sums = TF_IDF_Vec_List.sum(axis=0)
vec_list = pd.DataFrame(TF_IDF_Vec_List)

vectorz = sparse.csr_matrix(TF_IDF_Vec_List)
labee = np.array(label)

data = []

for col, term in enumerate(unique_term):
    data.append((term, sums[col]))
datt = pd.DataFrame(data, columns=["term", "score"])
mer = {
    'vector': vectorz,
    'label': labee.ravel()
}
gab = pd.DataFrame(mer)
    
ranking = pd.DataFrame(data, columns=['term', 'rank'])
rank = ranking.sort_values('rank', ascending=False)
def ranking(request):
    if request.method == 'POST':    
        m = rank.to_html(classes='table table-dark table-bordered table-hover')
        template = loader.get_template('covidapp/tfidf.html')
        postf = pos()
        negtf = nega()
        frqc = freq()
        postr = positerm()
        negtr = negaterm()
        context = {
            'data': m,
            'posi' : postf,
            'neg' : negtf,
            'freq' : frqc,
            'postr' : postr,
            'negtr' : negtr
        }
        return HttpResponse(template.render(context, request))
    else: pass

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def freq():
    plt.figure(figsize=(7,4))
    sns.countplot(x=gab['label'], data=gab, hue=gab['label'])
    plt.ylabel('Frekuensi', fontsize=12)
    plt.xlabel('Sentimen', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frekuensi Sentimen", fontsize=15)
    freq = get_graph()
    return freq

stopwords = ['kurva', 'vaksinasi', 'masker', 'psbb', 'lockdown', 'pandemi', 'virus', 'amerika']
def nega():
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
    cur = conn.cursor()
    comsd = "select * from covidapp_sentimen where label = %s"
    cur.execute(comsd,('-1',))
    rowuy = cur.fetchall()
    neg = []
    for r in rowuy:
        tweet_id = r[0]
        comsfd = "select tweet_text from covidapp_uji where tweet_id = %s"
        cur.execute(comsfd,(tweet_id,))
        rowf = cur.fetchall()
        for g in rowf:
            line = {
                'text': g[0]
            }
            neg.append(line)
            
    negs = pd.DataFrame(neg)
    nega = " ".join(word for word in negs.text)
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color='white', stopwords=stopwords)
    wordcloud.generate(nega)
    plt.figure(figsize=(7,4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    nega = get_graph()
    return nega

def negaterm():
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
    cur = conn.cursor()
    comsd = "select * from covidapp_sentimen where label = %s"
    cur.execute(comsd,('-1',))
    rowuy = cur.fetchall()
    
    negteks = ''
    for r in rowuy:
        tweet_id = r[0]
        comsfd = "select tweet_text from covidapp_uji where tweet_id = %s"
        cur.execute(comsfd,(tweet_id,))
        rowf = cur.fetchall()
        for g in rowf:
            negtekss = g[0] + ' '
            negteks += negtekss
            
    for text in negteks.split(" "):
        if text != text:
            cnt[text] = 0
        else:
            cnt[text] += 1
    mc = cnt.most_common(25)
    word_freq = pd.DataFrame(mc, columns = ['kata', 'jumlah'])
    word_freq.head()
    fig, ax = plt.subplots(figsize = (7,4))
    word_freq.sort_values(by = 'jumlah').plot.barh(x = 'kata', y = 'jumlah', ax = ax, color = "red")
    for word_freq in ax.containers:
        ax.bar_label(word_freq)
    ax.set_title("25 Term Yang Paling Sering Muncul Pada Data Sentimen Percaya")
    negateks = get_graph()
    return negateks

def pos():
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
    cur = conn.cursor()
    compo = "select * from covidapp_sentimen where label = %s"
    cur.execute(compo,('1',))
    rowpo = cur.fetchall()
    pos = []
    for r in rowpo:
        tweet_id = r[0]
        comsfd = "select tweet_text from covidapp_uji where tweet_id = %s"
        cur.execute(comsfd,(tweet_id,))
        rowf = cur.fetchall()
        for g in rowf:
            line = {
                'text': g[0]
            }
            pos.append(line)
            
    poss = pd.DataFrame(pos)
    posi = " ".join(word for word in poss.text)
    wordpos = WordCloud(width=1600, height=800, max_font_size=200, background_color='white', stopwords=stopwords)
    wordpos.generate(posi)

    plt.figure(figsize=(7,4))

    plt.imshow(wordpos, interpolation='bilinear')

    plt.axis("off")
    pos = get_graph()
    return pos

def positerm():
    conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
    cur = conn.cursor()
    comsd = "select * from covidapp_sentimen where label = %s"
    cur.execute(comsd,('1',))
    rowuy = cur.fetchall()
    
    posteks = ''
    for r in rowuy:
        tweet_id = r[0]
        comsfd = "select tweet_text from covidapp_uji where tweet_id = %s"
        cur.execute(comsfd,(tweet_id,))
        rowf = cur.fetchall()
        for g in rowf:
            postekss = g[0] + ' '
            posteks += postekss
            
    for text in posteks.split(" "):
        cnt[text] += 1
    mc = cnt.most_common(25)
    word_freq = pd.DataFrame(mc, columns = ['kata', 'jumlah'])
    word_freq.head()
    fig, ax = plt.subplots(figsize = (7,4))
    word_freq.sort_values(by = 'jumlah').plot.barh(x = 'kata', y = 'jumlah', ax = ax, color = "green")
    for word_freq in ax.containers:
        ax.bar_label(word_freq)
    ax.set_title("25 Term Yang Paling Sering Muncul Pada Data Sentimen Percaya")
    positeks = get_graph()
    return positeks

def top_tfidf_feats(row, features, top_n=15):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=15):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=15):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=15):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs, num_class=9):
    fig = plt.figure(figsize=(6, 40), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        #z = int(str(int(i/3)+1) + str((i%3)+1))
        ax = fig.add_subplot(num_class, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=12, fontsize=12)
        ax.set_ylabel("Word", labelpad=12, fontsize=12)
        ax.set_title("Class = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    
cur.close()
conn.close()