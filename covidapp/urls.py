from ast import keyword
from django.urls import path

from . import preprocessing
from . import views
from . import scrap
from . import tfidf
from . import svm, lt

urlpatterns = [
    path('', views.home_view, name='covidapp'),
    path('sentimen/', views.sentim, name='sentimen-analisis'),
    path('keyword/', views.keyword, name='berhasil'),
    path('tfidf/', views.tfidf, name='tfidf'),
    path('tf/', tfidf.ambil_tf, name='ambil_tf'),
    path('df/', tfidf.ambil_df, name='ambil_df'),
    path('idf/', tfidf.ambil_idf, name='ambil_idf'),
    path('tfidff/', tfidf.ambil_tfidf, name='ambil_tfidf'),
    path('ranking/', tfidf.ranking, name='ranking'),
    path('tweet/', scrap.scrapgw, name='scrap'),
    path('preprocessing/', views.prep, name='prep'),
    path('tokenize/', preprocessing.tokenizing, name='tokenize'),
    path('ambil_token/', views.show_tokenize, name='ambil_token'),
    path('removal/', preprocessing.stopwording, name='removal'),
    path('ambil_removal/', views.show_stopremoved, name='ambil_remove'),
    path('normalize/', preprocessing.normalizing, name='normalize'),
    path('ambil_normal/', views.show_normalized, name='ambil_normal'),
    path('stem/', preprocessing.stemming, name='stem'),
    path('unlabeled/', views.unlabeled, name='unlabeled'),
    path('labeled/', views.labeled, name='labeled'),
    path('lu/', views.lu, name='lu'),
    path('latih/', views.tr, name='latih'),
    path('uji/', views.te, name='uji'),
    path('simpan_label/', views.savelabel, name='savelabel'),
    path('ambil_data/', views.ambil_data, name='ambil_data'),
    path('ambil_semua/', views.ambil_all, name='ambil_all'),
    path('ambil_stem/', views.show_stemmed, name='ambil_stem'),
    path('svm/', views.svm, name='svm'),
    path('hapus/', views.delete, name='hapus'),
    path('del/', lt.delete, name='del'),
    path('dnsv/', svm.dnsv, name='dnsv'),
    path('prefor/', svm.prefor, name='prefor'),
    path('svmel/', svm.delete, name='elim'),
    path('klasif/', svm.svm, name='klasif')
]