# Generated by Django 4.0.4 on 2022-09-22 07:08

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('covidapp', '0013_normalized_sess_stemmed_sess_stopwords_sess_and_more'),
    ]

    operations = [
        migrations.DeleteModel(
            name='df',
        ),
        migrations.DeleteModel(
            name='idf',
        ),
        migrations.DeleteModel(
            name='tf',
        ),
        migrations.RemoveField(
            model_name='normalized',
            name='sess',
        ),
        migrations.RemoveField(
            model_name='sentimen',
            name='taken',
        ),
        migrations.RemoveField(
            model_name='stemmed',
            name='sess',
        ),
        migrations.RemoveField(
            model_name='stopwords',
            name='sess',
        ),
        migrations.RemoveField(
            model_name='tfidf',
            name='sess',
        ),
        migrations.RemoveField(
            model_name='tokenize',
            name='sess',
        ),
        migrations.RemoveField(
            model_name='tweet',
            name='taken',
        ),
    ]
