# Generated by Django 4.0.4 on 2022-08-25 12:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covidapp', '0011_sentimen_taken'),
    ]

    operations = [
        migrations.CreateModel(
            name='df',
            fields=[
                ('tweet_id', models.CharField(blank=True, max_length=100, primary_key=True, serialize=False)),
                ('df', models.CharField(default='0', max_length=2000)),
                ('sess', models.CharField(default='first', max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='idf',
            fields=[
                ('tweet_id', models.CharField(blank=True, max_length=100, primary_key=True, serialize=False)),
                ('idf', models.CharField(default='0', max_length=2000)),
                ('sess', models.CharField(default='first', max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='tf',
            fields=[
                ('tweet_id', models.CharField(blank=True, max_length=100, primary_key=True, serialize=False)),
                ('tf', models.CharField(default='0', max_length=2000)),
                ('sess', models.CharField(default='0', max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='tfidf',
            fields=[
                ('tweet_id', models.CharField(blank=True, max_length=100, primary_key=True, serialize=False)),
                ('tfidf', models.CharField(default='0', max_length=2000)),
                ('sess', models.CharField(default='first', max_length=200)),
            ],
        ),
    ]