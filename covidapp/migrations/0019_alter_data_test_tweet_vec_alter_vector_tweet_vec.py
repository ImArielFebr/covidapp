# Generated by Django 4.0.4 on 2022-10-20 08:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covidapp', '0018_alter_data_train_tweet_vec'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data_test',
            name='tweet_vec',
            field=models.CharField(default='0', max_length=10000),
        ),
        migrations.AlterField(
            model_name='vector',
            name='tweet_vec',
            field=models.CharField(default='0', max_length=10000),
        ),
    ]
