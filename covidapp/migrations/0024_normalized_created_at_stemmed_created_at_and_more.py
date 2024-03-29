# Generated by Django 4.0.4 on 2022-10-23 06:12

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('covidapp', '0023_rename_stopwords_stopworded'),
    ]

    operations = [
        migrations.AddField(
            model_name='normalized',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 12, 25, 830257, tzinfo=utc)),
        ),
        migrations.AddField(
            model_name='stemmed',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 12, 25, 830257, tzinfo=utc)),
        ),
        migrations.AddField(
            model_name='stopworded',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 12, 25, 830257, tzinfo=utc)),
        ),
        migrations.AddField(
            model_name='tokenize',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 12, 25, 830257, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='latih',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 12, 25, 830257, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='uji',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 12, 25, 830257, tzinfo=utc)),
        ),
    ]
