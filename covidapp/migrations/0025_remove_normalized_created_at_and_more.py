# Generated by Django 4.0.4 on 2022-10-23 06:14

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('covidapp', '0024_normalized_created_at_stemmed_created_at_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='normalized',
            name='created_at',
        ),
        migrations.RemoveField(
            model_name='stopworded',
            name='created_at',
        ),
        migrations.RemoveField(
            model_name='tokenize',
            name='created_at',
        ),
        migrations.AlterField(
            model_name='latih',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 13, 56, 567530, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='stemmed',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 13, 56, 567530, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='uji',
            name='created_at',
            field=models.DateField(default=datetime.datetime(2022, 10, 23, 6, 13, 56, 567530, tzinfo=utc)),
        ),
    ]
