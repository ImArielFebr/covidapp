# Generated by Django 4.0.4 on 2022-10-20 08:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covidapp', '0019_alter_data_test_tweet_vec_alter_vector_tweet_vec'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data_test',
            name='tweet_vec',
            field=models.CharField(default='0', max_length=1000000),
        ),
        migrations.AlterField(
            model_name='data_train',
            name='tweet_vec',
            field=models.CharField(default='0', max_length=1000000),
        ),
        migrations.AlterField(
            model_name='vector',
            name='tweet_vec',
            field=models.CharField(default='0', max_length=1000000),
        ),
    ]