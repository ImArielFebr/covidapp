# Generated by Django 4.0.4 on 2022-10-20 08:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covidapp', '0017_vector_rename_tfidf_data_test_tweet_vec_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data_train',
            name='tweet_vec',
            field=models.CharField(default='0', max_length=10000),
        ),
    ]
