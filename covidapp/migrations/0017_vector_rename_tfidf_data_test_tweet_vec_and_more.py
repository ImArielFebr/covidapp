# Generated by Django 4.0.4 on 2022-10-20 08:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covidapp', '0016_data_test_data_train_delete_sesi'),
    ]

    operations = [
        migrations.CreateModel(
            name='vector',
            fields=[
                ('tweet_id', models.CharField(blank=True, max_length=100, primary_key=True, serialize=False)),
                ('tweet_vec', models.CharField(default='0', max_length=2000)),
                ('label', models.CharField(default='0', max_length=2000)),
            ],
        ),
        migrations.RenameField(
            model_name='data_test',
            old_name='tfidf',
            new_name='tweet_vec',
        ),
        migrations.RenameField(
            model_name='data_train',
            old_name='tfidf',
            new_name='tweet_vec',
        ),
    ]
