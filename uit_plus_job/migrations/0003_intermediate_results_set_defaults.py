# -*- coding: utf-8 -*-
# Generated by Django 1.11.15 on 2019-02-28 16:17
from __future__ import unicode_literals

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('uit_plus_job', '0002_intermediate_results'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uitplusjob',
            name='last_intermediate_transfer',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
