# -*- coding: utf-8 -*-
# Generated by Django 1.11.15 on 2018-11-29 00:03
from __future__ import unicode_literals

from django.db import migrations
import picklefield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('uit_plus_job', '0003_auto_20181128_2302'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uitplusjob',
            name='_modules',
            field=picklefield.fields.PickledObjectField(default=dict, editable=False),
        ),
        migrations.AlterField(
            model_name='uitplusjob',
            name='_optional_directives',
            field=picklefield.fields.PickledObjectField(default=list, editable=False),
        ),
        migrations.AlterField(
            model_name='uitplusjob',
            name='archive_input_files',
            field=picklefield.fields.PickledObjectField(default=list, editable=False),
        ),
        migrations.AlterField(
            model_name='uitplusjob',
            name='archive_output_files',
            field=picklefield.fields.PickledObjectField(default=list, editable=False),
        ),
        migrations.AlterField(
            model_name='uitplusjob',
            name='home_input_files',
            field=picklefield.fields.PickledObjectField(default=list, editable=False),
        ),
        migrations.AlterField(
            model_name='uitplusjob',
            name='home_output_files',
            field=picklefield.fields.PickledObjectField(default=list, editable=False),
        ),
        migrations.AlterField(
            model_name='uitplusjob',
            name='transfer_input_files',
            field=picklefield.fields.PickledObjectField(default=list, editable=False),
        ),
        migrations.AlterField(
            model_name='uitplusjob',
            name='transfer_output_files',
            field=picklefield.fields.PickledObjectField(default=list, editable=False),
        ),
    ]
