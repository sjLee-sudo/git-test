#! /bin/bash
python ./manage.py migrate --fake-initial
echo 'static files collecting'
echo 'runserver 0:8000'
python ./manage.py runserver 0:8000