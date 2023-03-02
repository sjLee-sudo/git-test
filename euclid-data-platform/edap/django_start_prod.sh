#! /bin/bash
echo 'static files collecting'
echo yes | python ./manage.py collectstatic

gunicorn \
    config.wsgi:application \
    --config gunicorn_config.py \
    # --timeout 50000 \
    # --workers 5 \
    # --bind 0.0.0.0:8000 \
    # --max-requests 1000 \
    # --max-requests-jitter 50 \
    # --limit-request-line 0 \
    # --limit-request-field_size 0 \