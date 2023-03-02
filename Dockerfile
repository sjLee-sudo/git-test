FROM python:3.8-slim-buster AS builder
RUN pip install --upgrade pip
RUN apt-get update &&  apt-get install -y apt-utils vim gcc default-libmysqlclient-dev git psmisc logrotate
COPY euclid-data-platform/edap/django_start_prod.sh /
COPY euclid-data-platform/edap/django_start.sh /
COPY euclid-data-platform/edap/wait-for-db.sh /
COPY euclid-data-platform/edap/wait-for-db.py /
COPY gunicorn_logrotate /etc/logrotate.d/
RUN chmod 644 /etc/logrotate.d/gunicorn_logrotate
RUN chmod +x /wait-for-db.sh

WORKDIR /app
ENV PYTHONUNBUFFERED=0
COPY   euclid-data-platform/edap/requirements/   /app/requirements/
RUN    pip install -r /app/requirements/dev.txt
COPY euclid-data-platform/edap /app
RUN rm -rf /app/config/settings/secret/encrypt_database_info.json
RUN mv /app/config/settings/secret/kistep_encrypt_database_info.json /app/config/settings/secret/encrypt_database_info.json
WORKDIR /soynlp_branch
COPY soynlp_branch /soynlp_branch
RUN python setup.py install
WORKDIR /app

EXPOSE 8000