from django.conf import settings
from django.conf.urls import url
from django.urls import path, include, re_path
from rest_framework.urlpatterns import format_suffix_patterns
from . import views


urlpatterns = [
    re_path(r'^(?P<target_db>\w+)/(?P<term_type>\w+)/?$',views.TermListView.as_view()),
    re_path(r'^(?P<target_db>\w+)/(?P<term_type>\w+)/(?P<pk>[0-9]+)$',views.TermView.as_view()),
    re_path(r'^userdict/(?P<target_db>\w+)/(?P<term_type>\w+)/?$',views.make_userdict),
]

if settings.DEBUG:
    urlpatterns += [
        re_path(r'^userdict/(?P<target_db>\w+)/(?P<term_type>\w+)/?$',views.make_userdict),
    ]