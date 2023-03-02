from django.urls import re_path
from . import views


urlpatterns = [
    re_path(r'^kwd_srch/?$',views.get_docs_by_keyword),
    re_path(r'^topic_model_srch/?$',views.get_docs_by_query_string),
    re_path(r'^sbjt_list_srch/?$',views.get_docs_by_subject_list),
    re_path(r'^chart_data/?$',views.analyze_docs_by_full_query),
    re_path(r'^category/?$',views.get_category),
    re_path(r'^doc_ids_srch/?$',views.get_docs_by_doc_ids),
]
