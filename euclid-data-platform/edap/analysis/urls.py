from django.urls import re_path
from . import views


urlpatterns = [
    re_path(r'^sna/?$',views.get_sna),
    re_path(r'^player_sna/?$',views.get_player_sna),
    re_path(r'^rec_kwd/?$',views.get_term_recommendation),
    re_path(r'^lda/?$',views.get_topic_model_base_data),
    re_path(r'^wordcloud/?$',views.get_wordcloud_chart),
    re_path(r'^bar_chart/?$',views.get_bar_chart),
    re_path(r'^line_chart/?$',views.get_line_chart),
    re_path(r'^bubble_chart/?$',views.get_bubble_chart),
    re_path(r'^chart_source/?$',views.get_chart_source),
    # re_path(r'^tree_sna/?$',views.get_tree_sna),
    # re_path(r'^cache/?$',views.get_analysis_cache),
    # re_path(r'^recommend/(?P<target_db>\w+)/(?P<term_type>\w+)/?$',views.get_term_recommendation)
]
