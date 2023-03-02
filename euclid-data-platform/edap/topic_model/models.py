from django.db import models
if __name__ != '__main__':
    from config.constants import TOPIC_MODEL_DB_NAME

# Create your models here.
class TempTopicMap(models.Model):
    temp_field = models.CharField(max_length=100)

    class Meta:
        # settings/topic_model_router 에서 app_label이 topic_model인 모델을 자동 settings.DATABASE 에 등록한 DB로 맵핑 시켜줌
        app_label = TOPIC_MODEL_DB_NAME
        db_table = 'temp_topic_map'

