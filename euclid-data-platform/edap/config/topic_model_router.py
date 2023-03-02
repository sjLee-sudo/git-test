class TopicModelRouter:
    """
    A router to control all database operations on models in the
    auth and contenttypes applications.
    """
    route_app_labels = {'topic_model'}

    def db_for_read(self, model, **hints):
        """
        Attempts to read topic_model models go to topic_model.
        """
        if model._meta.app_label in self.route_app_labels:
            return 'topic_model'
        return None

    def db_for_write(self, model, **hints):
        """
        Attempts to write  topic_model models go to  topic_model.
        """
        if model._meta.app_label in self.route_app_labels:
            return 'topic_model'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if a model in the  topic_model apps is
        involved.
        """
        if (
            obj1._meta.app_label in self.route_app_labels or
            obj2._meta.app_label in self.route_app_labels
        ):
           return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Make sure the topic_model apps only appear in the
        ' topic_model' database.
        """
        if app_label in self.route_app_labels:
            return db == 'topic_model'
        # else:
        #     return False
        return None