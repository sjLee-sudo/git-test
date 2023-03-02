# from .models import Terminology
# from graphene_django import DjangoObjectType
# import graphene

# class TermType(DjangoObjectType):
#     class Meta:
#         model = Terminology

# class TermQuery(graphene.ObjectType):
#     all_terms = graphene.List(TermType, target_db=graphene.String())
#     terms_by_name = graphene.Field(TermType, target_db=graphene.String(required=True), name=graphene.String(required=True))
    
#     def resolve_all_terms(self, info, target_db=None, **kwargs):
#         if not target_db:
#             return None
#         return Terminology.objects.using(target_db).all()

#     def resolve_terms_by_name(self, info, target_db, name):
#         try:
#             return Terminology.objects.using(target_db).prefetch_related('synonym_source').prefetch_related('representative_source').get(term=name)
#         except Terminology.DoesNotExist:
#             return None

# schema = graphene.Schema(query=TermQuery)