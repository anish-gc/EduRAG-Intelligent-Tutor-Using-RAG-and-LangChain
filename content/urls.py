from django.urls import path
from .views import (
    UploadContentView,
    NaturalLanguageQueryView,
    ContentListView,
    # ContentDetailView
)

urlpatterns = [
    path('upload-content/', UploadContentView.as_view(), name='upload-content'),
    path('nl-query/', NaturalLanguageQueryView.as_view(), name='natural-language-query'),
    path('content/', ContentListView.as_view(), name='content-list'),
    # path('content/<int:content_id>/', ContentDetailView.as_view(), name='content-detail'),
]