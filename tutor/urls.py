from django.urls import path
from .views import (
    AskQuestionView,
    RateAnswerView,
    GetMetricsView,
    GetTopicsView,
    PersonaListView,
    QuestionHistoryView
)

urlpatterns = [
    path('ask/', AskQuestionView.as_view(), name='ask-question'),
    path('rate-answer/', RateAnswerView.as_view(), name='rate-answer'),
    path('metrics/', GetMetricsView.as_view(), name='get-metrics'),
    path('topics/', GetTopicsView.as_view(), name='get-topics'),
    path('personas/', PersonaListView.as_view(), name='list-personas'),
    path('history/', QuestionHistoryView.as_view(), name='question-history'),
]
