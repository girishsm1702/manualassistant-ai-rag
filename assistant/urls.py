from django.urls import path

from assistant.views import AskView, HealthView, IngestView, ManualListView

urlpatterns = [
    path("assistant/ask/", AskView.as_view(), name="assistant-ask"),
    path("assistant/ingest/", IngestView.as_view(), name="assistant-ingest"),
    path("assistant/manuals/", ManualListView.as_view(), name="assistant-manuals"),
    path("health/", HealthView.as_view(), name="health"),
]
