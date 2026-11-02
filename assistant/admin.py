from django.contrib import admin

from assistant.models import ManualChunk, ManualRecord, QuestionRecord


@admin.register(ManualRecord)
class ManualRecordAdmin(admin.ModelAdmin):
    list_display = ("machine_id", "manual_id", "revision", "title", "chunk_count", "created_at")
    search_fields = ("machine_id", "manual_id", "title")


@admin.register(QuestionRecord)
class QuestionRecordAdmin(admin.ModelAdmin):
    list_display = ("machine_id", "status", "latency_ms", "created_at")
    search_fields = ("machine_id", "question")


@admin.register(ManualChunk)
class ManualChunkAdmin(admin.ModelAdmin):
    list_display = ("chunk_id", "machine_id", "manual_id", "revision", "page_number")
    search_fields = ("machine_id", "manual_id", "text")
    list_filter = ("machine_id", "revision")
