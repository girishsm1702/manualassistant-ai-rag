from django.db import models


class ManualRecord(models.Model):
    machine_id = models.CharField(max_length=64, db_index=True)
    manual_id = models.CharField(max_length=64)
    revision = models.CharField(max_length=32)
    title = models.CharField(max_length=255)
    source_path = models.CharField(max_length=1024)
    page_count = models.PositiveIntegerField(default=0)
    chunk_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("machine_id", "manual_id", "revision")

    def __str__(self) -> str:
        return f"{self.machine_id} {self.manual_id} {self.revision}"


class ManualChunk(models.Model):
    """One retrieval chunk. machine_id is indexed so isolation happens in SQL."""

    chunk_id = models.CharField(max_length=255, unique=True)
    machine_id = models.CharField(max_length=64, db_index=True)
    manual_id = models.CharField(max_length=64)
    revision = models.CharField(max_length=32)
    title = models.CharField(max_length=255)
    page_number = models.PositiveIntegerField()
    text = models.TextField()
    source_path = models.CharField(max_length=1024)
    embedding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["machine_id", "manual_id", "revision"]),
        ]

    def __str__(self) -> str:
        return self.chunk_id


class QuestionRecord(models.Model):
    machine_id = models.CharField(max_length=64, db_index=True)
    question = models.TextField()
    status = models.CharField(max_length=64)
    answer = models.TextField()
    latency_ms = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
