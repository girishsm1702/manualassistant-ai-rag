from rest_framework import serializers


class AskRequestSerializer(serializers.Serializer):
    machine_id = serializers.CharField(max_length=64)
    question = serializers.CharField()
    revision = serializers.CharField(required=False, allow_blank=True, allow_null=True)


class IngestRequestSerializer(serializers.Serializer):
    machine_id = serializers.CharField(max_length=64)
    manual_id = serializers.CharField(max_length=64)
    revision = serializers.CharField(max_length=32)
    title = serializers.CharField(max_length=255)
    pdf_path = serializers.CharField(required=False, allow_blank=True)
    file = serializers.FileField(required=False)

    def validate(self, attrs):
        uploaded = attrs.get("file")
        pdf_path = (attrs.get("pdf_path") or "").strip()
        if not uploaded and not pdf_path:
            raise serializers.ValidationError("Provide a PDF file upload or pdf_path.")
        if uploaded:
            name = uploaded.name or ""
            if not name.lower().endswith(".pdf"):
                raise serializers.ValidationError("Only PDF files are accepted.")
            if uploaded.size and uploaded.size > 20 * 1024 * 1024:
                raise serializers.ValidationError("PDF is larger than 20 MB.")
        return attrs
