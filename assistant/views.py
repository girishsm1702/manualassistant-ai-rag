import logging

from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from assistant.serializers import AskRequestSerializer, IngestRequestSerializer
from assistant.services import ask_question, health_status, ingest_manual, list_manuals

logger = logging.getLogger("assistant")


class AskView(APIView):
    """POST /api/v1/assistant/ask/"""

    def post(self, request):
        serializer = AskRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            result = ask_question(
                machine_id=serializer.validated_data["machine_id"],
                question=serializer.validated_data["question"],
                revision=serializer.validated_data.get("revision"),
            )
        except ValueError as exc:
            logger.warning("Ask rejected: %s", exc)
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(result.to_dict(), status=status.HTTP_200_OK)


class IngestView(APIView):
    """POST /api/v1/assistant/ingest/

    Accepts either a multipart PDF upload (`file`) or a local `pdf_path`.
    """

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        serializer = IngestRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            result = ingest_manual(
                machine_id=data["machine_id"],
                manual_id=data["manual_id"],
                revision=data["revision"],
                title=data["title"],
                pdf_path=data.get("pdf_path") or None,
                file=data.get("file"),
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Ingest rejected: %s", exc)
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        http_status = status.HTTP_201_CREATED if result["status"] == "ingested" else status.HTTP_400_BAD_REQUEST
        return Response(result, status=http_status)


class ManualListView(APIView):
    """GET /api/v1/assistant/manuals/"""

    def get(self, request):
        machine_id = request.query_params.get("machine_id")
        return Response({"manuals": list_manuals(machine_id)})


class HealthView(APIView):
    """GET /api/v1/health/"""

    def get(self, request):
        payload = health_status()
        http_status = status.HTTP_200_OK if payload["status"] == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
        return Response(payload, status=http_status)
