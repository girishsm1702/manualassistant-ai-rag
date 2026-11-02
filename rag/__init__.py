"""Manufacturing Manual Assistant RAG package.

This package is independent of Django so backend developers can call it from
a service layer, and so tests can run without a database.
"""

from rag.services import ManualAssistantService

__all__ = ["ManualAssistantService"]
