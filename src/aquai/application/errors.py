class TabularInputError(ValueError):
    pass


class FileTooLargeError(TabularInputError):
    pass


class LLMProviderError(RuntimeError):
    pass


class SemanticIndexError(RuntimeError):
    """Error público del índice sin URLs, credenciales ni contenido original."""


class SourceNotFoundError(LookupError):
    pass


class SourceConflictError(RuntimeError):
    pass


class SourcesUnavailableError(RuntimeError):
    pass


class PlannerValidationError(ValueError):
    """La propuesta no respeta las referencias o la estructura del catálogo."""


class AnalysisBlocked(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ConversationNotFoundError(LookupError):
    pass


class ConversationConflictError(RuntimeError):
    pass


class ConversationsUnavailableError(RuntimeError):
    pass


class ConversationStorageError(RuntimeError):
    pass
