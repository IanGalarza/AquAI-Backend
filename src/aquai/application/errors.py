class SourceInputError(ValueError):
    pass


class TabularInputError(SourceInputError):
    pass


class DocumentInputError(SourceInputError):
    pass


class FileTooLargeError(TabularInputError):
    pass


class LLMProviderError(RuntimeError):
    pass


class SemanticIndexError(RuntimeError):
    pass


class SourceNotFoundError(LookupError):
    pass


class SourceConflictError(RuntimeError):
    pass


class SourcesUnavailableError(RuntimeError):
    pass


class PlannerValidationError(ValueError):
    pass


class AnalysisBlocked(ValueError):
    #
    # Guarda el mensaje y el código que explican por qué se bloqueó el análisis.
    #
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


class GenerationNotFoundError(LookupError):
    pass
