class TabularInputError(ValueError):
    pass


class FileTooLargeError(TabularInputError):
    pass


class LLMProviderError(RuntimeError):
    pass


class SourceNotFoundError(LookupError):
    pass


class SourceConflictError(RuntimeError):
    pass


class SourceDecisionError(TabularInputError):
    pass


class SourcesUnavailableError(RuntimeError):
    pass
