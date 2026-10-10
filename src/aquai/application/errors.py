class TabularInputError(ValueError):
    pass


class FileTooLargeError(TabularInputError):
    pass


class LLMProviderError(RuntimeError):
    pass
