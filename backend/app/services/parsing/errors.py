class ParsingError(Exception):
    """A safe, user-readable error; never includes tokens or signed URLs."""

    def __init__(self, message: str, *, code: str = "parse_error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class SubmissionUncertain(ParsingError):
    def __init__(self) -> None:
        super().__init__(
            "MinerU 提交结果未知；请核对远端任务，禁止自动重新提交", code="submission_uncertain"
        )
