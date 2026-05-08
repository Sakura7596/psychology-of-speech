class LLMError(Exception):
    """LLM 调用相关错误"""
    pass


class LLMConnectionError(LLMError):
    """连接错误"""
    pass


class LLMResponseError(LLMError):
    """响应解析错误"""
    pass
