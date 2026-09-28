import inspect
import logging

logger = logging.getLogger(__name__)

TOOL_REGISTRY = {}


def tool(description: str):
    def decorator(func):
        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            # 简单映射：str->string, int ->integer
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
            properties[name] = {
                "type": type_map.get(param.annotation, "string"),
                "description": f"参数{name}",
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)

        TOOL_REGISTRY[func.__name__] = {
            "function": func,
            "schema": {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            },
        }
        logger.info(f"注册工具: {func.__name__}")
        return func

    return decorator


def get_tools_schema():
    """返回所有工具的 schema 列表，用于传给 API"""
    return [item["schema"] for item in TOOL_REGISTRY.values()]


def execute_tool(name: str, args: dict):
    """执行工具"""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"工具 {name} 不存在")
    return TOOL_REGISTRY[name]["function"](**args)
