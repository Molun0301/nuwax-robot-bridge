from __future__ import annotations

from typing import Dict, Optional, Any
import json

from fastapi import APIRouter
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.responses import Response

from gateways.auth import GatewayAccessManager
from gateways.errors import GatewayError, as_gateway_error
from gateways.runtime import GatewayRuntime
from gateways.serialization import to_json_text, to_jsonable


def _jsonrpc_result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, *, code: int, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def build_mcp_router(runtime: GatewayRuntime, access_manager: GatewayAccessManager) -> APIRouter:
    """构造 MCP JSON-RPC 路由。"""

    router = APIRouter()

    @router.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        try:
            access = access_manager.authenticate_http(request)
        except GatewayError as exc:
            return JSONResponse(_jsonrpc_error(None, code=exc.jsonrpc_code, message=exc.message, data=exc.to_payload()), status_code=exc.http_status)

        raw = await request.body()
        try:
            body = json.loads(raw)
        except Exception:
            return JSONResponse(_jsonrpc_error(None, code=-32700, message="Parse error"), status_code=400)

        if not isinstance(body, dict):
            return JSONResponse(_jsonrpc_error(None, code=-32600, message="Invalid Request"), status_code=400)

        request_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {}) or {}

        try:
            if method == "initialize":
                return JSONResponse(
                    _jsonrpc_result(
                        request_id,
                        {
                            "protocolVersion": "2025-11-25",
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {"name": "nuwax_robot_bridge", "version": "0.9.0"},
                            "metadata": {
                                "events": {
                                    "websocket_url": f"{runtime.config.gateway.public_base_url}/events/ws",
                                    "sse_url": f"{runtime.config.gateway.public_base_url}/events/stream",
                                },
                                "artifacts": {"base_url": f"{runtime.config.gateway.public_base_url}/artifacts"},
                            },
                        },
                    )
                )

            if method == "tools/list":
                views = runtime.list_tools(
                    exposed_only=None if access.role.value == "admin" else True,
                    include_unsupported=False,
                )
                tools = []
                for view in views:
                    descriptor = runtime.capability_registry.get_descriptor(view.capability_name)
                    description = (
                        f"{view.descriptor.description}\n"
                        f"{descriptor.description}\n"
                        f"执行模式: {descriptor.execution_mode.value}；"
                        f"风险级别: {descriptor.risk_level.value}。"
                    )
                    tools.append(
                        {
                            "name": view.descriptor.name,
                            "description": description,
                            "inputSchema": descriptor.input_schema or {"type": "object", "properties": {}, "additionalProperties": False},
                        }
                    )
                return JSONResponse(_jsonrpc_result(request_id, {"tools": tools}))

            if method == "tools/call":
                name = str(params.get("name", "")).strip()
                arguments = params.get("arguments") or {}
                if not name:
                    raise GatewayError(
                        "tools/call 缺少 name。",
                        error_code="invalid_params",
                        http_status=422,
                        jsonrpc_code=-32602,
                    )
                tool_descriptor = runtime.get_tool_descriptor(name)
                descriptor = runtime.capability_registry.get_descriptor(tool_descriptor.capability_name)
                access_manager.authorize_capability(access, descriptor)
                try:
                    result = await run_in_threadpool(
                        lambda: runtime.invoke_tool(
                            name,
                            arguments,
                            requested_by=access.principal,
                        )
                    )
                    structured = to_jsonable(result)
                except Exception as exc:
                    raise as_gateway_error(exc) from exc
                return JSONResponse(
                    _jsonrpc_result(
                        request_id,
                        {
                            "content": [{"type": "text", "text": to_json_text(structured)}],
                            "structuredContent": structured,
                        },
                    )
                )
            return JSONResponse(_jsonrpc_error(request_id, code=-32601, message=f"Unknown method: {method}"))
        except Exception as exc:
            gateway_error = as_gateway_error(exc)
            return JSONResponse(
                _jsonrpc_error(
                    request_id,
                    code=gateway_error.jsonrpc_code,
                    message=gateway_error.message,
                    data=gateway_error.to_payload(),
                )
            )

    return router
