from __future__ import annotations

from typing import Dict, Optional, Any

from fastapi import APIRouter
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from gateways.auth import GatewayAccessManager
from gateways.errors import as_gateway_error
from gateways.runtime import GatewayRuntime
from gateways.serialization import to_jsonable
from gateways.tcp_legacy.policy import build_tcp_legacy_policy


class CapabilityInvokeRequest(BaseModel):
    """能力调用请求体。"""

    arguments: Dict[str, Any] = Field(default_factory=dict, description="调用参数。")


def build_http_router(runtime: GatewayRuntime, access_manager: GatewayAccessManager) -> APIRouter:
    """构造 HTTP 调试与管理接口。"""

    router = APIRouter()

    def _build_layout_snapshot() -> Dict[str, Any]:
        return {
            "base_dir": runtime.config.base_dir,
            "log_dir": runtime.config.logging.log_dir,
            "log_file": runtime.config.logging.log_file,
            "artifact_dir": runtime.config.gateway.artifact_dir,
            "state_store_mode": "in_memory",
            "recommended_runtime_dirs": {
                "logs": runtime.config.logging.log_dir,
                "artifacts": runtime.config.gateway.artifact_dir,
                "state_cache": f"{runtime.config.base_dir}/var/state",
                "tmp": f"{runtime.config.base_dir}/var/tmp",
            },
        }

    def _render_ops_page() -> str:
        health = runtime.get_health_summary()
        security_policy = access_manager.export_policy()
        layout = _build_layout_snapshot()
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>nuwax_robot_bridge 运维页</title>
  <style>
    body {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; margin: 24px; background: #f4f6f8; color: #1f2937; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08); }}
    .mono {{ font-family: "JetBrains Mono", "Consolas", monospace; word-break: break-all; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
    li {{ margin: 4px 0; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>nuwax_robot_bridge 运维页</h1>
  <p class="muted">用于快速查看宿主机运行时、目录布局和安全策略的最低可用状态页。</p>
  <div class="grid">
    <section class="card">
      <h2>运行时</h2>
      <ul>
        <li>机器人：<span class="mono">{health["robot_name"]}</span></li>
        <li>型号：<span class="mono">{health["robot_model"]}</span></li>
        <li>已启动：<span class="mono">{health["runtime_started"]}</span></li>
        <li>活动任务数：<span class="mono">{health["active_task_count"]}</span></li>
        <li>最新事件游标：<span class="mono">{health["latest_event_cursor"]}</span></li>
      </ul>
    </section>
    <section class="card">
      <h2>目录布局</h2>
      <ul>
        <li>基础目录：<span class="mono">{layout["base_dir"]}</span></li>
        <li>日志目录：<span class="mono">{layout["log_dir"]}</span></li>
        <li>日志文件：<span class="mono">{layout["log_file"]}</span></li>
        <li>制品目录：<span class="mono">{layout["artifact_dir"]}</span></li>
        <li>状态缓存：<span class="mono">{layout["state_store_mode"]}</span></li>
      </ul>
    </section>
    <section class="card">
      <h2>安全策略</h2>
      <ul>
        <li>允许网段：<span class="mono">{", ".join(security_policy["allowed_source_cidrs"]) or "未限制"}</span></li>
        <li>允许未知来源：<span class="mono">{security_policy["allow_unknown_client_hosts"]}</span></li>
        <li>Agent 令牌数：<span class="mono">{security_policy["agent_token_count"]}</span></li>
        <li>Admin 令牌数：<span class="mono">{security_policy["admin_token_count"]}</span></li>
        <li>高风险白名单：<span class="mono">{", ".join(security_policy["agent_high_risk_allowlist"]) or "空"}</span></li>
      </ul>
    </section>
    <section class="card">
      <h2>调试入口</h2>
      <ul>
        <li><span class="mono">/api/health</span></li>
        <li><span class="mono">/api/tools</span></li>
        <li><span class="mono">/api/security/policy</span></li>
        <li><span class="mono">/api/deployment/layout</span></li>
        <li><span class="mono">/api/tasks</span></li>
        <li><span class="mono">/events/stream</span></li>
      </ul>
    </section>
  </div>
</body>
</html>"""

    @router.get("/api/health")
    async def api_health(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return to_jsonable(runtime.get_health_summary())

    @router.get("/api/capabilities")
    async def api_capabilities(
        request: Request,
        exposed_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        access = access_manager.authenticate_http(request)
        descriptors = runtime.list_capabilities(exposed_only=exposed_only)
        allowed = access_manager.filter_capabilities(access, descriptors)
        return {"capabilities": [descriptor.model_dump(mode="json") for descriptor in allowed]}

    @router.get("/api/tools")
    async def api_tools(
        request: Request,
        include_unsupported: bool = False,
    ) -> Dict[str, Any]:
        access = access_manager.authenticate_http(request)
        exposed_only = None if access.role.value == "admin" else True
        views = runtime.list_tools(
            exposed_only=exposed_only,
            include_unsupported=include_unsupported,
        )
        return {"tools": to_jsonable(views)}

    @router.get("/api/security/policy")
    async def api_security_policy(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return access_manager.export_policy()

    @router.get("/api/deployment/layout")
    async def api_deployment_layout(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return _build_layout_snapshot()

    @router.get("/ops", response_class=HTMLResponse)
    async def ops_page(request: Request) -> HTMLResponse:
        access_manager.authenticate_http(request)
        return HTMLResponse(_render_ops_page())

    @router.post("/api/capabilities/{capability_name}/invoke")
    async def api_invoke_capability(
        capability_name: str,
        body: CapabilityInvokeRequest,
        request: Request,
    ) -> Dict[str, Any]:
        access = access_manager.authenticate_http(request)
        descriptor = runtime.capability_registry.get_descriptor(capability_name)
        access_manager.authorize_capability(access, descriptor)
        try:
            result = await run_in_threadpool(
                lambda: runtime.invoke_capability(
                    capability_name,
                    body.arguments,
                    requested_by=access.principal,
                )
            )
        except Exception as exc:
            raise as_gateway_error(exc) from exc
        return to_jsonable(result)

    @router.get("/api/tasks")
    async def api_tasks(
        request: Request,
        history_limit: int = 20,
    ) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {
            "active_tasks": to_jsonable(runtime.task_manager.list_active_tasks()),
            "history_tasks": to_jsonable(runtime.task_manager.list_history(limit=history_limit)),
        }

    @router.get("/api/tasks/{task_id}")
    async def api_task_status(task_id: str, request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        try:
            return to_jsonable(runtime.get_task_snapshot(task_id))
        except Exception as exc:
            raise as_gateway_error(exc) from exc

    @router.post("/api/tasks/{task_id}/cancel")
    async def api_cancel_task(task_id: str, request: Request) -> Dict[str, Any]:
        access = access_manager.authenticate_http(request)
        try:
            return to_jsonable(
                runtime._handle_cancel_task({"task_id": task_id}, requested_by=access.principal)
            )
        except Exception as exc:
            raise as_gateway_error(exc) from exc

    @router.get("/api/legacy/tcp_policy")
    async def api_tcp_policy(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return to_jsonable(build_tcp_legacy_policy())

    @router.get("/api/state/latest")
    async def api_state_latest(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        snapshot = runtime.robot_state_service.get_latest_snapshot()
        if snapshot is None:
            snapshot = await run_in_threadpool(runtime.robot_state_service.refresh)
        return to_jsonable(snapshot)

    @router.get("/api/state/history")
    async def api_state_history(request: Request, limit: int = 20) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {"history": to_jsonable(runtime.robot_state_service.list_history(limit=limit))}

    @router.get("/api/state/diagnostics")
    async def api_state_diagnostics(request: Request, limit: int = 20) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {"diagnostics": to_jsonable(runtime.robot_state_service.list_diagnostics(limit=limit))}

    @router.get("/api/observations/latest")
    async def api_observations_latest(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        context = runtime.observation_service.get_latest_observation(camera_id)
        return {"observation_context": to_jsonable(context)}

    @router.get("/api/observations/history")
    async def api_observations_history(
        request: Request,
        camera_id: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {
            "history": to_jsonable(runtime.observation_service.list_history(camera_id=camera_id, limit=limit)),
            "latest_contexts": to_jsonable(runtime.observation_service.list_latest_contexts()),
        }

    @router.get("/api/artifacts/summary")
    async def api_artifacts_summary(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return to_jsonable(runtime.artifact_service.get_summary())

    @router.get("/api/perception/latest")
    async def api_perception_latest(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        context = runtime.perception_service.get_latest_perception(camera_id)
        return {"perception_context": to_jsonable(context)}

    @router.get("/api/perception/history")
    async def api_perception_history(
        request: Request,
        camera_id: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {
            "history": to_jsonable(runtime.perception_service.list_history(camera_id=camera_id, limit=limit)),
            "latest_contexts": to_jsonable(runtime.perception_service.list_latest_contexts()),
        }

    @router.get("/api/localization/latest")
    async def api_localization_latest(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        snapshot = runtime.localization_service.get_latest_snapshot()
        if snapshot is None and runtime.localization_service.is_available():
            snapshot = await run_in_threadpool(runtime.localization_service.refresh)
        return {"localization_snapshot": to_jsonable(snapshot)}

    @router.get("/api/localization/history")
    async def api_localization_history(request: Request, limit: int = 20) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {"history": to_jsonable(runtime.localization_service.list_history(limit=limit))}

    @router.get("/api/maps/latest")
    async def api_maps_latest(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        snapshot = runtime.mapping_service.get_latest_snapshot()
        if snapshot is None and runtime.mapping_service.is_available():
            snapshot = await run_in_threadpool(runtime.mapping_service.refresh)
        return {"map_snapshot": to_jsonable(snapshot)}

    @router.get("/api/maps/history")
    async def api_maps_history(request: Request, limit: int = 20) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {"history": to_jsonable(runtime.mapping_service.list_history(limit=limit))}

    @router.get("/api/navigation/latest")
    async def api_navigation_latest(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        navigation_context = runtime.navigation_service.get_latest_navigation_context()
        exploration_context = runtime.navigation_service.get_latest_exploration_context()
        if navigation_context is None and runtime.navigation_service.is_navigation_available():
            navigation_context = await run_in_threadpool(runtime.navigation_service.refresh_navigation)
        if exploration_context is None and runtime.navigation_service.is_exploration_available():
            exploration_context = await run_in_threadpool(runtime.navigation_service.refresh_exploration)
        return {
            "navigation_context": to_jsonable(navigation_context),
            "exploration_context": to_jsonable(exploration_context),
        }

    @router.get("/api/navigation/history")
    async def api_navigation_history(request: Request, limit: int = 20) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {
            "navigation_history": to_jsonable(runtime.navigation_service.list_navigation_history(limit=limit)),
            "exploration_history": to_jsonable(runtime.navigation_service.list_exploration_history(limit=limit)),
        }

    @router.get("/api/memory/summary")
    async def api_memory_summary(request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {"memory_summary": to_jsonable(runtime.memory_service.get_summary())}

    @router.get("/api/memory/locations")
    async def api_memory_locations(request: Request, limit: int = 20) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {
            "locations": to_jsonable(runtime.memory_service.list_tagged_locations(limit=limit)),
            "memory_summary": to_jsonable(runtime.memory_service.get_summary()),
        }

    @router.get("/api/memory/semantic")
    async def api_memory_semantic(request: Request, limit: int = 20) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        return {
            "entries": to_jsonable(runtime.memory_service.list_semantic_memories(limit=limit)),
            "memory_summary": to_jsonable(runtime.memory_service.get_summary()),
        }

    @router.get("/artifacts/{artifact_id}/meta")
    async def api_artifact_meta(artifact_id: str, request: Request) -> Dict[str, Any]:
        access_manager.authenticate_http(request)
        try:
            return runtime.artifact_store.get_ref(artifact_id).model_dump(mode="json")
        except Exception as exc:
            raise as_gateway_error(exc) from exc

    @router.get("/artifacts/{artifact_id}")
    async def api_artifact_raw(artifact_id: str, request: Request) -> FileResponse:
        access_manager.authenticate_http(request)
        try:
            ref = runtime.artifact_store.get_ref(artifact_id)
            path = runtime.artifact_store.get_file_path(artifact_id)
        except Exception as exc:
            raise as_gateway_error(exc) from exc
        return FileResponse(path, media_type=ref.mime_type, filename=path.name)

    return router
