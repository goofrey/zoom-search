"""Unified LLM contracts, capability registry, and adapter base layer."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import AsyncIterator
from typing import Protocol

from zoom_search.demo import create_demo_llm_provider
from zoom_search.errors import call_failure
from zoom_search.errors import extract_openai_error_fields
from zoom_search.errors import llm_http_error_reason
from zoom_search.models import ProviderCapability
from zoom_search.models import LLMRequestOptions
from zoom_search.models import RawDiagnostics
from zoom_search.models import RuntimeContext
from zoom_search.models import UnifiedFinishReason
from zoom_search.models import UnifiedLLMRequest
from zoom_search.models import UnifiedLLMResponse
from zoom_search.models import UnifiedMessage
from zoom_search.models import UnifiedToolCall
from zoom_search.models import ZoomSearchError
from zoom_search.providers.capabilities import BUILTIN_LLM_CAPABILITIES
from zoom_search.transport import normalize_transport_error


class LLMProvider(Protocol):
    async def generate(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        ...

    async def stream_generate(self, request: UnifiedLLMRequest) -> AsyncIterator[dict[str, Any]]:
        ...

    async def generate_json(self, *, prompt: str, context: RuntimeContext, json_strategy: str) -> object:
        ...


@dataclass(slots=True)
class NormalizedLLMRequest:
    request: UnifiedLLMRequest
    response_format: dict[str, Any] | None
    omitted_params: list[str]
    clamped_params: dict[str, Any]
    patched_params: dict[str, Any]
    debug: dict[str, Any]


@dataclass(slots=True)
class ProviderPatchResult:
    request: UnifiedLLMRequest
    response_format: dict[str, Any] | None
    omitted_params: list[str]
    patched_params: dict[str, Any]
    warnings: list[str]
    debug: dict[str, Any]


class LLMCapabilityRegistry:
    def __init__(self, capabilities: dict[str, ProviderCapability]) -> None:
        self._capabilities = dict(capabilities)

    def has(self, engine: str) -> bool:
        return engine in self._capabilities

    def get(self, engine: str) -> ProviderCapability:
        return self._capabilities[engine]

    def recommended_query_rewriting_mode(self, engine: str) -> str:
        return self.get(engine).recommended_query_rewriting_mode

    def supports_structured_output(self, engine: str) -> bool:
        return self.get(engine).structured_output_level in {
            "native_json_schema_strict",
            "native_json_schema_subset",
            "native_json_object_only",
        }

    def supports_json_schema(self, engine: str) -> bool:
        return self.get(engine).structured_output_level in {
            "native_json_schema_strict",
            "native_json_schema_subset",
        }

    def supports_json_object(self, engine: str) -> bool:
        return self.get(engine).structured_output_level in {
            "native_json_schema_strict",
            "native_json_schema_subset",
            "native_json_object_only",
        }


BUILTIN_LLM_REGISTRY = LLMCapabilityRegistry(BUILTIN_LLM_CAPABILITIES)
_CUSTOM_NATIVE_CONTROL_FIELDS = {"endpoint_path", "path", "request_mapping", "response_mapping"}
_OPENAI_COMPATIBLE_COMMON_CONTROL_FIELDS = {"supports_streaming"}
_OPENAI_COMPATIBLE_PATCH_INPUT_FIELDS = {
    "gemini": {"logprobs", "top_logprobs"},
    "doubao-global": {"endpoint_id", "model_id"},
    "doubao-china": {"endpoint_id", "model_id"},
    "baichuan": {"top_k", "with_search_enhance"},
    "minimax-global": {"max_completion_tokens"},
    "minimax-china": {
        "max_completion_tokens",
        "prefer_schema_native",
        "structured_output_model",
        "structured_output_route",
    },
}
_OPENAI_COMPATIBLE_CONTROL_FIELDS = {
    "gemini": {"logprobs", "top_logprobs"},
    "doubao-global": {"endpoint_id", "model_id"},
    "doubao-china": {"endpoint_id", "model_id"},
    "minimax-china": {
        "api_route",
        "prefer_schema_native",
        "structured_output_model",
        "structured_output_route",
    },
}


def _read_response_payload(response: Any) -> Any:
    try:
        return response.json()
    except Exception:  # noqa: BLE001
        text = getattr(response, "text", None)
        return {"error": {"message": str(text).strip() or "HTTP response body is not valid JSON"}}


def create_llm_provider(*, context: RuntimeContext, transport_context: Any | None = None) -> LLMProvider:
    if context.llm_provider.provider_kind == "demo":
        return create_demo_llm_provider(seed=context.request.seed)
    if context.llm_provider.engine == "claude":
        return ClaudeAdapter(context=context, transport_context=transport_context)
    if context.llm_provider.engine == "replicate":
        return ReplicateAdapter(context=context, transport_context=transport_context)
    if context.llm_provider.capability.adapter_class in {"openai_compatible", "openai_compatible_patched"}:
        return OpenAICompatibleAdapter(context=context, transport_context=transport_context)
    if (context.llm_provider.protocol or context.llm_provider.capability.protocol) == "openai_compatible":
        return OpenAICompatibleAdapter(context=context, transport_context=transport_context)
    if (context.llm_provider.protocol or context.llm_provider.capability.protocol) == "native":
        return CustomNativeLLMAdapter(context=context, transport_context=transport_context)
    return _UnsupportedLLMProvider(context=context)


class BaseLLMAdapter:
    def __init__(
        self,
        *,
        capability: ProviderCapability,
        provider_engine: str,
        provider_model: str | None = None,
        request_options: LLMRequestOptions | None = None,
    ) -> None:
        self.capability = capability
        self.provider_engine = provider_engine
        self.provider_model = provider_model
        self.request_options = request_options or LLMRequestOptions()

    async def generate(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        normalized = self.normalize_request(request)
        return UnifiedLLMResponse(
            provider=self.provider_engine,
            model=request.model,
            adapter_class=self.capability.adapter_class,
            finish_reason="unknown",
            provider_metadata={"normalized_request": normalized.debug},
        )

    async def stream_generate(self, request: UnifiedLLMRequest) -> AsyncIterator[dict[str, Any]]:
        response = await self.generate(request.copy_with(stream=False))
        if response.text:
            yield {"event": "token", "delta_text": response.text}
        yield {"event": "done", "response": response}

    def normalize_request(self, request: UnifiedLLMRequest) -> NormalizedLLMRequest:
        normalized = request.copy_with(provider=request.provider or self.provider_engine)
        omitted: list[str] = []
        clamped: dict[str, Any] = {}
        patched: dict[str, Any] = {}

        if self._request_option_disabled("temperature") and normalized.temperature is not None:
            normalized = normalized.copy_with(temperature=None)
            omitted.append("temperature")

        if self._request_option_disabled("seed") and normalized.seed is not None:
            normalized = normalized.copy_with(seed=None)
            omitted.append("seed")

        if self._request_option_disabled("stream") and normalized.stream:
            normalized = normalized.copy_with(stream=False)
            omitted.append("stream")

        temperature = normalized.temperature
        if temperature is not None:
            minimum = self.capability.min_temperature
            maximum = self.capability.max_temperature
            if minimum is not None and temperature < minimum:
                temperature = minimum
                clamped["temperature"] = minimum
            if maximum is not None and temperature > maximum:
                temperature = maximum
                clamped["temperature"] = maximum
            normalized = normalized.copy_with(temperature=temperature)

        if self.capability.supports_seed == "unsupported" and normalized.seed is not None:
            normalized = normalized.copy_with(seed=None)
            omitted.append("seed")

        if normalized.stream and self.capability.streaming_level == "unsupported":
            normalized = normalized.copy_with(stream=False)
            omitted.append("stream")

        response_format, route_debug = build_response_format_route(capability=self.capability, request=normalized)
        if self._request_option_disabled("response_format") and response_format is not None:
            response_format = None
            omitted.append("response_format")

        if normalized.tools and self.capability.tool_calling_level == "unsupported":
            normalized = normalized.copy_with(tools=[], tool_choice=None)
            omitted.extend(["tools", "tool_choice"])

        if normalized.tools and self.capability.tool_choice_mode == "auto_only" and not _is_auto_tool_choice(normalized.tool_choice):
            normalized = normalized.copy_with(tool_choice="auto")
            patched["tool_choice"] = "auto"

        patch_result = apply_provider_request_patches(
            capability=self.capability,
            request=normalized,
            response_format=response_format,
            request_options=self.request_options,
        )
        normalized = patch_result.request
        response_format = patch_result.response_format
        omitted.extend(patch_result.omitted_params)
        patched.update(patch_result.patched_params)

        return NormalizedLLMRequest(
            request=normalized,
            response_format=response_format,
            omitted_params=omitted,
            clamped_params=clamped,
            patched_params=patched,
            debug={
                "provider": self.provider_engine,
                "task": normalized.task,
                "structured_output_route": route_debug,
                "provider_patch": patch_result.debug,
                "omitted_params": list(omitted),
                "clamped_params": dict(clamped),
                "patched_params": dict(patched),
                "warnings": list(patch_result.warnings),
            },
        )

    def _request_option_disabled(self, field_name: str) -> bool:
        return getattr(self.request_options, field_name, None) is False


class OpenAICompatibleAdapter(BaseLLMAdapter):
    def __init__(self, *, context: RuntimeContext, transport_context: Any | None) -> None:
        super().__init__(
            capability=context.llm_provider.capability,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
            request_options=context.llm_provider.request_options,
        )
        self._context = context
        self._transport_context = transport_context

    def normalize_request(self, request: UnifiedLLMRequest) -> NormalizedLLMRequest:
        patch_input_fields = _OPENAI_COMPATIBLE_PATCH_INPUT_FIELDS.get(self.provider_engine, ())
        patch_extra = {
            key: _copy_jsonable(value)
            for key, value in self._context.llm_provider.extra.items()
            if key in patch_input_fields
        }
        if patch_extra:
            patch_extra.update(request.extra_params)
            request = request.copy_with(extra_params=patch_extra)
        return super().normalize_request(request)

    async def generate(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        normalized = self.normalize_request(request)
        payload = self.build_provider_payload(normalized.request, normalized.response_format)
        client = getattr(self._transport_context, "llm_client", None)
        try:
            response = await client.post(
                self._chat_completions_path(normalized.request),
                json=payload,
                headers=self._build_headers(),
            )
            body = _read_response_payload(response)
            if getattr(response, "status_code", 200) >= 400:
                raise self.normalize_error(
                    error=RuntimeError(f"OpenAI-compatible HTTP {response.status_code}"),
                    http_status=response.status_code,
                    payload=body,
                )
        except ZoomSearchError:
            raise
        except Exception as error:  # noqa: BLE001
            raise self.normalize_error(error=error) from error

        if self.capability.engine == "spark" and _is_spark_business_error(body):
            raise self.normalize_error(
                error=RuntimeError(str(body.get("message") or body.get("code") or "spark business error")),
                http_status=getattr(response, "status_code", 200),
                payload=body,
            )

        return normalize_llm_response_payload(
            payload=body,
            capability=self.capability,
            provider=self.provider_engine,
            model=normalized.request.model,
        )

    async def stream_generate(self, request: UnifiedLLMRequest) -> AsyncIterator[dict[str, Any]]:
        normalized = self.normalize_request(request.copy_with(stream=True))
        if not normalized.request.stream:
            async for event in super().stream_generate(normalized.request):
                yield event
            return

        payload = self.build_provider_payload(normalized.request, normalized.response_format)
        client = getattr(self._transport_context, "llm_client", None)
        text_parts: list[str] = []
        usage = normalize_usage(None)
        try:
            async with client.stream(
                "POST",
                self._chat_completions_path(normalized.request),
                json=payload,
                headers=self._build_headers(),
            ) as response:
                if getattr(response, "status_code", 200) >= 400:
                    raise self.normalize_error(
                        error=RuntimeError(f"OpenAI-compatible HTTP {response.status_code}"),
                        http_status=response.status_code,
                    )
                async for line in response.aiter_lines():
                    event = self.normalize_stream_line(line)
                    if event is None:
                        continue
                    if event.get("event") == "token":
                        text_parts.append(str(event.get("delta_text") or ""))
                    elif event.get("event") == "usage":
                        usage = dict(event.get("usage") or usage)
                    yield event
        except ZoomSearchError:
            raise
        except Exception as error:  # noqa: BLE001
            raise self.normalize_error(error=error) from error

        yield {
            "event": "done",
            "response": UnifiedLLMResponse(
                provider=self.provider_engine,
                model=normalized.request.model,
                adapter_class=self.capability.adapter_class,
                text="".join(text_parts),
                finish_reason="stop",
                usage=usage,
                provider_metadata={"normalized_request": normalized.debug},
            ),
        }

    def normalize_stream_line(self, line: str) -> dict[str, Any] | None:
        if not line or not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return {"event": "done"} if data == "[DONE]" else None
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
            return {"event": "usage", "usage": normalize_usage(payload)}
        choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
        if not choices:
            return None
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        text = delta.get("content") or choice.get("text")
        if text is None:
            return None
        return {"event": "token", "delta_text": str(text)}

    async def generate_json(self, *, prompt: str, context: RuntimeContext, json_strategy: str) -> object:
        response = await self.generate(
            UnifiedLLMRequest(
                task="query_rewriting",
                provider=context.llm_provider.engine,
                messages=[UnifiedMessage(role="user", content=prompt)],
                model=context.llm_provider.model or "",
                temperature=0,
                expect_json=True,
                json_schema=_query_rewriting_schema() if json_strategy == "json_schema" else None,
                json_object=json_strategy == "json_object",
                seed=context.request.seed,
                trace_context={"request_id": context.request_id, "phase": "Query Rewriting"},
            )
        )
        if response.json_payload is not None or response.text is not None:
            return response
        raise call_failure(
            category="llm_call_failure",
            component="llm",
            message="LLM provider returned no usable payload.",
            user_message="LLM request failed.",
            request_id=context.request_id,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
            reason_code="empty_output",
            retryable=True,
        )

    def build_provider_payload(
        self,
        request: UnifiedLLMRequest,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        control_fields = _OPENAI_COMPATIBLE_COMMON_CONTROL_FIELDS | _OPENAI_COMPATIBLE_CONTROL_FIELDS.get(self.provider_engine, set())
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._serialize_message(message) for message in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            max_tokens_param = self.capability.max_output_tokens_param or "max_tokens"
            payload[max_tokens_param] = request.extra_params.get(max_tokens_param, request.max_tokens)
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stream:
            payload["stream"] = True
        if request.tools:
            payload["tools"] = [self._serialize_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if self._context.llm_provider.extra:
            for key, value in self._context.llm_provider.extra.items():
                if key in control_fields:
                    continue
                payload.setdefault(key, _copy_jsonable(value))
        if request.extra_params:
            for key, value in request.extra_params.items():
                if key in control_fields:
                    continue
                payload[key] = _copy_jsonable(value)
        if self.provider_engine in {"kimi-global", "kimi-china"}:
            payload.pop("temperature", None)
        if self.provider_engine == "gemini":
            payload.pop("logprobs", None)
            payload.pop("top_logprobs", None)
        return payload

    def normalize_error(self, *, error: Exception, http_status: int | None = None, payload: Any = None) -> ZoomSearchError:
        return normalize_llm_provider_error(
            error=error,
            request_id=self._context.request_id,
            provider_engine=self.provider_engine,
            provider_model=self.provider_model,
            http_status=http_status,
            payload=payload,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self._context.llm_provider.headers)
        if self._context.llm_provider.api_key:
            headers.setdefault("Authorization", f"Bearer {self._context.llm_provider.api_key}")
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _chat_completions_path(self, request: UnifiedLLMRequest) -> str:
        if self.provider_engine == "minimax-china" and request.extra_params.get("api_route") == "native_chatcompletion_v2":
            return "/v1/text/chatcompletion_v2"
        return "/chat/completions"

    def _serialize_message(self, message: UnifiedMessage) -> dict[str, Any]:
        serialized = {"role": message.role, "content": message.content}
        if message.name:
            serialized["name"] = message.name
        if message.tool_call_id:
            serialized["tool_call_id"] = message.tool_call_id
        if isinstance(message.metadata.get("tool_calls"), list):
            serialized["tool_calls"] = _copy_jsonable(message.metadata["tool_calls"])
        return serialized

    def _serialize_tool(self, tool: UnifiedToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        }


class _UnsupportedLLMProvider(BaseLLMAdapter):
    def __init__(self, *, context: RuntimeContext) -> None:
        super().__init__(
            capability=context.llm_provider.capability,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
            request_options=context.llm_provider.request_options,
        )
        self._context = context

    async def generate(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        raise call_failure(
            category="llm_call_failure",
            component="llm",
            message="Built-in HTTP LLM adapters are not implemented in this window.",
            user_message="The selected llm provider is not implemented yet.",
            request_id=self._context.request_id,
            provider_engine=self._context.llm_provider.engine,
            provider_model=self._context.llm_provider.model,
            reason_code="provider_not_implemented",
        )

    async def generate_json(self, *, prompt: str, context: RuntimeContext, json_strategy: str) -> object:
        request = UnifiedLLMRequest(
            task="query_rewriting",
            provider=context.llm_provider.engine,
            messages=[UnifiedMessage(role="user", content=prompt)],
            model=context.llm_provider.model or "",
            temperature=0,
            expect_json=True,
            json_schema=None,
            json_object=json_strategy == "json_object",
            seed=context.request.seed,
            trace_context={"request_id": context.request_id},
        )
        response = await self.generate(request)
        if response.json_payload is not None or response.text is not None:
            return response
        raise call_failure(
            category="llm_call_failure",
            component="llm",
            message="LLM provider returned no usable payload.",
            user_message="LLM request failed.",
            request_id=context.request_id,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
            reason_code="empty_output",
            retryable=True,
        )


class CustomNativeLLMAdapter(BaseLLMAdapter):
    def __init__(self, *, context: RuntimeContext, transport_context: Any | None) -> None:
        super().__init__(
            capability=context.llm_provider.capability,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
            request_options=context.llm_provider.request_options,
        )
        self._context = context
        self._transport_context = transport_context

    async def generate(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        normalized = self.normalize_request(request)
        payload = self.build_provider_payload(normalized.request, normalized.response_format)
        client = getattr(self._transport_context, "llm_client", None)
        try:
            response = await client.post(
                self._endpoint_path(),
                json=payload,
                headers=self._build_headers(),
            )
            body = _read_response_payload(response)
            if getattr(response, "status_code", 200) >= 400:
                raise self.normalize_error(
                    error=RuntimeError(f"Custom native LLM HTTP {response.status_code}"),
                    http_status=response.status_code,
                    payload=body,
                )
        except ZoomSearchError:
            raise
        except Exception as error:  # noqa: BLE001
            raise self.normalize_error(error=error) from error

        return normalize_custom_native_llm_response_payload(
            payload=body,
            mapping=self._response_mapping(),
            capability=self.capability,
            provider=self.provider_engine,
            model=normalized.request.model,
            expect_json=normalized.request.expect_json,
            normalized_debug=normalized.debug,
        )

    async def generate_json(self, *, prompt: str, context: RuntimeContext, json_strategy: str) -> object:
        response = await self.generate(
            UnifiedLLMRequest(
                task="query_rewriting",
                provider=context.llm_provider.engine,
                messages=[UnifiedMessage(role="user", content=prompt)],
                model=context.llm_provider.model or "",
                temperature=0,
                expect_json=True,
                json_object=json_strategy == "json_object",
                seed=context.request.seed,
                trace_context={"request_id": context.request_id, "phase": "Query Rewriting"},
            )
        )
        if response.json_payload is not None or response.text is not None:
            return response
        raise call_failure(
            category="llm_call_failure",
            component="llm",
            message="LLM provider returned no usable payload.",
            user_message="LLM request failed.",
            request_id=context.request_id,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
            reason_code="empty_output",
            retryable=True,
        )

    def build_provider_payload(self, request: UnifiedLLMRequest, response_format: dict[str, Any] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in self._context.llm_provider.extra.items():
            if key in _CUSTOM_NATIVE_CONTROL_FIELDS:
                continue
            payload.setdefault(key, _copy_jsonable(value))
        for key, value in request.extra_params.items():
            payload[key] = _copy_jsonable(value)
        mapping = self._request_mapping()
        model_path = mapping.get("model_path", "model")
        messages_path = mapping.get("messages_path", "messages")
        if model_path and request.model:
            _write_mapping_path(payload, model_path, request.model)
        if messages_path:
            _write_mapping_path(payload, messages_path, self._serialize_messages(request.messages))
        prompt_path = mapping.get("prompt_path") or mapping.get("input_path")
        if prompt_path:
            _, prompt = _flatten_messages_for_prompt(request.messages, request.system_prompt)
            _write_mapping_path(payload, prompt_path, prompt)
        self._write_optional(payload, mapping.get("temperature_path"), request.temperature)
        self._write_optional(payload, mapping.get("top_p_path"), request.top_p)
        self._write_optional(payload, mapping.get("max_tokens_path"), request.max_tokens)
        self._write_optional(payload, mapping.get("seed_path"), request.seed)
        self._write_optional(payload, mapping.get("stream_path"), request.stream if request.stream else None)
        self._write_optional(payload, mapping.get("stop_sequences_path") or mapping.get("stop_path"), list(request.stop_sequences) if request.stop_sequences else None)
        self._write_optional(payload, mapping.get("response_format_path"), response_format)
        return payload

    def normalize_error(self, *, error: Exception, http_status: int | None = None, payload: Any = None) -> ZoomSearchError:
        return normalize_llm_provider_error(
            error=error,
            request_id=self._context.request_id,
            provider_engine=self.provider_engine,
            provider_model=self.provider_model,
            http_status=http_status,
            payload=payload,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self._context.llm_provider.headers)
        if self._context.llm_provider.api_key:
            headers.setdefault("Authorization", f"Bearer {self._context.llm_provider.api_key}")
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _endpoint_path(self) -> str:
        value = self._context.llm_provider.extra.get("endpoint_path") or self._context.llm_provider.extra.get("path")
        return str(value) if value else ""

    def _response_mapping(self) -> dict[str, str]:
        value = self._context.llm_provider.extra.get("response_mapping")
        return {str(key): str(path) for key, path in value.items()} if isinstance(value, dict) else {}

    def _request_mapping(self) -> dict[str, str]:
        value = self._context.llm_provider.extra.get("request_mapping")
        return {str(key): str(path) for key, path in value.items()} if isinstance(value, dict) else {}

    def _serialize_messages(self, messages: list[UnifiedMessage]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for message in messages:
            item = {"role": message.role, "content": message.content}
            if message.name:
                item["name"] = message.name
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            serialized.append(item)
        return serialized

    def _write_optional(self, payload: dict[str, Any], path: str | None, value: Any) -> None:
        if path and value is not None:
            _write_mapping_path(payload, path, _copy_jsonable(value))


def _query_rewriting_schema() -> dict[str, Any]:
    group_properties = {
        "group": {"type": "integer"},
        "original_input": {"type": "string"},
        "comparison_question": {"type": "string"},
        "split_question": {"type": "string"},
        "main_term": {"type": "string"},
        "key_noun": {"type": "string"},
        "alias1": {"type": "string"},
        "alias2": {"type": "string"},
        "query1": {"type": "string"},
        "query2": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": {
            "previous_conversation": {"type": "array", "items": {"type": "string"}},
            "search_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": group_properties,
                    "required": ["group", "split_question", "main_term", "key_noun", "query1", "query2"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["search_groups"],
        "additionalProperties": False,
    }


def _ensure_object_schema_defaults(schema: dict[str, Any]) -> dict[str, Any]:
    patched = _copy_jsonable(schema)
    patched.setdefault("type", "object")
    patched.setdefault("properties", {})
    patched.setdefault("additionalProperties", False)
    return patched


def _normalize_claude_usage(usage: Any) -> dict[str, int | None]:
    if not isinstance(usage, dict):
        return normalize_usage(None)
    input_tokens = _coerce_int(usage.get("input_tokens"))
    output_tokens = _coerce_int(usage.get("output_tokens"))
    cached_input_tokens = _coerce_int(usage.get("cache_read_input_tokens"))
    total_tokens = None
    if input_tokens is not None or output_tokens is not None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": None,
        "cached_input_tokens": cached_input_tokens,
    }


def _flatten_messages_for_prompt(messages: list[UnifiedMessage], system_prompt: str | None) -> tuple[str | None, str]:
    system_parts: list[str] = []
    if system_prompt and system_prompt.strip():
        system_parts.append(system_prompt)
    prompt_parts: list[str] = []
    for message in messages:
        if message.role == "system":
            if message.content.strip():
                system_parts.append(message.content)
            continue
        prompt_parts.append(f"{message.role.upper()}: {message.content}")
    return ("\n\n".join(system_parts) or None, "\n\n".join(prompt_parts).strip())


def _join_replicate_output(output: Any) -> str | None:
    if isinstance(output, list):
        return "".join(str(item) for item in output)
    if isinstance(output, str):
        return output
    return None


def _extract_json_from_text(text: str | None) -> dict[str, Any] | list[Any] | None:
    if not isinstance(text, str):
        return None
    direct = _try_parse_json_payload(text)
    if direct is not None:
        return direct
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        parsed = _try_parse_json_payload(snippet)
        if parsed is not None:
            return parsed
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        parsed = _try_parse_json_payload(snippet)
        if parsed is not None:
            return parsed
    return None


def _read_nested(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


class ClaudeAdapter(BaseLLMAdapter):
    def __init__(self, *, context: RuntimeContext, transport_context: Any | None) -> None:
        super().__init__(
            capability=context.llm_provider.capability,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
            request_options=context.llm_provider.request_options,
        )
        self._context = context
        self._transport_context = transport_context

    async def generate(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        normalized = self.normalize_request(request)
        payload = self.build_provider_payload(normalized.request, normalized.response_format)
        client = getattr(self._transport_context, "llm_client", None)
        try:
            response = await client.post(
                "/v1/messages",
                json=payload,
                headers=self._build_headers(),
            )
            body = _read_response_payload(response)
            if getattr(response, "status_code", 200) >= 400:
                raise self.normalize_error(
                    error=RuntimeError(f"Claude HTTP {response.status_code}"),
                    http_status=response.status_code,
                    payload=body,
                )
        except ZoomSearchError:
            raise
        except Exception as error:  # noqa: BLE001
            raise self.normalize_error(error=error) from error
        normalized_response = self.parse_provider_response(
            payload=body,
            request=normalized.request,
            normalized_debug=normalized.debug,
        )
        return normalized_response

    async def stream_generate(self, request: UnifiedLLMRequest) -> AsyncIterator[dict[str, Any]]:
        normalized = self.normalize_request(request.copy_with(stream=True))
        if not normalized.request.stream:
            async for event in super().stream_generate(normalized.request):
                yield event
            return

        payload = self.build_provider_payload(normalized.request, normalized.response_format)
        client = getattr(self._transport_context, "llm_client", None)
        text_parts: list[str] = []
        tool_calls: list[UnifiedToolCall] = []
        finish_reason: UnifiedFinishReason = "unknown"
        usage = normalize_usage(None)
        warnings: list[str] = []
        pending_event: str | None = None
        try:
            async with client.stream("POST", "/v1/messages", json=payload, headers=self._build_headers()) as response:
                if getattr(response, "status_code", 200) >= 400:
                    raise self.normalize_error(
                        error=RuntimeError(f"Claude HTTP {response.status_code}"),
                        http_status=response.status_code,
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        pending_event = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        event_payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    normalized_event = self.normalize_stream_event(event_payload)
                    if normalized_event is not None:
                        event_name = str(normalized_event.get("event") or "")
                        if event_name == "token":
                            text_parts.append(str(normalized_event.get("delta_text") or ""))
                        elif event_name == "tool_call_start":
                            tool_calls.append(
                                UnifiedToolCall(
                                    id=str(normalized_event.get("tool_call_id") or f"tool_use_{len(tool_calls)}"),
                                    name=str(normalized_event.get("tool_name") or ""),
                                    arguments_json_text="",
                                    arguments_object={},
                                    provider_call_index=len(tool_calls),
                                )
                            )
                        elif event_name == "tool_arguments" and tool_calls:
                            delta_text = str(normalized_event.get("delta_text") or "")
                            tool_calls[-1].arguments_json_text += delta_text
                        elif event_name == "message_delta":
                            finish_reason = normalized_event.get("finish_reason") or finish_reason
                        yield normalized_event
                    if pending_event == "message_start":
                        message = event_payload.get("message") if isinstance(event_payload.get("message"), dict) else {}
                        usage = _normalize_claude_usage(message.get("usage"))
                    elif pending_event == "message_delta":
                        usage_payload = event_payload.get("usage")
                        if isinstance(usage_payload, dict):
                            usage = _normalize_claude_usage(usage_payload)
                        delta = event_payload.get("delta") if isinstance(event_payload.get("delta"), dict) else {}
                        stop_reason = normalize_finish_reason(delta.get("stop_reason"), error_shape_family=self.capability.error_shape_family)
                        if stop_reason != "unknown":
                            finish_reason = stop_reason
                    pending_event = None
        except ZoomSearchError:
            raise
        except Exception as error:  # noqa: BLE001
            raise self.normalize_error(error=error) from error

        text = "".join(text_parts) or None
        json_payload = _try_parse_json_payload(text) if text is not None and normalized.request.expect_json else None
        if normalized.request.expect_json and json_payload is None and text is not None:
            warnings.append("structured_output_parse_failed")
        for tool_call in tool_calls:
            parsed_arguments = _try_parse_json_object(tool_call.arguments_json_text)
            tool_call.arguments_object = parsed_arguments
        yield {
            "event": "done",
            "response": UnifiedLLMResponse(
                provider=self.provider_engine,
                model=normalized.request.model,
                adapter_class=self.capability.adapter_class,
                text=text,
                json_payload=json_payload,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                warnings=warnings,
                provider_metadata={"normalized_request": normalized.debug},
            ),
        }

    def build_provider_payload(
        self,
        request: UnifiedLLMRequest,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        system_blocks = [message.content for message in request.messages if message.role == "system" and message.content.strip()]
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or 1024,
            "messages": self._build_claude_messages(request.messages),
        }
        if system_blocks:
            payload["system"] = "\n\n".join(system_blocks)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop_sequences:
            payload["stop_sequences"] = list(request.stop_sequences)
        if request.stream:
            payload["stream"] = True
        if request.extra_params.get("metadata") is not None:
            payload["metadata"] = _copy_jsonable(request.extra_params["metadata"])
        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters or {"type": "object", "properties": {}, "additionalProperties": True},
                }
                for tool in request.tools
            ]
            tool_choice = self._normalize_claude_tool_choice(request.tool_choice)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["output_config"] = {"format": self._build_claude_output_format(response_format)}
        return payload

    def parse_provider_response(
        self,
        *,
        payload: dict[str, Any],
        request: UnifiedLLMRequest,
        normalized_debug: dict[str, Any],
    ) -> UnifiedLLMResponse:
        content_blocks = payload.get("content") if isinstance(payload.get("content"), list) else []
        text_parts: list[str] = []
        tool_calls: list[UnifiedToolCall] = []
        for index, block in enumerate(content_blocks):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
                continue
            if block_type == "tool_use":
                tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
                tool_calls.append(
                    UnifiedToolCall(
                        id=str(block.get("id") or f"tool_use_{index}"),
                        name=str(block.get("name") or ""),
                        arguments_json_text=json.dumps(tool_input, ensure_ascii=False),
                        arguments_object=tool_input,
                        provider_call_index=index,
                        raw_tool_call=block,
                    )
                )
        text = "".join(text_parts) or None
        json_payload = _try_parse_json_payload(text) if text is not None and request.expect_json else None
        warnings = []
        if request.expect_json and json_payload is None and text is not None:
            warnings.append("structured_output_parse_failed")
        return UnifiedLLMResponse(
            provider=self.provider_engine,
            model=request.model,
            adapter_class=self.capability.adapter_class,
            text=text,
            json_payload=json_payload,
            tool_calls=tool_calls,
            finish_reason=normalize_finish_reason(payload.get("stop_reason"), error_shape_family=self.capability.error_shape_family),
            usage=_normalize_claude_usage(payload.get("usage")),
            warnings=warnings,
            raw_response=payload,
            provider_metadata={
                "normalized_request": normalized_debug,
                "content_blocks": content_blocks,
                "stop_sequence": payload.get("stop_sequence"),
            },
        )

    def normalize_stream_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = str(event.get("type") or "")
        if event_type == "content_block_delta":
            delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                return {"event": "token", "delta_text": str(delta.get("text") or "")}
            if delta_type == "input_json_delta":
                return {"event": "tool_arguments", "delta_text": str(delta.get("partial_json") or "")}
        if event_type == "content_block_start":
            block = event.get("content_block") if isinstance(event.get("content_block"), dict) else {}
            if block.get("type") == "tool_use":
                return {
                    "event": "tool_call_start",
                    "tool_call_id": str(block.get("id") or ""),
                    "tool_name": str(block.get("name") or ""),
                }
        if event_type == "message_delta":
            delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
            return {
                "event": "message_delta",
                "finish_reason": normalize_finish_reason(delta.get("stop_reason"), error_shape_family=self.capability.error_shape_family),
            }
        if event_type == "message_stop":
            return {"event": "done"}
        return None

    def normalize_error(self, *, error: Exception, http_status: int | None = None, payload: Any = None) -> ZoomSearchError:
        return normalize_llm_provider_error(
            error=error,
            request_id=self._context.request_id,
            provider_engine=self.provider_engine,
            provider_model=self.provider_model,
            http_status=http_status,
            payload=payload,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self._context.llm_provider.headers)
        headers.setdefault("x-api-key", self._context.llm_provider.api_key or "")
        headers.setdefault("anthropic-version", "2023-06-01")
        headers.setdefault("content-type", "application/json")
        return headers

    def _build_claude_messages(self, messages: list[UnifiedMessage]) -> list[dict[str, Any]]:
        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "assistant" and isinstance(message.metadata.get("tool_calls"), list):
                blocks = []
                if message.content.strip():
                    blocks.append({"type": "text", "text": message.content})
                for index, tool_call in enumerate(message.metadata.get("tool_calls") or []):
                    if not isinstance(tool_call, dict):
                        continue
                    arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(tool_call.get("id") or f"tool_call_{index}"),
                            "name": str(tool_call.get("name") or ""),
                            "input": arguments,
                        }
                    )
                normalized_messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": message.content}]})
                continue
            if message.role == "user" and message.tool_call_id:
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.metadata.get("tool_result") or message.content,
                }
                blocks = [tool_result_block]
                trailing_text = message.metadata.get("text")
                if isinstance(trailing_text, str) and trailing_text.strip():
                    blocks.append({"type": "text", "text": trailing_text})
                normalized_messages.append({"role": "user", "content": blocks})
                continue
            normalized_messages.append(
                {
                    "role": message.role,
                    "content": [{"type": "text", "text": message.content}],
                }
            )
        return normalized_messages

    def _build_claude_output_format(self, response_format: dict[str, Any]) -> dict[str, Any]:
        route = response_format.get("type")
        if route == "json_schema":
            schema = response_format.get("json_schema")
            if isinstance(schema, dict) and "schema" in schema and isinstance(schema["schema"], dict):
                schema = schema["schema"]
            sanitized_schema = _ensure_object_schema_defaults(schema if isinstance(schema, dict) else {"type": "object"})
            return {"type": "json_schema", "schema": sanitized_schema}
        return {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        }

    def _normalize_claude_tool_choice(self, tool_choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
        if tool_choice is None or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if isinstance(tool_choice, str):
            return {"type": "tool", "name": tool_choice}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
                return {"type": "tool", "name": str(function.get("name") or "")}
            if tool_choice.get("type") in {"auto", "any", "none", "tool"}:
                return _copy_jsonable(tool_choice)
        return None


class ReplicateAdapter(BaseLLMAdapter):
    def __init__(self, *, context: RuntimeContext, transport_context: Any | None) -> None:
        super().__init__(
            capability=context.llm_provider.capability,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
            request_options=context.llm_provider.request_options,
        )
        self._context = context
        self._transport_context = transport_context

    async def generate(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        normalized = self.normalize_request(request)
        creation = self.build_prediction_request(normalized.request)
        client = getattr(self._transport_context, "llm_client", None)
        try:
            create_response = await client.post(
                self._prediction_path(normalized.request.model),
                json=creation,
                headers=self._build_headers(wait=True),
            )
            prediction = _read_response_payload(create_response)
            if getattr(create_response, "status_code", 200) >= 400:
                raise self.normalize_error(
                    error=RuntimeError(f"Replicate HTTP {create_response.status_code}"),
                    http_status=create_response.status_code,
                    payload=prediction,
                )
        except ZoomSearchError:
            raise
        except Exception as error:  # noqa: BLE001
            raise self.normalize_error(error=error) from error
        completed = await self._await_prediction(
            client=client,
            prediction=prediction,
            timeout_seconds=normalized.request.request_timeout_seconds,
            poll_interval_seconds=self._poll_interval_seconds(normalized.request),
        )
        return self.parse_prediction_response(prediction=completed, request=normalized.request, normalized_debug=normalized.debug)

    async def stream_generate(self, request: UnifiedLLMRequest) -> AsyncIterator[dict[str, Any]]:
        normalized = self.normalize_request(request.copy_with(stream=True))
        if not normalized.request.stream:
            async for event in super().stream_generate(normalized.request):
                yield event
            return

        client = getattr(self._transport_context, "llm_client", None)
        creation = self.build_prediction_request(normalized.request)
        create_response = await client.post(
            self._prediction_path(normalized.request.model),
            json=creation,
            headers=self._build_headers(wait=False),
        )
        prediction = _read_response_payload(create_response)
        if getattr(create_response, "status_code", 200) >= 400:
            raise self.normalize_error(
                error=RuntimeError(f"Replicate HTTP {create_response.status_code}"),
                http_status=create_response.status_code,
                payload=prediction,
            )
        stream_url = _read_nested(prediction, ["urls", "stream"])
        if not isinstance(stream_url, str) or not stream_url:
            yield {
                "event": "done",
                "response": await self.generate(normalized.request.copy_with(stream=False)),
            }
            return

        chunks: list[str] = []
        async with client.stream("GET", stream_url, headers=self._build_headers(wait=False)) as response:
            async for line in response.aiter_lines():
                event = self.normalize_stream_line(line)
                if event is None:
                    continue
                if event.get("event") == "token":
                    chunks.append(str(event.get("delta_text") or ""))
                yield event

        text = "".join(chunks) or None
        json_payload = _extract_json_from_text(text) if normalized.request.expect_json else None
        warnings: list[str] = []
        if normalized.request.expect_json and json_payload is None and text is not None:
            warnings.append("structured_output_parse_failed")
        yield {
            "event": "done",
            "response": UnifiedLLMResponse(
                provider=self.provider_engine,
                model=normalized.request.model,
                adapter_class=self.capability.adapter_class,
                text=text,
                json_payload=json_payload,
                finish_reason="stop",
                usage=normalize_usage(None),
                warnings=warnings,
                provider_metadata={"prediction_id": prediction.get("id"), "stream_url": stream_url, "normalized_request": normalized.debug},
            ),
        }

    def build_prediction_request(self, request: UnifiedLLMRequest) -> dict[str, Any]:
        system_prompt, prompt = _flatten_messages_for_prompt(request.messages, request.system_prompt)
        input_payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": request.max_tokens or 512,
        }
        if system_prompt:
            input_payload["system_prompt"] = system_prompt
        if request.temperature is not None:
            input_payload["temperature"] = request.temperature
        if request.top_p is not None:
            input_payload["top_p"] = request.top_p
        if request.stop_sequences:
            input_payload["stop_sequences"] = ",".join(request.stop_sequences)
        for key, value in request.extra_params.items():
            input_payload.setdefault(key, value)
        body = {"input": input_payload}
        if "/" not in request.model:
            body["version"] = request.model
        return body

    async def _await_prediction(
        self,
        *,
        client: Any,
        prediction: dict[str, Any],
        timeout_seconds: int,
        poll_interval_seconds: float,
    ) -> dict[str, Any]:
        status = str(prediction.get("status") or "")
        if status in {"succeeded", "failed", "canceled"}:
            return prediction
        get_url = _read_nested(prediction, ["urls", "get"])
        if not isinstance(get_url, str) or not get_url:
            return prediction
        if timeout_seconds <= 0:
            return prediction
        attempts = max(1, int(timeout_seconds / max(poll_interval_seconds, 0.1)))
        current = prediction
        for _ in range(attempts):
            poll_response = await client.get(get_url, headers=self._build_headers(wait=False))
            current = _read_response_payload(poll_response)
            if getattr(poll_response, "status_code", 200) >= 400:
                raise self.normalize_error(
                    error=RuntimeError(f"Replicate HTTP {poll_response.status_code}"),
                    http_status=poll_response.status_code,
                    payload=current,
                )
            status = str(current.get("status") or "")
            if status in {"succeeded", "failed", "canceled"}:
                return current
            if poll_interval_seconds > 0:
                await asyncio.sleep(poll_interval_seconds)
        return current

    async def stream(self, request: UnifiedLLMRequest) -> list[dict[str, Any]]:
        normalized = self.normalize_request(request.copy_with(stream=True))
        creation = self.build_prediction_request(normalized.request)
        client = getattr(self._transport_context, "llm_client", None)
        create_response = await client.post(
            self._prediction_path(normalized.request.model),
            json=creation,
            headers=self._build_headers(wait=False),
        )
        prediction = _read_response_payload(create_response)
        if getattr(create_response, "status_code", 200) >= 400:
            raise self.normalize_error(
                error=RuntimeError(f"Replicate HTTP {create_response.status_code}"),
                http_status=create_response.status_code,
                payload=prediction,
            )
        stream_url = _read_nested(prediction, ["urls", "stream"])
        if not isinstance(stream_url, str) or not stream_url:
            return []
        events: list[dict[str, Any]] = []
        async with client.stream("GET", stream_url, headers=self._build_headers(wait=False)) as response:
            async for line in response.aiter_lines():
                parsed = self.normalize_stream_line(line)
                if parsed is not None:
                    events.append(parsed)
        return events

    def normalize_stream_line(self, line: str) -> dict[str, Any] | None:
        if not line:
            return None
        if line.startswith("event: output"):
            return {"event": "output"}
        if line.startswith("event: done"):
            return {"event": "done"}
        if line.startswith("event: error"):
            return {"event": "error"}
        if line.startswith("data:"):
            return {"event": "token", "delta_text": line[5:].lstrip()}
        return None

    def parse_prediction_response(
        self,
        *,
        prediction: dict[str, Any],
        request: UnifiedLLMRequest,
        normalized_debug: dict[str, Any],
    ) -> UnifiedLLMResponse:
        status = str(prediction.get("status") or "")
        if status == "failed":
            raise self.normalize_error(
                error=RuntimeError(str(prediction.get("error") or "Prediction failed")),
                payload=prediction,
            )
        if status == "canceled":
            raise self.normalize_error(
                error=RuntimeError("Prediction canceled"),
                payload=prediction,
            )
        raw_output = prediction.get("output")
        text = _join_replicate_output(raw_output)
        json_payload = _extract_json_from_text(text) if request.expect_json else None
        warnings: list[str] = []
        if request.expect_json and json_payload is None:
            warnings.append("structured_output_parse_failed")
        return UnifiedLLMResponse(
            provider=self.provider_engine,
            model=request.model,
            adapter_class=self.capability.adapter_class,
            text=text,
            json_payload=json_payload,
            tool_calls=[],
            finish_reason=normalize_finish_reason(status, error_shape_family=self.capability.error_shape_family),
            usage=normalize_usage(None),
            warnings=warnings,
            raw_response=prediction,
            provider_metadata={
                "normalized_request": normalized_debug,
                "prediction_id": prediction.get("id"),
                "prediction_status": status,
                "stream_url": _read_nested(prediction, ["urls", "stream"]),
                "metrics": prediction.get("metrics"),
            },
        )

    def normalize_error(self, *, error: Exception, http_status: int | None = None, payload: Any = None) -> ZoomSearchError:
        mapped_payload = payload
        if isinstance(payload, dict) and "detail" in payload and "error" not in payload:
            mapped_payload = {"error": {"message": payload.get("detail"), "type": payload.get("status") or "replicate_error", "code": payload.get("code")}}
        if isinstance(payload, dict) and payload.get("error") and "detail" not in payload and not isinstance(payload.get("error"), dict):
            mapped_payload = {"error": {"message": payload.get("error"), "type": payload.get("status") or "prediction_failed", "code": payload.get("code")}}
        return normalize_llm_provider_error(
            error=error,
            request_id=self._context.request_id,
            provider_engine=self.provider_engine,
            provider_model=self.provider_model,
            http_status=http_status,
            payload=mapped_payload,
        )

    def _prediction_path(self, model: str) -> str:
        if "/" in model:
            owner, model_name = model.split("/", 1)
            return f"/models/{owner}/{model_name}/predictions"
        return "/predictions"

    def _build_headers(self, *, wait: bool) -> dict[str, str]:
        headers = dict(self._context.llm_provider.headers)
        headers.setdefault("Authorization", f"Bearer {self._context.llm_provider.api_key or ''}")
        headers.setdefault("content-type", "application/json")
        if wait:
            headers["Prefer"] = "wait"
        return headers

    def _poll_interval_seconds(self, request: UnifiedLLMRequest) -> float:
        value = request.extra_params.get("poll_interval_seconds", self._context.llm_provider.extra.get("poll_interval_seconds", 1.0))
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 1.0


def build_response_format_route(*, capability: ProviderCapability, request: UnifiedLLMRequest) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    mode = "text"
    route = capability.recommended_query_rewriting_mode
    response_format: dict[str, Any] | None = None

    if request.expect_json:
        if request.json_schema and capability.structured_output_level in {
            "native_json_schema_strict",
            "native_json_schema_subset",
        }:
            mode = "json_schema"
            route = "json_schema"
            response_format = {
                "type": "json_schema",
                "json_schema": request.json_schema,
            }
        elif capability.structured_output_level in {
            "native_json_schema_strict",
            "native_json_schema_subset",
            "native_json_object_only",
        }:
            mode = "json_object"
            route = "json_object"
            response_format = {"type": "json_object"}
        else:
            mode = "prompt_only_json"
            route = "prompt_only_json"
            response_format = None
    elif request.tools and capability.tool_calling_level != "unsupported":
        mode = "text"
        route = "tools"

    return response_format, {
        "mode": mode,
        "route": route,
        "expects_json": request.expect_json,
        "has_schema": request.json_schema is not None,
        "has_tools": bool(request.tools),
    }


def normalize_finish_reason(value: object, *, error_shape_family: str = "openai_like") -> UnifiedFinishReason:
    text = str(value or "").strip().lower()
    if text in {"stop", "end_turn"}:
        return "stop"
    if text in {"length", "max_tokens", "model_context_window_exceeded"}:
        return "length"
    if text in {"tool_calls", "tool_use", "function_call"}:
        return "tool_calls"
    if text in {"content_filter", "sensitive", "safety", "moderation"}:
        return "content_filter"
    if text in {"cancelled", "canceled"}:
        return "cancelled"
    if text == "timeout":
        return "timeout"
    if text in {"error", "failed", "network_error"}:
        return "error"
    if error_shape_family == "stateful_async" and text in {"succeeded", "completed"}:
        return "stop"
    return "unknown"


def normalize_usage(payload: Any) -> dict[str, int | None]:
    if not isinstance(payload, dict):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
        }

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else payload
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    completion_details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    return {
        "input_tokens": _coerce_int(usage.get("prompt_tokens") if isinstance(usage, dict) else None),
        "output_tokens": _coerce_int(usage.get("completion_tokens") if isinstance(usage, dict) else None),
        "total_tokens": _coerce_int(usage.get("total_tokens") if isinstance(usage, dict) else None),
        "reasoning_tokens": _coerce_int(completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None),
        "cached_input_tokens": _coerce_int(prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None),
    }


def normalize_tool_calls(payload: Any) -> list[UnifiedToolCall]:
    if not isinstance(payload, list):
        return []

    normalized: list[UnifiedToolCall] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments_text = json.dumps(arguments, ensure_ascii=False)
            arguments_object = arguments
        else:
            arguments_text = str(arguments or "{}")
            arguments_object = _try_parse_json_object(arguments_text)
        normalized.append(
            UnifiedToolCall(
                id=str(item.get("id") or f"tool_call_{index}"),
                name=str(function.get("name") or ""),
                arguments_json_text=arguments_text,
                arguments_object=arguments_object,
                provider_call_index=index,
                raw_tool_call=item,
            )
        )
    return normalized


def normalize_llm_response_payload(*, payload: Any, capability: ProviderCapability, provider: str, model: str | None = None) -> UnifiedLLMResponse:
    if not isinstance(payload, dict):
        text = str(payload) if payload is not None else None
        return UnifiedLLMResponse(
            provider=provider,
            model=model,
            adapter_class=capability.adapter_class,
            text=text,
            finish_reason="unknown",
            usage=normalize_usage(None),
            raw_response=payload,
        )

    payload = apply_provider_response_patches(payload=payload, capability=capability)
    message = _read_first_message(payload)
    text = _coerce_text(message.get("content")) if isinstance(message, dict) else None
    tool_calls = normalize_tool_calls(message.get("tool_calls") if isinstance(message, dict) else None)
    finish_reason = normalize_finish_reason(_read_finish_reason(payload), error_shape_family=capability.error_shape_family)

    json_payload = None
    warnings: list[str] = []
    if isinstance(text, str):
        json_payload = _try_parse_json_payload(text)
        if json_payload is None and capability.structured_output_level in {
            "native_json_schema_strict",
            "native_json_schema_subset",
            "native_json_object_only",
        }:
            warnings.append("structured_output_parse_failed")

    provider_metadata = _collect_provider_metadata(payload=payload, message=message, capability=capability)
    if provider_metadata.get("patched_finish_reason"):
        warnings.append("patched_finish_reason")
    if provider_metadata.get("patched_tool_arguments"):
        warnings.append("patched_tool_arguments")
    if provider_metadata.get("business_error_detected_in_200"):
        warnings.append("business_error_detected_in_200")

    return UnifiedLLMResponse(
        provider=provider,
        model=model,
        adapter_class=capability.adapter_class,
        text=text,
        json_payload=json_payload,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=normalize_usage(payload),
        warnings=warnings,
        raw_response=payload,
        provider_metadata=provider_metadata,
    )


def normalize_custom_native_llm_response_payload(
    *,
    payload: Any,
    mapping: dict[str, str],
    capability: ProviderCapability,
    provider: str,
    model: str | None = None,
    expect_json: bool = False,
    normalized_debug: dict[str, Any] | None = None,
) -> UnifiedLLMResponse:
    text_value = _read_mapping_path(payload, mapping.get("content_path") or mapping.get("text_path"))
    text = _coerce_text(text_value)
    json_payload = _read_mapping_path(payload, mapping.get("json_payload_path"))
    if json_payload is None and isinstance(text, str):
        json_payload = _try_parse_json_payload(text)
    if json_payload is not None and not isinstance(json_payload, (dict, list)):
        json_payload = None

    warnings: list[str] = []
    if expect_json and json_payload is None:
        warnings.append("structured_output_parse_failed")

    finish_reason_value = _read_mapping_path(payload, mapping.get("finish_reason_path"))
    usage = _normalize_custom_native_usage(payload=payload, mapping=mapping)
    provider_metadata: dict[str, Any] = {
        "normalized_request": normalized_debug or {},
        "response_mapping": dict(mapping),
    }

    return UnifiedLLMResponse(
        provider=provider,
        model=model,
        adapter_class=capability.adapter_class,
        text=text,
        json_payload=json_payload,
        tool_calls=[],
        finish_reason=normalize_finish_reason(finish_reason_value, error_shape_family=capability.error_shape_family),
        usage=usage,
        warnings=warnings,
        raw_response=payload,
        provider_metadata=provider_metadata,
    )


def _normalize_custom_native_usage(*, payload: Any, mapping: dict[str, str]) -> dict[str, int | None]:
    usage_path = mapping.get("usage_path")
    if usage_path:
        usage_payload = _read_mapping_path(payload, usage_path)
        normalized = normalize_usage(usage_payload)
    else:
        normalized = normalize_usage(None)
    overrides = {
        "input_tokens": mapping.get("usage_input_tokens_path"),
        "output_tokens": mapping.get("usage_output_tokens_path"),
        "total_tokens": mapping.get("usage_total_tokens_path"),
        "reasoning_tokens": mapping.get("usage_reasoning_tokens_path"),
        "cached_input_tokens": mapping.get("usage_cached_input_tokens_path"),
    }
    for key, path in overrides.items():
        if path:
            normalized[key] = _coerce_int(_read_mapping_path(payload, path))
    return normalized


def _read_mapping_path(payload: Any, path: str | None) -> Any:
    if not path:
        return None
    current = payload
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, list) and segment.isdigit():
            index = int(segment)
            current = current[index] if 0 <= index < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _write_mapping_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = target
    for index, segment in enumerate(parts):
        is_last = index == len(parts) - 1
        list_mode = segment.endswith("[]")
        key = segment[:-2] if list_mode else segment
        if is_last:
            if list_mode:
                current[key] = value if isinstance(value, list) else [value]
            else:
                current[key] = value
            return
        if list_mode:
            bucket = current.setdefault(key, [])
            if not bucket:
                bucket.append({})
            current = bucket[0]
            continue
        bucket = current.setdefault(key, {})
        if not isinstance(bucket, dict):
            bucket = {}
            current[key] = bucket
        current = bucket


def normalize_llm_provider_error(
    *,
    error: Exception | Any,
    request_id: str,
    provider_engine: str,
    provider_model: str | None = None,
    http_status: int | None = None,
    payload: Any = None,
) -> ZoomSearchError:
    if isinstance(error, ZoomSearchError):
        return error

    fields = extract_openai_error_fields(payload)
    if provider_engine in {"minimax-global", "minimax-china"} and isinstance(payload, dict):
        base_resp = payload.get("base_resp") if isinstance(payload.get("base_resp"), dict) else {}
        base_code = base_resp.get("status_code")
        base_message = base_resp.get("status_msg")
        if base_code not in {None, 0, "0"}:
            fields["code"] = str(base_code)
            if not fields.get("message") and base_message:
                fields["message"] = str(base_message)
    reason, retryable = map_provider_error_reason(
        capability_engine=provider_engine,
        http_status=http_status,
        error_code=fields["code"],
        error_type=fields["type"],
        message=fields["message"],
    )
    message = fields["message"] or str(error)
    return call_failure(
        category="llm_call_failure",
        component="llm",
        message=message,
        user_message="LLM request failed.",
        request_id=request_id,
        provider_engine=provider_engine,
        provider_model=provider_model,
        reason_code=reason or llm_http_error_reason(http_status) or "provider_call_failed",
        http_status=http_status,
        retryable=retryable,
        provider_error_code=fields["code"],
        provider_error_type=fields["type"],
        provider_error_param=fields["param"],
        raw_diagnostics=RawDiagnostics(
            http_status=http_status,
            provider_error_body=payload,
            provider_error_code=fields["code"],
            provider_error_type=fields["type"],
            provider_error_param=fields["param"],
            transport_exception_class=type(error).__name__,
            transport_exception_message=str(error),
        ),
    )


def apply_provider_request_patches(
    *,
    capability: ProviderCapability,
    request: UnifiedLLMRequest,
    response_format: dict[str, Any] | None,
    request_options: LLMRequestOptions | None = None,
) -> ProviderPatchResult:
    engine = capability.engine
    working_request = request
    working_response_format = _copy_jsonable(response_format)
    omitted: list[str] = []
    patched: dict[str, Any] = {}
    warnings: list[str] = []
    reasoning_debug = _initialize_reasoning_debug(capability=capability, request=request, request_options=request_options)

    if engine == "gemini":
        for field in ("logprobs", "top_logprobs"):
            if field in working_request.extra_params:
                working_request = _drop_extra_param(working_request, field)
                omitted.append(field)
        if working_request.seed is not None:
            working_request = working_request.copy_with(seed=None)
            omitted.append("seed")
        if working_request.tools:
            working_request = _set_extra_param(working_request, "parallel_tool_calls", False)
            patched["parallel_tool_calls"] = False
        if working_response_format and working_response_format.get("type") == "json_schema":
            schema = working_response_format.get("json_schema")
            sanitized = sanitize_schema_for_provider(schema, provider=engine)
            if sanitized != schema:
                working_response_format["json_schema"] = sanitized
                patched["json_schema_sanitized"] = True

    if engine in {"doubao-global", "doubao-china"}:
        model_override = request.extra_params.get("endpoint_id") or request.extra_params.get("model_id")
        if model_override and model_override != working_request.model:
            working_request = working_request.copy_with(model=str(model_override))
            patched["model"] = str(model_override)
        if _should_apply_default_reasoning_disable(request_options):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source=_reasoning_patch_source(request_options))
        if working_response_format and working_response_format.get("type") == "json_schema":
            schema_payload = working_response_format.get("json_schema")
            if isinstance(schema_payload, dict) and "strict" not in schema_payload:
                working_response_format["strict"] = True
                patched["response_format.strict"] = True
            warnings.append("doubao_json_schema_beta")
        if engine == "doubao-china" and working_response_format and working_response_format.get("type") == "json_object":
            working_request, injected = ensure_json_keyword_hint(working_request)
            if injected:
                patched["json_keyword_hint"] = True

    if engine in {"qwen-global", "qwen-china"}:
        if working_response_format and working_response_format.get("type") == "json_schema":
            working_response_format = {"type": "json_object"}
            patched["response_format"] = "json_object"
            warnings.append("json_schema_downgraded_to_json_object")
        if working_request.expect_json or (working_response_format and working_response_format.get("type") == "json_object"):
            working_request, injected = ensure_json_keyword_hint(working_request)
            if injected:
                patched["json_keyword_hint"] = True
            if _should_apply_default_reasoning_disable(request_options):
                thinking_key = "enable_thinking" if engine == "qwen-china" else "thinking"
                thinking_value = False if engine == "qwen-china" else {"type": "disabled"}
                working_request = _set_extra_param(working_request, thinking_key, thinking_value)
                patched[thinking_key] = thinking_value
                _record_reasoning_patch(reasoning_debug, mode="disabled", param=thinking_key, value=thinking_value, source=_reasoning_patch_source(request_options))
        elif _is_reasoning_off_requested(request_options):
            thinking_key = "enable_thinking" if engine == "qwen-china" else "thinking"
            thinking_value = False if engine == "qwen-china" else {"type": "disabled"}
            working_request = _set_extra_param(working_request, thinking_key, thinking_value)
            patched[thinking_key] = thinking_value
            _record_reasoning_patch(reasoning_debug, mode="disabled", param=thinking_key, value=thinking_value, source="explicit_false")
        if working_request.tools and working_request.stream:
            working_request = working_request.copy_with(stream=False)
            patched["stream"] = False
            warnings.append("tools_disable_stream")
        if working_request.stream:
            working_request = _set_extra_param(working_request, "stream_options", {"include_usage": True})
            patched["stream_options.include_usage"] = True

    if engine == "glm-china":
        if working_response_format and working_response_format.get("type") == "json_schema":
            working_response_format = {"type": "json_object"}
            patched["response_format"] = "json_object"
            warnings.append("json_schema_downgraded_to_json_object")
        if working_request.tools and not _is_auto_tool_choice(working_request.tool_choice):
            working_request = working_request.copy_with(tool_choice="auto")
            patched["tool_choice"] = "auto"
        if working_request.expect_json:
            working_request, injected = ensure_json_keyword_hint(working_request)
            if injected:
                patched["json_keyword_hint"] = True
        if _is_reasoning_off_requested(request_options):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
        if working_request.temperature == 0:
            working_request = _set_extra_param(working_request, "do_sample", False)
            patched["do_sample"] = False
            warnings.append("determinism_via_do_sample_false")

    if engine == "glm-global":
        if request.json_schema is not None:
            warnings.append("json_schema_downgraded_to_json_object")
        if _is_reasoning_off_requested(request_options):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
        elif working_request.expect_json and _should_apply_default_reasoning_disable(request_options):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source=_reasoning_patch_source(request_options))
        if working_request.expect_json:
            working_request, injected = ensure_json_keyword_hint(working_request)
            if injected:
                patched["json_keyword_hint"] = True
        if working_request.temperature == 0:
            working_request = working_request.copy_with(temperature=0.01)
            patched["temperature"] = 0.01
            warnings.append("glm_global_temperature_raised_for_stability")
        if working_request.top_p is not None:
            top_p = min(max(working_request.top_p, 0.01), 1.0)
            if top_p != working_request.top_p:
                working_request = working_request.copy_with(top_p=top_p)
                patched["top_p"] = top_p
        if len(working_request.stop_sequences) > 1:
            working_request = working_request.copy_with(stop_sequences=working_request.stop_sequences[:1])
            patched["stop_sequences"] = working_request.stop_sequences
            warnings.append("glm_global_stop_sequences_trimmed")
        if working_response_format and working_response_format.get("type") == "json_schema":
            working_response_format = {"type": "json_object"}
            patched["response_format"] = "json_object"
            warnings.append("json_schema_downgraded_to_json_object")
        if working_request.tools and not _is_auto_tool_choice(working_request.tool_choice):
            working_request = working_request.copy_with(tool_choice="auto")
            patched["tool_choice"] = "auto"
            warnings.append("tool_choice_downgraded_to_auto")
        if working_request.seed is not None:
            working_request = working_request.copy_with(seed=None)
            omitted.append("seed")

    if engine in {"kimi-global", "kimi-china"} and working_request.temperature is not None:
        working_request = working_request.copy_with(temperature=None)
        omitted.append("temperature")
        warnings.append("temperature_omitted_for_provider_compatibility")

    if engine in {"kimi-global", "kimi-china"} and _is_reasoning_off_requested(request_options):
        if _supports_kimi_reasoning_disable(working_request.model):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
        elif _is_kimi_reasoning_always_on(working_request.model):
            warnings.append("reasoning_always_on_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="always_on", warning="reasoning_always_on_for_model")
        elif _is_moonshot_non_reasoning_family(working_request.model):
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")
        else:
            warnings.append("reasoning_control_unknown_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")

    if engine == "deepseek" and _is_reasoning_off_requested(request_options):
        if _is_deepseek_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
        else:
            warnings.append("reasoning_control_unknown_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")

    if engine == "openai" and _is_reasoning_off_requested(request_options):
        if _is_openai_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "reasoning", {"effort": "none"})
            patched["reasoning"] = {"effort": "none"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="reasoning", value={"effort": "none"}, source="explicit_false")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")

    if engine == "together" and _is_reasoning_off_requested(request_options):
        working_request = _set_extra_param(working_request, "reasoning", {"enabled": False})
        patched["reasoning"] = {"enabled": False}
        _record_reasoning_patch(reasoning_debug, mode="disabled", param="reasoning", value={"enabled": False}, source="explicit_false")

    if engine in {"fireworks", "openrouter"} and _is_reasoning_off_requested(request_options):
        working_request = _set_extra_param(working_request, "reasoning", {"effort": "none"})
        patched["reasoning"] = {"effort": "none"}
        _record_reasoning_patch(reasoning_debug, mode="disabled", param="reasoning", value={"effort": "none"}, source="explicit_false")

    if engine == "cohere" and _is_reasoning_off_requested(request_options):
        working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
        patched["thinking"] = {"type": "disabled"}
        _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")

    if engine == "siliconflow" and _is_reasoning_off_requested(request_options):
        if _is_siliconflow_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "enable_thinking", False)
            patched["enable_thinking"] = False
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="enable_thinking", value=False, source="explicit_false")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")

    if engine == "novita" and _is_reasoning_off_requested(request_options):
        if _is_novita_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "enable_thinking", False)
            patched["enable_thinking"] = False
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="enable_thinking", value=False, source="explicit_false")
        else:
            warnings.append("reasoning_control_unknown_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")

    if engine == "deepinfra" and _is_reasoning_off_requested(request_options):
        working_request = _set_extra_param(working_request, "reasoning", {"enabled": False})
        patched["reasoning"] = {"enabled": False}
        _record_reasoning_patch(reasoning_debug, mode="disabled", param="reasoning", value={"enabled": False}, source="explicit_false")

    if engine == "hunyuan" and _is_reasoning_off_requested(request_options):
        if _is_hunyuan_tokenhub_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
        elif _is_hunyuan_legacy_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "EnableThinking", False)
            patched["EnableThinking"] = False
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="EnableThinking", value=False, source="explicit_false")
        elif _is_hunyuan_always_on_reasoning_model(working_request.model):
            warnings.append("reasoning_disable_unsupported_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="unsupported_for_model", warning="reasoning_disable_unsupported_for_model")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")

    if engine == "stepfun" and _is_reasoning_off_requested(request_options):
        if _is_stepfun_reasoning_model(working_request.model):
            warnings.append("reasoning_disable_unsupported_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="unsupported_for_model", warning="reasoning_disable_unsupported_for_model")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")

    if engine == "groq" and _is_reasoning_off_requested(request_options):
        if _is_groq_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "reasoning_effort", "none")
            patched["reasoning_effort"] = "none"
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="reasoning_effort", value="none", source="explicit_false")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")
            warnings.append("reasoning_control_unknown_for_model")

    if engine == "cerebras" and _is_reasoning_off_requested(request_options):
        if _is_cerebras_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "reasoning_effort", "none")
            patched["reasoning_effort"] = "none"
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="reasoning_effort", value="none", source="explicit_false")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")

    if engine == "grok" and _is_reasoning_off_requested(request_options):
        if _is_grok_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "reasoning", {"effort": "none"})
            patched["reasoning"] = {"effort": "none"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="reasoning", value={"effort": "none"}, source="explicit_false")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")

    if engine == "mistral" and _is_reasoning_off_requested(request_options):
        if _is_mistral_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "reasoning_effort", "none")
            patched["reasoning_effort"] = "none"
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="reasoning_effort", value="none", source="explicit_false")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")

    if engine == "mimo" and _is_reasoning_off_requested(request_options):
        if _is_mimo_reasoning_model(working_request.model):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
        else:
            _record_reasoning_diagnostic(reasoning_debug, status="not_applicable")

    if engine == "ollama" and _is_reasoning_off_requested(request_options):
        if _is_ollama_disable_unsupported_model(working_request.model):
            warnings.append("reasoning_disable_unsupported_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="unsupported_for_model", warning="reasoning_disable_unsupported_for_model")
        else:
            working_request = _set_extra_param(working_request, "think", False)
            patched["think"] = False
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="think", value=False, source="explicit_false")

    if engine == "baichuan":
        if request.json_schema is not None:
            warnings.append("json_schema_downgraded_to_json_object")
        working_request = _set_extra_param(working_request, "top_k", working_request.extra_params.get("top_k", 5))
        patched["top_k"] = working_request.extra_params.get("top_k", 5)
        working_request = _set_extra_param(
            working_request,
            "with_search_enhance",
            working_request.extra_params.get("with_search_enhance", False),
        )
        patched["with_search_enhance"] = working_request.extra_params.get("with_search_enhance", False)
        if working_response_format and working_response_format.get("type") == "json_schema":
            working_response_format = {"type": "json_object"}
            patched["response_format"] = "json_object"
            warnings.append("json_schema_downgraded_to_json_object")
        if working_request.expect_json:
            working_request, injected = ensure_json_keyword_hint(working_request)
            if injected:
                patched["json_keyword_hint"] = True

    if engine == "spark":
        if not _supports_spark_system_messages(working_request.model):
            folded_request = _fold_system_messages_into_first_user_message(working_request)
            if folded_request.messages != working_request.messages:
                working_request = folded_request
                patched["messages"] = "system_folded_into_user"
                warnings.append("spark_system_folded_into_user")
        if request.json_schema is not None:
            warnings.append("json_schema_downgraded_to_json_object")
        if working_response_format and working_response_format.get("type") == "json_schema":
            working_response_format = {"type": "json_object"}
            patched["response_format"] = "json_object"
            warnings.append("json_schema_downgraded_to_json_object")
        if working_request.expect_json:
            working_request, injected = ensure_json_keyword_hint(working_request)
            if injected:
                patched["json_keyword_hint"] = True
        if working_request.tools and _supports_spark_tools(working_request.model):
            working_request = _set_extra_param(working_request, "tool_calls_switch", True)
            patched["tool_calls_switch"] = True
        elif working_request.tools:
            working_request = working_request.copy_with(tools=[], tool_choice=None)
            patched["tools"] = "omitted_for_model"
            warnings.append("spark_tools_unsupported_for_model")

    if engine == "huggingface":
        if "/" not in working_request.model:
            warnings.append("huggingface_model_validation_failed")
        if _is_reasoning_off_requested(request_options):
            warnings.append("reasoning_control_unknown_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")
        if working_request.tools:
            working_request = working_request.copy_with(tools=[], tool_choice=None)
            omitted.extend(["tools", "tool_choice"])
            warnings.append("tools_unsupported_for_huggingface_default")
        if working_response_format:
            route_type = working_response_format.get("type")
            if route_type == "json_object":
                working_response_format = {"type": "json", "value": {"type": "object"}}
                patched["response_format"] = "hf_json_object"
                warnings.append("response_format_rewritten_for_huggingface")
            elif route_type == "json_schema":
                schema_payload = working_response_format.get("json_schema")
                schema_value = schema_payload.get("schema") if isinstance(schema_payload, dict) and "schema" in schema_payload else schema_payload
                patched_schema = _patch_huggingface_schema(schema_value)
                working_response_format = {"type": "json_schema", "value": patched_schema}
                patched["response_format"] = "hf_json_schema"
                if patched_schema != schema_value:
                    patched["json_schema_patched"] = True
                warnings.append("response_format_rewritten_for_huggingface")

    if engine == "perplexity" and _is_reasoning_off_requested(request_options):
        warnings.append("reasoning_control_unknown_for_model")
        _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")

    if engine == "lepton" and _is_reasoning_off_requested(request_options):
        warnings.append("reasoning_control_unknown_for_model")
        _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")

    if engine in {"minimax-global", "minimax-china"}:
        if working_request.temperature == 0:
            working_request = working_request.copy_with(temperature=0.01)
            patched["temperature"] = 0.01
            warnings.append("minimax_temperature_raised_for_stability")
        if working_request.temperature is not None and working_request.temperature > 1.0:
            working_request = working_request.copy_with(temperature=1.0)
            patched["temperature"] = 1.0
        if working_request.max_tokens is not None:
            provider_max_tokens = working_request.extra_params.get(capability.max_output_tokens_param)
            target_max_tokens = working_request.max_tokens
            if provider_max_tokens is not None:
                coerced_provider_max_tokens = _coerce_int(provider_max_tokens)
                if coerced_provider_max_tokens is not None:
                    target_max_tokens = min(target_max_tokens, coerced_provider_max_tokens)
                    warnings.append("minimax_max_tokens_reconciled")
            working_request = _drop_extra_param(working_request, "max_tokens")
            working_request = _set_extra_param(working_request, capability.max_output_tokens_param, target_max_tokens)
            patched[capability.max_output_tokens_param] = target_max_tokens
        if engine == "minimax-global":
            if _is_reasoning_off_requested(request_options):
                if _is_minimax_m3_model(working_request.model):
                    working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
                    patched["thinking"] = {"type": "disabled"}
                    _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
                elif _is_minimax_m2_model(working_request.model):
                    warnings.append("reasoning_always_on_for_model")
                    _record_reasoning_diagnostic(reasoning_debug, status="always_on", warning="reasoning_always_on_for_model")
                else:
                    warnings.append("reasoning_control_unknown_for_model")
                    _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")
            working_request = _set_extra_param(working_request, "reasoning_split", True)
            patched["reasoning_split"] = True
            if request.json_schema is not None:
                warnings.append("minimax-global-schema-downgraded-to-prompt-json")
            elif request.json_object:
                warnings.append("minimax-global-json-object-ignored-by-provider")
            if working_response_format and working_response_format.get("type") == "json_schema":
                working_response_format = None
                patched["response_format"] = "prompt_only_json"
            elif working_response_format and working_response_format.get("type") == "json_object":
                working_response_format = None
                patched["response_format"] = "prompt_only_json"
        if engine == "minimax-china":
            working_request = working_request.copy_with(tool_choice="auto") if working_request.tools and not _is_auto_tool_choice(working_request.tool_choice) else working_request
            if working_request.tools and not _is_auto_tool_choice(request.tool_choice):
                patched["tool_choice"] = "auto"
            route = _select_minimax_china_route(request=working_request, response_format=working_response_format)
            patched["route"] = route
            if route == "native":
                if _is_reasoning_off_requested(request_options):
                    if _is_minimax_m3_model(working_request.model):
                        working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
                        patched["thinking"] = {"type": "disabled"}
                        _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
                    elif _is_minimax_m2_model(working_request.model):
                        warnings.append("reasoning_always_on_for_model")
                        _record_reasoning_diagnostic(reasoning_debug, status="always_on", warning="reasoning_always_on_for_model")
                    else:
                        warnings.append("reasoning_control_unknown_for_model")
                        _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")
                structured_model = working_request.extra_params.get("structured_output_model")
                if isinstance(structured_model, str) and structured_model.strip():
                    working_request = working_request.copy_with(model=structured_model.strip())
                    patched["model"] = structured_model.strip()
                if working_response_format and working_response_format.get("type") == "json_schema":
                    working_request = _set_extra_param(working_request, "native_response_format", _copy_jsonable(working_response_format))
                    patched["native_response_format"] = "json_schema"
                working_request = _set_extra_param(working_request, "api_route", "native_chatcompletion_v2")
                patched["api_route"] = "native_chatcompletion_v2"
            else:
                working_request = _set_extra_param(working_request, "reasoning_split", True)
                patched["reasoning_split"] = True
                if working_response_format is not None:
                    working_response_format = None
                    patched["response_format"] = "prompt_only_json"
                    warnings.append("minimax-china-compat-structured-output-downgraded")
                if working_request.expect_json:
                    working_request, injected = ensure_json_keyword_hint(working_request)
                    if injected:
                        patched["json_keyword_hint"] = True

    if engine == "spark" and _is_reasoning_off_requested(request_options):
        if _is_spark_x1_model(working_request.model):
            working_request = _set_extra_param(working_request, "thinking", {"type": "disabled"})
            patched["thinking"] = {"type": "disabled"}
            _record_reasoning_patch(reasoning_debug, mode="disabled", param="thinking", value={"type": "disabled"}, source="explicit_false")
        else:
            warnings.append("reasoning_control_unknown_for_model")
            _record_reasoning_diagnostic(reasoning_debug, status="unknown_model_family", warning="reasoning_control_unknown_for_model")

    if engine == "claude":
        if _is_reasoning_off_requested(request_options):
            _record_reasoning_diagnostic(reasoning_debug, status="omission_based_disable")
        elif _is_reasoning_on_requested(request_options):
            warnings.append("reasoning_enable_not_implemented")
            _record_reasoning_diagnostic(reasoning_debug, status="enable_not_implemented", warning="reasoning_enable_not_implemented")

    return ProviderPatchResult(
        request=working_request,
        response_format=working_response_format,
        omitted_params=omitted,
        patched_params=patched,
        warnings=warnings,
        debug={
            "engine": engine,
            "reasoning": reasoning_debug,
            "response_format": working_response_format,
            "warnings": warnings,
        },
    )


def _initialize_reasoning_debug(
    *,
    capability: ProviderCapability,
    request: UnifiedLLMRequest,
    request_options: LLMRequestOptions | None,
) -> dict[str, Any]:
    return {
        "requested": None if request_options is None else request_options.reasoning,
        "provider_support": capability.supports_reasoning_control,
        "model": request.model,
        "status": "provider_default",
        "param": None,
        "value": None,
        "source": None,
        "warning": None,
    }


def _record_reasoning_patch(debug: dict[str, Any], *, mode: str, param: str, value: Any, source: str) -> None:
    debug["status"] = mode
    debug["param"] = param
    debug["value"] = _copy_jsonable(value)
    debug["source"] = source


def _record_reasoning_diagnostic(debug: dict[str, Any], *, status: str, warning: str | None = None) -> None:
    debug["status"] = status
    if warning:
        debug["warning"] = warning


def _is_reasoning_off_requested(request_options: LLMRequestOptions | None) -> bool:
    return bool(request_options and request_options.reasoning is False)


def _is_reasoning_on_requested(request_options: LLMRequestOptions | None) -> bool:
    return bool(request_options and request_options.reasoning is True)


def _should_apply_default_reasoning_disable(request_options: LLMRequestOptions | None) -> bool:
    return not _is_reasoning_on_requested(request_options)


def _reasoning_patch_source(request_options: LLMRequestOptions | None) -> str:
    return "explicit_false" if _is_reasoning_off_requested(request_options) else "json_default"


def _normalized_model_name(model: str | None) -> str:
    return (model or "").strip().lower()


def _supports_kimi_reasoning_disable(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return normalized.startswith("kimi-k2.5") or normalized.startswith("kimi-k2.6")


def _is_kimi_reasoning_always_on(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("kimi-k2.7-code")


def _is_moonshot_non_reasoning_family(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("moonshot-v1")


def _is_deepseek_reasoning_model(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("deepseek-v4")


def _is_openai_reasoning_model(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _is_minimax_m3_model(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("minimax-m3")


def _is_minimax_m2_model(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("minimax-m2")


def _is_spark_x1_model(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("x1")


def _supports_spark_system_messages(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return normalized.startswith(("4.0ultra", "max", "x1", "x2"))


def _supports_spark_tools(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return normalized.startswith(("4.0ultra", "max", "x1", "x2"))


def _is_siliconflow_reasoning_model(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return normalized in {
        "pro/zai-org/glm-5",
        "pro/zai-org/glm-4.7",
        "deepseek-ai/deepseek-v3.2",
        "pro/deepseek-ai/deepseek-v3.2",
        "zai-org/glm-4.6",
        "qwen/qwen3-8b",
        "qwen/qwen3-14b",
        "qwen/qwen3-32b",
        "qwen/qwen3-30b-a3b",
        "tencent/hunyuan-a13b-instruct",
        "zai-org/glm-4.5v",
        "deepseek-ai/deepseek-v3.1-terminus",
        "pro/deepseek-ai/deepseek-v3.1-terminus",
        "qwen/qwen3.5-397b-a17b",
        "qwen/qwen3.5-122b-a10b",
        "qwen/qwen3.5-35b-a3b",
        "qwen/qwen3.5-27b",
        "qwen/qwen3.5-9b",
        "qwen/qwen3.5-4b",
    }


def _is_novita_reasoning_model(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return "deepseek-r1" in normalized or normalized in {
        "deepseek/deepseek-v3.1-terminus",
        "zai-org/glm-5.1",
        "minimax/minimax-m3",
        "openai/gpt-oss-120b",
    }


def _is_hunyuan_tokenhub_reasoning_model(model: str | None) -> bool:
    return _normalized_model_name(model) == "hy3-preview"


def _is_hunyuan_legacy_reasoning_model(model: str | None) -> bool:
    return "hunyuan-a13b" in _normalized_model_name(model)


def _is_hunyuan_always_on_reasoning_model(model: str | None) -> bool:
    return _normalized_model_name(model) == "hunyuan-2.0-thinking-20251109"


def _is_stepfun_reasoning_model(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return normalized in {"step-3.5-flash", "step-3.5-flash-2603", "step-3.7-flash"}


def _is_groq_reasoning_model(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return "qwen3" in normalized


def _is_cerebras_reasoning_model(model: str | None) -> bool:
    normalized = _normalized_model_name(model)
    return "glm-4.7" in normalized or "zai-glm-4.7" in normalized


def _is_grok_reasoning_model(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("grok-4.3")


def _is_mistral_reasoning_model(model: str | None) -> bool:
    return "magistral" in _normalized_model_name(model)


def _is_mimo_reasoning_model(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("mimo-v2.5-pro")


def _is_ollama_disable_unsupported_model(model: str | None) -> bool:
    return _normalized_model_name(model).startswith("gpt-oss")


def apply_provider_response_patches(*, payload: Any, capability: ProviderCapability) -> Any:
    if not isinstance(payload, dict):
        return payload
    engine = capability.engine
    patched = _copy_jsonable(payload)
    if engine == "glm-china":
        choice = _first_choice_dict(patched)
        if choice is not None:
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                mapped = normalize_finish_reason(reason, error_shape_family=capability.error_shape_family)
                if mapped in {"content_filter", "error", "length", "tool_calls", "stop"}:
                    choice["finish_reason"] = mapped
    if engine == "glm-global":
        choice = _first_choice_dict(patched)
        if choice is not None:
            message = choice.get("message") if isinstance(choice.get("message"), dict) else None
            if message is not None:
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else None
                        if function and isinstance(function.get("arguments"), dict):
                            function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False)
                            patched.setdefault("_zoom_search_patch", {})["patched_tool_arguments"] = True
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                normalized_reason = normalize_finish_reason(reason, error_shape_family=capability.error_shape_family)
                if normalized_reason != reason:
                    patched.setdefault("_zoom_search_patch", {})["patched_finish_reason"] = True
                choice["finish_reason"] = normalized_reason
    if engine == "spark":
        if _is_spark_business_error(patched):
            patched.setdefault("spark_business_error", True)
            patched.setdefault("_zoom_search_patch", {})["business_error_detected_in_200"] = True
        choice = _first_choice_dict(patched)
        if choice is not None and not choice.get("finish_reason"):
            choice["finish_reason"] = "stop"
            patched.setdefault("_zoom_search_patch", {})["patched_finish_reason"] = True
    if engine in {"minimax-global", "minimax-china"}:
        base_resp = patched.get("base_resp") if isinstance(patched.get("base_resp"), dict) else {}
        status_code = _coerce_int(base_resp.get("status_code"))
        if status_code not in {None, 0}:
            patched.setdefault("_zoom_search_patch", {})["business_error_detected_in_200"] = True
        choice = _first_choice_dict(patched)
        if choice is not None:
            message = choice.get("message") if isinstance(choice.get("message"), dict) else None
            if message is not None:
                raw_content = _coerce_text(message.get("content"))
                cleaned_content, think_debug = _strip_think_and_json_wrappers(raw_content)
                if cleaned_content != raw_content:
                    message["content"] = cleaned_content
                    patched.setdefault("_zoom_search_patch", {}).update(think_debug)
                reasoning_details = message.get("reasoning_details")
                if reasoning_details is not None:
                    patched.setdefault("_zoom_search_patch", {})["reasoning_split_effective"] = True
    return patched


def map_provider_error_reason(
    *,
    capability_engine: str,
    http_status: int | None,
    error_code: str | None,
    error_type: str | None,
    message: str | None,
) -> tuple[str | None, bool]:
    code = (error_code or "").lower()
    err_type = (error_type or "").lower()
    text = (message or "").lower()

    if capability_engine == "gemini":
        if "resource_exhausted" in text or "rate_limit" in code:
            return "rate_limited", True
        if "invalid_argument" in text or http_status == 400:
            return "invalid_request", False
        if "safety" in text:
            return "content_filtered", False

    if capability_engine in {"doubao-global", "doubao-china"}:
        if "sensitivecontentdetected" in code or "inputtextsensitivecontentdetected" in code:
            return "content_filtered", False
        if "modelidaccessdisabled" in code or "invalidendpointormodel" in code or "modelnotopen" in code:
            return "invalid_request", False
        if "quotaexceeded" in code or "accountoverdue" in code:
            return "rate_limited", False
        if http_status == 429:
            return "rate_limited", True

    if capability_engine in {"qwen-global", "qwen-china"}:
        if "datainspectionfailed" in code:
            return "content_filtered", False
        if "quota" in text:
            return "rate_limited", False
        if http_status == 429:
            return "rate_limited", True

    if capability_engine == "glm-china":
        if code == "1301":
            return "content_filtered", False
        if code in {"1302", "1305", "1312"}:
            return "rate_limited", code == "1302"
        if code in {"1211", "1213", "1214", "1261"}:
            return "invalid_request", False

    if capability_engine == "glm-global":
        if code == "1301" or "sensitive" in text:
            return "content_filtered", False
        if code in {"1302", "1303", "1305"}:
            return "rate_limited", True
        if code in {"1210", "1211", "1213", "1214"}:
            return "invalid_request", False

    if capability_engine == "baichuan":
        if http_status == 429:
            return "rate_limited", True
        if http_status in {400, 422}:
            return "invalid_request", False

    if capability_engine == "spark":
        if code in {"10013", "10014"}:
            return "content_filtered", False
        if code in {"11201", "11202", "11203"}:
            return "rate_limited", True
        if code in {"10907", "10007"}:
            return "invalid_request", False

    if capability_engine == "huggingface":
        if code == "429" or "rate_limit" in text:
            return "rate_limited", True
        if http_status == 503:
            return "provider_server_error", True
        if http_status in {400, 422}:
            return "invalid_request", False

    if capability_engine in {"minimax-global", "minimax-china"}:
        if code in {"1002"}:
            return "rate_limited", True
        if code in {"1004", "2049"}:
            return "auth_error", False
        if code in {"1008"}:
            return "quota_exceeded", False
        if code in {"1026", "1027"}:
            return "content_filtered", False
        if code in {"1039"}:
            return "context_length_exceeded", False
        if code in {"2013"}:
            return "invalid_request", False
        if code:
            return "provider_business_error", False

    if http_status in {429, 500, 502, 503, 504}:
        return llm_http_error_reason(http_status), True
    return llm_http_error_reason(http_status), False


def sanitize_schema_for_provider(schema: Any, *, provider: str) -> Any:
    if provider != "gemini":
        return _copy_jsonable(schema)
    return _sanitize_gemini_schema(schema)


def ensure_json_keyword_hint(request: UnifiedLLMRequest) -> tuple[UnifiedLLMRequest, bool]:
    if any("json" in message.content.lower() for message in request.messages):
        return request, False
    messages = list(request.messages)
    hint = "Return valid JSON only."
    if messages and messages[0].role == "system":
        messages[0] = replace(messages[0], content=f"{messages[0].content}\n{hint}")
    else:
        messages.insert(0, UnifiedMessage(role="system", content=hint))
    return request.copy_with(messages=messages), True


def _fold_system_messages_into_first_user_message(request: UnifiedLLMRequest) -> UnifiedLLMRequest:
    system_blocks = [message.content.strip() for message in request.messages if message.role == "system" and message.content.strip()]
    if not system_blocks:
        return request
    non_system_messages = [message for message in request.messages if message.role != "system"]
    system_prefix = "\n\n".join(system_blocks)
    if not non_system_messages:
        non_system_messages = [UnifiedMessage(role="user", content=system_prefix)]
        return request.copy_with(messages=non_system_messages)
    first_message = non_system_messages[0]
    if first_message.role == "user":
        merged_content = f"System instructions:\n{system_prefix}\n\n{first_message.content}".strip()
        non_system_messages[0] = replace(first_message, content=merged_content)
    else:
        non_system_messages.insert(0, UnifiedMessage(role="user", content=f"System instructions:\n{system_prefix}"))
    return request.copy_with(messages=non_system_messages)


def _sanitize_gemini_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_sanitize_gemini_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    sanitized: dict[str, Any] = {}
    allowed = {
        "type",
        "properties",
        "items",
        "required",
        "description",
        "enum",
        "additionalProperties",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "anyOf",
        "$ref",
        "nullable",
    }
    for key, value in schema.items():
        if key in {"title", "$defs", "definitions", "patternProperties", "oneOf", "allOf", "default", "examples"}:
            continue
        if key == "properties" and isinstance(value, dict):
            sanitized[key] = {prop_key: _sanitize_gemini_schema(prop_value) for prop_key, prop_value in value.items()}
            continue
        if key not in allowed:
            continue
        if key == "additionalProperties" and value is not False:
            continue
        sanitized[key] = _sanitize_gemini_schema(value)
    return sanitized


def _drop_extra_param(request: UnifiedLLMRequest, key: str) -> UnifiedLLMRequest:
    if key not in request.extra_params:
        return request
    updated = dict(request.extra_params)
    updated.pop(key, None)
    return request.copy_with(extra_params=updated)


def _set_extra_param(request: UnifiedLLMRequest, key: str, value: Any) -> UnifiedLLMRequest:
    updated = dict(request.extra_params)
    updated[key] = value
    return request.copy_with(extra_params=updated)


def _copy_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_jsonable(item) for item in value]
    return value


def _first_choice_dict(payload: dict[str, Any]) -> dict[str, Any] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return choice if isinstance(choice, dict) else None


def normalize_llm_exception(*, error: Exception, context: RuntimeContext) -> ZoomSearchError:
    if isinstance(error, ZoomSearchError):
        return error
    if _is_transport_exception(error):
        return normalize_transport_error(
            error=error,
            component="llm",
            request_id=context.request_id,
            provider_engine=context.llm_provider.engine,
            provider_model=context.llm_provider.model,
        )
    return call_failure(
        category="llm_call_failure",
        component="llm",
        message=str(error) or error.__class__.__name__,
        user_message="LLM request failed.",
        request_id=context.request_id,
        provider_engine=context.llm_provider.engine,
        provider_model=context.llm_provider.model,
        reason_code="provider_call_failed",
        raw_diagnostics=RawDiagnostics(
            transport_exception_class=error.__class__.__name__,
            transport_exception_message=str(error) or error.__class__.__name__,
        ),
    )


def _is_transport_exception(error: Exception) -> bool:
    module = error.__class__.__module__
    name = error.__class__.__name__.lower()
    if module.startswith("httpx"):
        return any(marker in name for marker in ("timeout", "network", "connect", "proxy", "transport"))
    return any(marker in name for marker in ("timeout", "connecterror", "networkerror", "proxyerror"))


def _read_first_message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                return message
    return {}


def _read_finish_reason(payload: dict[str, Any]) -> Any:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            return choice.get("finish_reason")
    if isinstance(payload.get("stop_reason"), str):
        return payload.get("stop_reason")
    if isinstance(payload.get("status"), str):
        return payload.get("status")
    return None


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _try_parse_json_payload(text: str) -> dict[str, Any] | list[Any] | None:
    text = _clean_json_candidate(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for candidate in _extract_json_candidates(text):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, (dict, list)):
                return payload
        return None
    if isinstance(payload, (dict, list)):
        return payload
    return None


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    parsed = _try_parse_json_payload(text)
    return parsed if isinstance(parsed, dict) else None


def _is_auto_tool_choice(value: str | dict[str, Any] | None) -> bool:
    return value is None or value == "auto"


def _collect_provider_metadata(*, payload: dict[str, Any], message: dict[str, Any], capability: ProviderCapability) -> dict[str, Any]:
    patch_state = payload.get("_zoom_search_patch") if isinstance(payload.get("_zoom_search_patch"), dict) else {}
    metadata = {
        "citations": payload.get("citations") if isinstance(payload.get("citations"), list) else (message.get("citations") if isinstance(message, dict) and isinstance(message.get("citations"), list) else None),
        "reasoning_content": message.get("reasoning_content") if isinstance(message, dict) else None,
        "reasoning_details": message.get("reasoning_details") if isinstance(message, dict) else None,
        "provider_request_id": payload.get("request_id") or payload.get("sid"),
        "patched_finish_reason": bool(patch_state.get("patched_finish_reason")),
        "patched_tool_arguments": bool(patch_state.get("patched_tool_arguments")),
        "business_error_detected_in_200": bool(patch_state.get("business_error_detected_in_200")),
        "think_tags_detected": bool(patch_state.get("think_tags_detected")),
        "think_tags_stripped": bool(patch_state.get("think_tags_stripped")),
        "reasoning_split_requested": bool(patch_state.get("reasoning_split_requested")),
        "reasoning_split_effective": bool(patch_state.get("reasoning_split_effective")),
        "json_candidate_extracted": bool(patch_state.get("json_candidate_extracted")),
        "route": patch_state.get("route"),
        "structured_output_source": patch_state.get("structured_output_source"),
        "base_resp": payload.get("base_resp") if isinstance(payload.get("base_resp"), dict) else None,
    }
    if capability.engine == "spark":
        metadata["spark_code"] = payload.get("code")
    return metadata


def _clean_json_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    if stripped.lower().startswith("json\n"):
        stripped = stripped[5:].strip()
    return stripped


def _select_minimax_china_route(*, request: UnifiedLLMRequest, response_format: dict[str, Any] | None) -> str:
    extra = request.extra_params
    if request.stream:
        return "compat"
    if request.tools:
        return "compat"
    if extra.get("structured_output_route") == "native_chatcompletion_v2":
        return "native"
    if request.task == "query_rewriting" and extra.get("prefer_schema_native", True):
        return "native"
    if request.expect_json and response_format and response_format.get("type") == "json_schema":
        return "native"
    return "compat"


def _strip_think_and_json_wrappers(text: str | None) -> tuple[str | None, dict[str, Any]]:
    if text is None:
        return None, {}
    debug: dict[str, Any] = {"reasoning_split_requested": True}
    cleaned = text
    if "<think>" in cleaned:
        debug["think_tags_detected"] = True
        without_balanced = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        if without_balanced != cleaned:
            cleaned = without_balanced
            debug["think_tags_stripped"] = True
        if "<think>" in cleaned:
            cleaned = cleaned.split("<think>", 1)[0]
            debug["think_tags_stripped"] = True
    before_wrapper_cleanup = cleaned
    cleaned = _clean_json_candidate(cleaned)
    cleaned = re.sub(r"^Here is the JSON:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^JSON:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    if cleaned != before_wrapper_cleanup:
        debug["json_candidate_extracted"] = True
    return cleaned or None, debug


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1].strip())
    return candidates


def _patch_huggingface_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_patch_huggingface_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    patched: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, list):
            patched["anyOf"] = [{"type": item} for item in value]
            continue
        patched[key] = _patch_huggingface_schema(value)
    return patched


def _is_spark_business_error(payload: dict[str, Any]) -> bool:
    code = payload.get("code")
    if code is None:
        return False
    try:
        return int(code) != 0
    except (TypeError, ValueError):
        return False
