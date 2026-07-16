from __future__ import annotations

import pytest

from zoom_search.models import LLMConfig
from zoom_search.models import LLMRequestOptions
from zoom_search.models import ResolvedProvider
from zoom_search.models import RuntimeContext
from zoom_search.models import SearchConfig
from zoom_search.models import SearchRequest
from zoom_search.models import UnifiedLLMRequest
from zoom_search.models import UnifiedMessage
from zoom_search.models import UnifiedToolDefinition
from zoom_search.models import WarningInfo
from zoom_search.models import ZoomSearchError
from zoom_search.providers.capabilities import BUILTIN_LLM_CAPABILITIES
from zoom_search.providers.capabilities import BUILTIN_SEARCH_CAPABILITIES
from zoom_search.providers.llm import ClaudeAdapter
from zoom_search.providers.llm import CustomNativeLLMAdapter
from zoom_search.providers.llm import OpenAICompatibleAdapter
from zoom_search.providers.llm import ReplicateAdapter
from zoom_search.providers.llm import create_llm_provider


class MockResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self._payload


class NonJsonMockResponse(MockResponse):
    def __init__(self, text: str, status_code: int) -> None:
        super().__init__({}, status_code=status_code)
        self.text = text

    def json(self) -> object:
        raise ValueError("invalid json")


class MockStreamResponse:
    def __init__(self, lines: list[str], *, status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class MockLLMClient:
    def __init__(self, *, post_responses: list[MockResponse] | None = None, get_responses: list[MockResponse] | None = None, stream_lines: list[str] | None = None, stream_status_code: int = 200) -> None:
        self.post_responses = list(post_responses or [])
        self.get_responses = list(get_responses or [])
        self.stream_lines = list(stream_lines or [])
        self.stream_status_code = stream_status_code
        self.posts: list[dict] = []
        self.gets: list[dict] = []
        self.stream_calls: list[dict] = []

    async def post(self, path: str, *, json: dict, headers: dict[str, str]) -> MockResponse:
        self.posts.append({"path": path, "json": json, "headers": headers})
        return self.post_responses.pop(0)

    async def get(self, url: str, *, headers: dict[str, str]) -> MockResponse:
        self.gets.append({"url": url, "headers": headers})
        return self.get_responses.pop(0)

    def stream(self, method: str, url: str, **kwargs) -> MockStreamResponse:
        self.stream_calls.append({"method": method, "url": url, **kwargs})
        return MockStreamResponse(self.stream_lines, status_code=self.stream_status_code)


class MockTransportContext:
    def __init__(self, llm_client: MockLLMClient) -> None:
        self.llm_client = llm_client


def _build_context(
    engine: str,
    model: str,
    *,
    llm_headers: dict[str, str] | None = None,
    llm_extra: dict | None = None,
    request_options: LLMRequestOptions | None = None,
) -> RuntimeContext:
    capability = BUILTIN_LLM_CAPABILITIES.get(engine)
    if capability is None:
        from zoom_search.models import ProviderCapability

        capability = ProviderCapability(
            engine=engine,
            provider_kind="custom",
            protocol="native",
            adapter_class="custom_native",
            structured_output_level="native_json_object_only",
            supports_seed="supported",
        )
    return RuntimeContext(
        request_id="req_test",
        request=SearchRequest(
            question="Test question",
            llm=LLMConfig(engine=engine, model=model, api_key="secret", request_options=request_options or LLMRequestOptions()),
            search=SearchConfig(engine="tavily", api_key="search-secret"),
        ),
        llm_provider=ResolvedProvider(
            engine=engine,
            provider_kind="builtin",
            component="llm",
            capability=capability,
            model=model,
            protocol="native",
            base_url=capability.default_base_url,
            api_key="secret",
            headers=dict(llm_headers or {}),
            request_options=request_options or LLMRequestOptions(),
            extra=dict(llm_extra or {}),
        ),
        search_provider=ResolvedProvider(
            engine="tavily",
            provider_kind="builtin",
            component="search",
            capability=BUILTIN_SEARCH_CAPABILITIES["tavily"],
            base_url=BUILTIN_SEARCH_CAPABILITIES["tavily"].default_base_url,
            api_key="search-secret",
        ),
        warnings=[],
        retry_budget={},
        transport_context={},
        semaphore_limits={},
    )


def test_create_llm_provider_routes_to_dedicated_adapters() -> None:
    claude = create_llm_provider(
        context=_build_context("claude", "claude-sonnet-4-5"),
        transport_context=MockTransportContext(MockLLMClient()),
    )
    replicate = create_llm_provider(
        context=_build_context("replicate", "meta/meta-llama-3-70b-instruct"),
        transport_context=MockTransportContext(MockLLMClient()),
    )

    assert isinstance(claude, ClaudeAdapter)
    assert isinstance(replicate, ReplicateAdapter)


def test_create_llm_provider_routes_openai_compatible_to_shared_adapter() -> None:
    provider = create_llm_provider(
        context=_build_context("gemini", "gemini-2.5-flash"),
        transport_context=MockTransportContext(MockLLMClient()),
    )

    assert isinstance(provider, OpenAICompatibleAdapter)


def test_create_llm_provider_routes_custom_native_to_native_adapter() -> None:
    context = _build_context(
        "custom",
        "native-model",
        llm_extra={"response_mapping": {"content_path": "data.answer"}},
    )
    context.llm_provider.provider_kind = "custom"
    context.llm_provider.protocol = "native"

    provider = create_llm_provider(
        context=context,
        transport_context=MockTransportContext(MockLLMClient()),
    )

    assert isinstance(provider, CustomNativeLLMAdapter)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_factory", "engine", "model", "expected_status"),
    [
        (OpenAICompatibleAdapter, "gemini", "gemini-2.5-flash", 429),
        (CustomNativeLLMAdapter, "custom", "native-model", 503),
        (ClaudeAdapter, "claude", "claude-sonnet-4-5", 500),
    ],
)
async def test_llm_adapters_preserve_http_status_for_non_json_error_body(adapter_factory, engine: str, model: str, expected_status: int) -> None:
    context = _build_context(engine, model, llm_extra={"response_mapping": {"content_path": "data.answer"}})
    if engine == "custom":
        context.llm_provider.provider_kind = "custom"
        context.llm_provider.protocol = "native"
    client = MockLLMClient(post_responses=[NonJsonMockResponse("upstream unavailable", expected_status)])
    adapter = adapter_factory(context=context, transport_context=MockTransportContext(client))

    with pytest.raises(ZoomSearchError) as caught:
        await adapter.generate(
            UnifiedLLMRequest(
                task="answer_synthesis",
                provider=engine,
                model=model,
                messages=[UnifiedMessage(role="user", content="answer")],
            )
        )

    assert caught.value.details.http_status == expected_status
    assert caught.value.raw_diagnostics is not None
    assert caught.value.raw_diagnostics.provider_error_body == {"error": {"message": "upstream unavailable"}}


@pytest.mark.asyncio
async def test_replicate_poll_http_error_preserves_status_for_non_json_body() -> None:
    context = _build_context("replicate", "owner/model")
    client = MockLLMClient(
        post_responses=[MockResponse({"status": "processing", "urls": {"get": "https://replicate.example/predictions/1"}})],
        get_responses=[NonJsonMockResponse("service down", 503)],
    )
    adapter = ReplicateAdapter(context=context, transport_context=MockTransportContext(client))

    with pytest.raises(ZoomSearchError) as caught:
        await adapter.generate(
            UnifiedLLMRequest(
                task="answer_synthesis",
                provider="replicate",
                model="owner/model",
                messages=[UnifiedMessage(role="user", content="answer")],
                request_timeout_seconds=1,
                extra_params={"poll_interval_seconds": 1},
            )
        )

    assert caught.value.details.http_status == 503
    assert caught.value.details.reason_code == "provider_server_error"


@pytest.mark.asyncio
async def test_custom_native_adapter_uses_extra_as_body_and_mapping_for_unified_fields() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "data": {"output": {"text": '{"ok": true}'}},
                    "usage": {"total": 9},
                    "state": {"finish": "stop"},
                }
            )
        ]
    )
    context = _build_context(
        "custom",
        "native-model",
        llm_headers={"x-api-key": "native-secret"},
        llm_extra={
            "endpoint_path": "/v1/native-generate",
            "apiKey": "body-secret",
            "temperature": 0.8,
            "providerOptions": {"reasoning": False},
            "request_mapping": {
                "model_path": "payload.modelName",
                "messages_path": "payload.chatMessages",
                "temperature_path": "payload.temperature",
                "seed_path": "payload.seed",
                "response_format_path": "payload.responseFormat",
            },
            "response_mapping": {
                "content_path": "data.output.text",
                "usage_total_tokens_path": "usage.total",
                "finish_reason_path": "state.finish",
            },
        },
    )
    context.llm_provider.provider_kind = "custom"
    context.llm_provider.protocol = "native"
    adapter = CustomNativeLLMAdapter(context=context, transport_context=MockTransportContext(client))

    normalized = adapter.normalize_request(
        UnifiedLLMRequest(
            task="query_rewriting",
            provider="custom",
            model="native-model",
            messages=[UnifiedMessage(role="user", content="Return JSON")],
            temperature=0.2,
            expect_json=True,
            json_object=True,
            seed=7,
        )
    )
    assert normalized.response_format == {"type": "json_object"}

    response = await adapter.generate(
        UnifiedLLMRequest(
            task="query_rewriting",
            provider="custom",
            model="native-model",
            messages=[UnifiedMessage(role="user", content="Return JSON")],
            temperature=0.2,
            expect_json=True,
            json_object=True,
            seed=7,
        )
    )

    assert client.posts[0]["path"] == "/v1/native-generate"
    assert client.posts[0]["headers"]["x-api-key"] == "native-secret"
    assert client.posts[0]["headers"]["Authorization"] == "Bearer secret"
    payload = client.posts[0]["json"]
    assert payload["apiKey"] == "body-secret"
    assert payload["providerOptions"] == {"reasoning": False}
    assert payload["payload"]["modelName"] == "native-model"
    assert payload["payload"]["chatMessages"] == [{"role": "user", "content": "Return JSON"}]
    assert payload["payload"]["temperature"] == 0.2
    assert payload["payload"]["seed"] == 7
    assert payload["payload"]["responseFormat"] == {"type": "json_object"}
    assert "endpoint_path" not in payload
    assert "request_mapping" not in payload
    assert "response_mapping" not in payload
    assert response.json_payload == {"ok": True}
    assert response.usage["total_tokens"] == 9


@pytest.mark.asyncio
async def test_custom_native_adapter_defaults_to_model_and_messages_only() -> None:
    client = MockLLMClient(post_responses=[MockResponse({"data": {"answer": "ok"}})])
    context = _build_context(
        "custom",
        "native-model",
        llm_extra={"response_mapping": {"content_path": "data.answer"}, "temperature": 0.8},
    )
    context.llm_provider.provider_kind = "custom"
    context.llm_provider.protocol = "native"
    adapter = CustomNativeLLMAdapter(context=context, transport_context=MockTransportContext(client))

    await adapter.generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            provider="custom",
            model="native-model",
            messages=[UnifiedMessage(role="user", content="Answer")],
            temperature=0.2,
            seed=7,
        )
    )

    assert client.posts[0]["json"] == {
        "temperature": 0.8,
        "model": "native-model",
        "messages": [{"role": "user", "content": "Answer"}],
    }


def test_create_llm_provider_routes_new_openai_compatible_engines_to_shared_adapter() -> None:
    engines = [
        "deepseek",
        "kimi-china",
        "kimi-global",
        "yi",
        "hunyuan",
        "stepfun",
        "siliconflow",
        "together",
        "fireworks",
        "groq",
        "cerebras",
        "perplexity",
        "grok",
        "mistral",
        "cohere",
        "openrouter",
        "mimo",
        "deepinfra",
        "novita",
        "hyperbolic",
        "lepton",
        "ollama",
    ]

    for engine in engines:
        provider = create_llm_provider(
            context=_build_context(engine, "model-name"),
            transport_context=MockTransportContext(MockLLMClient()),
        )
        assert isinstance(provider, OpenAICompatibleAdapter)


@pytest.mark.asyncio
async def test_openai_compatible_adapter_generates_json_payload() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "choices": [
                        {
                            "message": {"content": '{"search_groups": [], "previous_conversation": []}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
                }
            )
        ]
    )
    adapter = OpenAICompatibleAdapter(
        context=_build_context("gemini", "gemini-2.5-flash"),
        transport_context=MockTransportContext(client),
    )
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[UnifiedMessage(role="user", content="Return JSON")],
        expect_json=True,
        json_schema={"type": "object", "properties": {"search_groups": {"type": "array"}}},
    )

    response = await adapter.generate(request)

    assert response.json_payload == {"search_groups": [], "previous_conversation": []}
    assert response.finish_reason == "stop"
    assert response.usage["total_tokens"] == 17
    assert client.posts[0]["path"] == "/chat/completions"
    assert client.posts[0]["headers"]["Authorization"] == "Bearer secret"
    assert client.posts[0]["json"]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_openai_compatible_adapter_passes_llm_extra_to_body_and_headers_to_headers() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                }
            )
        ]
    )
    adapter = OpenAICompatibleAdapter(
        context=_build_context(
            "gemini",
            "gemini-2.5-flash",
            llm_headers={"X-Tenant-ID": "tenant-a"},
            llm_extra={
                "logprobs": True,
                "reasoning_effort": "low",
                "temperature": 0.9,
                "top_logprobs": 2,
            },
        ),
        transport_context=MockTransportContext(client),
    )
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[UnifiedMessage(role="user", content="Answer")],
        temperature=0.2,
    )

    await adapter.generate(request)

    assert client.posts[0]["headers"]["X-Tenant-ID"] == "tenant-a"
    assert client.posts[0]["headers"]["Authorization"] == "Bearer secret"
    assert client.posts[0]["json"]["reasoning_effort"] == "low"
    assert client.posts[0]["json"]["temperature"] == 0.2
    assert "logprobs" not in client.posts[0]["json"]
    assert "top_logprobs" not in client.posts[0]["json"]


@pytest.mark.asyncio
async def test_openai_compatible_adapter_applies_provider_patch_fields_from_llm_extra() -> None:
    cases = [
        (
            "doubao-global",
            "doubao-seed-1-6",
            {"endpoint_id": "ep-123", "supports_streaming": True},
            {"model": "ep-123"},
        ),
        (
            "baichuan",
            "Baichuan4-Turbo",
            {"top_k": 9, "with_search_enhance": True},
            {"top_k": 9, "with_search_enhance": True},
        ),
    ]

    for engine, model, llm_extra, expected_body in cases:
        client = MockLLMClient(
            post_responses=[MockResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})]
        )
        adapter = OpenAICompatibleAdapter(
            context=_build_context(engine, model, llm_extra=llm_extra),
            transport_context=MockTransportContext(client),
        )

        await adapter.generate(
            UnifiedLLMRequest(
                task="answer_synthesis",
                provider=engine,
                model=model,
                messages=[UnifiedMessage(role="user", content="Answer")],
            )
        )

        for key, value in expected_body.items():
            assert client.posts[0]["json"][key] == value
        assert "endpoint_id" not in client.posts[0]["json"]
        assert "supports_streaming" not in client.posts[0]["json"]


@pytest.mark.asyncio
async def test_openai_compatible_adapter_uses_minimax_control_fields_without_sending_them() -> None:
    client = MockLLMClient(
        post_responses=[MockResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})]
    )
    adapter = OpenAICompatibleAdapter(
        context=_build_context(
            "minimax-china",
            "MiniMax-M2.1",
            llm_extra={
                "structured_output_route": "native_chatcompletion_v2",
                "structured_output_model": "MiniMax-Text-01",
            },
        ),
        transport_context=MockTransportContext(client),
    )

    await adapter.generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            provider="minimax-china",
            model="MiniMax-M2.1",
            messages=[UnifiedMessage(role="user", content="Answer")],
        )
    )

    assert client.posts[0]["path"] == "/v1/text/chatcompletion_v2"
    assert client.posts[0]["json"]["model"] == "MiniMax-Text-01"
    for control_field in ("api_route", "structured_output_model", "structured_output_route"):
        assert control_field not in client.posts[0]["json"]


@pytest.mark.asyncio
async def test_custom_openai_compatible_adapter_preserves_provider_body_fields() -> None:
    client = MockLLMClient(
        post_responses=[MockResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})]
    )
    adapter = OpenAICompatibleAdapter(
        context=_build_context(
            "openai-compatible",
            "custom-model",
            llm_extra={
                "model_id": "provider-model-id",
                "structured_output_model": "provider-structured-model",
                "structured_output_route": "provider-route",
                "supports_streaming": True,
            },
        ),
        transport_context=MockTransportContext(client),
    )

    await adapter.generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            provider="openai-compatible",
            model="custom-model",
            messages=[UnifiedMessage(role="user", content="Answer")],
        )
    )

    body = client.posts[0]["json"]
    assert body["model"] == "custom-model"
    assert body["model_id"] == "provider-model-id"
    assert body["structured_output_model"] == "provider-structured-model"
    assert body["structured_output_route"] == "provider-route"
    assert "supports_streaming" not in body


@pytest.mark.asyncio
async def test_openai_compatible_adapter_omits_temperature_for_kimi_providers() -> None:
    for engine, model in (("kimi-global", "kimi-k2.6"), ("kimi-china", "moonshot-v1-8k")):
        client = MockLLMClient(
            post_responses=[
                MockResponse(
                    {
                        "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}],
                    }
                )
            ]
        )
        adapter = OpenAICompatibleAdapter(
            context=_build_context(
                engine,
                model,
                llm_extra={"temperature": 0.9, "reasoning_effort": "low"},
            ),
            transport_context=MockTransportContext(client),
        )
        request = UnifiedLLMRequest(
            task="answer_synthesis",
            provider=engine,
            model=model,
            messages=[UnifiedMessage(role="user", content="Answer")],
            temperature=0.2,
        )

        await adapter.generate(request)

        assert "temperature" not in client.posts[0]["json"]
        assert client.posts[0]["json"]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_openai_compatible_adapter_honors_disabled_request_options() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}],
                }
            )
        ]
    )
    adapter = OpenAICompatibleAdapter(
        context=_build_context(
            "deepseek",
            "deepseek-chat",
            request_options=LLMRequestOptions(
                temperature=False,
                response_format=False,
                seed=False,
                stream=False,
                reasoning=False,
            ),
        ),
        transport_context=MockTransportContext(client),
    )

    await adapter.generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            provider="deepseek",
            model="deepseek-chat",
            messages=[UnifiedMessage(role="user", content="Answer")],
            temperature=0.2,
            expect_json=True,
            json_schema={"type": "object"},
            seed=7,
            stream=True,
        )
    )

    payload = client.posts[0]["json"]
    assert "temperature" not in payload
    assert "response_format" not in payload
    assert "seed" not in payload
    assert "stream" not in payload


@pytest.mark.asyncio
async def test_openai_compatible_adapter_maps_reasoning_false_for_openai_reasoning_models() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                }
            )
        ]
    )
    adapter = OpenAICompatibleAdapter(
        context=_build_context(
            "openai",
            "gpt-5-mini",
            request_options=LLMRequestOptions(reasoning=False),
        ),
        transport_context=MockTransportContext(client),
    )

    await adapter.generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            provider="openai",
            model="gpt-5-mini",
            messages=[UnifiedMessage(role="user", content="Answer")],
        )
    )

    assert client.posts[0]["json"]["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_openai_compatible_adapter_maps_reasoning_false_for_together() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                }
            )
        ]
    )
    adapter = OpenAICompatibleAdapter(
        context=_build_context(
            "together",
            "zai-org/GLM-5.2",
            request_options=LLMRequestOptions(reasoning=False),
        ),
        transport_context=MockTransportContext(client),
    )

    await adapter.generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            provider="together",
            model="zai-org/GLM-5.2",
            messages=[UnifiedMessage(role="user", content="Answer")],
        )
    )

    assert client.posts[0]["json"]["reasoning"] == {"enabled": False}


@pytest.mark.asyncio
async def test_openai_compatible_adapter_maps_reasoning_false_for_additional_providers() -> None:
    cases = [
        ("fireworks", "accounts/fireworks/models/deepseek-r1", "reasoning", {"effort": "none"}),
        ("cohere", "command-a-thinking", "thinking", {"type": "disabled"}),
        ("siliconflow", "Qwen/Qwen3-32B", "enable_thinking", False),
        ("deepinfra", "deepseek-ai/DeepSeek-R1", "reasoning", {"enabled": False}),
        ("ollama", "llama3.2", "think", False),
    ]

    for engine, model, field_name, expected in cases:
        client = MockLLMClient(
            post_responses=[
                MockResponse(
                    {
                        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    }
                )
            ]
        )
        adapter = OpenAICompatibleAdapter(
            context=_build_context(
                engine,
                model,
                request_options=LLMRequestOptions(reasoning=False),
            ),
            transport_context=MockTransportContext(client),
        )

        await adapter.generate(
            UnifiedLLMRequest(
                task="answer_synthesis",
                provider=engine,
                model=model,
                messages=[UnifiedMessage(role="user", content="Answer")],
            )
        )

        assert client.posts[0]["json"][field_name] == expected


def test_claude_system_message_mapping_and_json_schema_route() -> None:
    adapter = ClaudeAdapter(
        context=_build_context("claude", "claude-sonnet-4-5"),
        transport_context=MockTransportContext(MockLLMClient()),
    )
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="claude",
        model="claude-sonnet-4-5",
        messages=[
            UnifiedMessage(role="system", content="System A"),
            UnifiedMessage(role="system", content="System B"),
            UnifiedMessage(role="user", content="Return JSON"),
        ],
        expect_json=True,
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
    )

    normalized = adapter.normalize_request(request)
    payload = adapter.build_provider_payload(normalized.request, normalized.response_format)

    assert payload["system"] == "System A\n\nSystem B"
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Return JSON"}]}]
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_claude_structured_output_response_parsing() -> None:
    adapter = ClaudeAdapter(
        context=_build_context("claude", "claude-sonnet-4-5"),
        transport_context=MockTransportContext(MockLLMClient()),
    )
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="claude",
        model="claude-sonnet-4-5",
        messages=[UnifiedMessage(role="user", content="Return JSON")],
        expect_json=True,
    )

    response = adapter.parse_provider_response(
        payload={
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3},
        },
        request=request,
        normalized_debug={},
    )

    assert response.json_payload == {"ok": True}
    assert response.finish_reason == "stop"
    assert response.usage["total_tokens"] == 15
    assert response.usage["cached_input_tokens"] == 3


def test_claude_tool_calling_mapping_and_response_normalization() -> None:
    adapter = ClaudeAdapter(
        context=_build_context("claude", "claude-sonnet-4-5"),
        transport_context=MockTransportContext(MockLLMClient()),
    )
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="claude",
        model="claude-sonnet-4-5",
        messages=[
            UnifiedMessage(
                role="assistant",
                content="",
                metadata={
                    "tool_calls": [
                        {"id": "call_1", "name": "lookup", "arguments": {"q": "hotel"}},
                    ]
                },
            ),
            UnifiedMessage(
                role="user",
                content="tool output",
                tool_call_id="call_1",
                metadata={"tool_result": "tool output", "text": "Use this result"},
            ),
        ],
        tools=[UnifiedToolDefinition(name="lookup", parameters={"type": "object"})],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )

    normalized = adapter.normalize_request(request)
    payload = adapter.build_provider_payload(normalized.request, normalized.response_format)

    assert payload["tools"][0]["input_schema"] == {"type": "object"}
    assert payload["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert payload["messages"][0]["content"][0]["type"] == "tool_use"
    assert payload["messages"][1]["content"][0]["type"] == "tool_result"
    assert payload["messages"][1]["content"][1]["type"] == "text"

    parsed = adapter.parse_provider_response(
        payload={
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "hotel"}},
            ],
            "stop_reason": "tool_use",
        },
        request=request,
        normalized_debug={},
    )

    assert parsed.finish_reason == "tool_calls"
    assert parsed.tool_calls[0].arguments_object == {"q": "hotel"}


def test_claude_streaming_event_normalization() -> None:
    adapter = ClaudeAdapter(
        context=_build_context("claude", "claude-sonnet-4-5"),
        transport_context=MockTransportContext(MockLLMClient()),
    )

    assert adapter.normalize_stream_event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}}) == {
        "event": "token",
        "delta_text": "Hel",
    }
    assert adapter.normalize_stream_event(
        {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "call_1", "name": "lookup"},
        }
    ) == {"event": "tool_call_start", "tool_call_id": "call_1", "tool_name": "lookup"}
    assert adapter.normalize_stream_event({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}) == {
        "event": "message_delta",
        "finish_reason": "length",
    }


@pytest.mark.asyncio
async def test_replicate_prediction_create_and_poll() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "id": "pred_1",
                    "status": "starting",
                    "output": None,
                    "urls": {"get": "https://api.replicate.com/v1/predictions/pred_1", "stream": "https://stream.example/pred_1"},
                }
            )
        ],
        get_responses=[
            MockResponse(
                {
                    "id": "pred_1",
                    "status": "succeeded",
                    "output": ["{\"ok\":", " true}"],
                    "urls": {"get": "https://api.replicate.com/v1/predictions/pred_1", "stream": "https://stream.example/pred_1"},
                }
            )
        ],
    )
    adapter = ReplicateAdapter(
        context=_build_context("replicate", "meta/meta-llama-3-70b-instruct"),
        transport_context=MockTransportContext(client),
    )
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="replicate",
        model="meta/meta-llama-3-70b-instruct",
        messages=[UnifiedMessage(role="system", content="Return JSON only"), UnifiedMessage(role="user", content="Return JSON")],
        expect_json=True,
    )

    response = await adapter.generate(request)

    assert client.posts[0]["path"] == "/models/meta/meta-llama-3-70b-instruct/predictions"
    assert client.posts[0]["headers"]["Prefer"] == "wait"
    assert client.posts[0]["json"]["input"]["system_prompt"] == "Return JSON only"
    assert "USER: Return JSON" in client.posts[0]["json"]["input"]["prompt"]
    assert client.gets[0]["url"] == "https://api.replicate.com/v1/predictions/pred_1"
    assert response.text == '{"ok": true}'
    assert response.json_payload == {"ok": True}
    assert response.usage["total_tokens"] is None


@pytest.mark.asyncio
async def test_claude_stream_generate_uses_native_sse_protocol() -> None:
    client = MockLLMClient(
        stream_lines=[
            'event: message_start',
            'data: {"message": {"usage": {"input_tokens": 10, "output_tokens": 0}}}',
            'event: content_block_delta',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "{\\"ok\\":"}}',
            'event: content_block_delta',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " true}"}}',
            'event: message_delta',
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 10, "output_tokens": 5}}',
            'event: message_stop',
            'data: {"type": "message_stop"}',
        ]
    )
    adapter = ClaudeAdapter(
        context=_build_context("claude", "claude-sonnet-4-5"),
        transport_context=MockTransportContext(client),
    )

    events = []
    async for event in adapter.stream_generate(
        UnifiedLLMRequest(
            task="query_rewriting",
            provider="claude",
            model="claude-sonnet-4-5",
            messages=[UnifiedMessage(role="user", content="Return JSON")],
            expect_json=True,
            stream=True,
        )
    ):
        events.append(event)

    assert client.stream_calls[0]["method"] == "POST"
    assert client.stream_calls[0]["url"] == "/v1/messages"
    assert events[0] == {"event": "token", "delta_text": '{"ok":'}
    assert events[1] == {"event": "token", "delta_text": " true}"}
    assert events[-1]["event"] == "done"
    assert events[-1]["response"].json_payload == {"ok": True}
    assert events[-1]["response"].usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_replicate_stream_generate_uses_prediction_stream_url() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "id": "pred_stream",
                    "status": "starting",
                    "urls": {"stream": "https://stream.example/pred_stream"},
                }
            )
        ],
        stream_lines=[
            "event: output",
            'data: {"ok":',
            'data: true}',
            "event: done",
        ],
    )
    adapter = ReplicateAdapter(
        context=_build_context("replicate", "meta/meta-llama-3-70b-instruct"),
        transport_context=MockTransportContext(client),
    )

    events = []
    async for event in adapter.stream_generate(
        UnifiedLLMRequest(
            task="query_rewriting",
            provider="replicate",
            model="meta/meta-llama-3-70b-instruct",
            messages=[UnifiedMessage(role="user", content="Return JSON")],
            expect_json=True,
            stream=True,
        )
    ):
        events.append(event)

    assert client.stream_calls[0]["method"] == "GET"
    assert client.stream_calls[0]["url"] == "https://stream.example/pred_stream"
    assert events[1] == {"event": "token", "delta_text": '{"ok":'}
    assert events[2] == {"event": "token", "delta_text": "true}"}
    assert events[-1]["response"].json_payload == {"ok": True}


@pytest.mark.asyncio
async def test_replicate_polling_waits_for_later_completion() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "id": "pred_slow",
                    "status": "starting",
                    "output": None,
                    "urls": {"get": "https://api.replicate.com/v1/predictions/pred_slow"},
                }
            )
        ],
        get_responses=[
            MockResponse({"id": "pred_slow", "status": "processing", "output": None, "urls": {"get": "https://api.replicate.com/v1/predictions/pred_slow"}}),
            MockResponse({"id": "pred_slow", "status": "succeeded", "output": ["done"]}),
        ],
    )
    adapter = ReplicateAdapter(
        context=_build_context("replicate", "meta/meta-llama-3-70b-instruct"),
        transport_context=MockTransportContext(client),
    )

    response = await adapter.generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            provider="replicate",
            model="meta/meta-llama-3-70b-instruct",
            messages=[UnifiedMessage(role="user", content="Answer")],
            extra_params={"poll_interval_seconds": 0},
            request_timeout_seconds=2,
        )
    )

    assert len(client.gets) == 2
    assert response.text == "done"


@pytest.mark.asyncio
async def test_replicate_stream_url_normalization() -> None:
    client = MockLLMClient(
        post_responses=[
            MockResponse(
                {
                    "id": "pred_2",
                    "status": "starting",
                    "urls": {"get": "https://api.replicate.com/v1/predictions/pred_2", "stream": "https://stream.example/pred_2"},
                }
            )
        ],
        stream_lines=["event: output", "data: Hel", "data: lo", "event: done"],
    )
    adapter = ReplicateAdapter(
        context=_build_context("replicate", "meta/meta-llama-3-70b-instruct"),
        transport_context=MockTransportContext(client),
    )
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="replicate",
        model="meta/meta-llama-3-70b-instruct",
        messages=[UnifiedMessage(role="user", content="hello")],
        stream=True,
    )

    events = await adapter.stream(request)

    assert client.stream_calls[0]["url"] == "https://stream.example/pred_2"
    assert events == [
        {"event": "output"},
        {"event": "token", "delta_text": "Hel"},
        {"event": "token", "delta_text": "lo"},
        {"event": "done"},
    ]


def test_replicate_json_extraction_and_fallback() -> None:
    adapter = ReplicateAdapter(
        context=_build_context("replicate", "meta/meta-llama-3-70b-instruct"),
        transport_context=MockTransportContext(MockLLMClient()),
    )
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="replicate",
        model="meta/meta-llama-3-70b-instruct",
        messages=[UnifiedMessage(role="user", content="Return JSON")],
        expect_json=True,
    )

    parsed = adapter.parse_prediction_response(
        prediction={"id": "pred_3", "status": "succeeded", "output": ["prefix ", '{"ok": true}', " suffix"]},
        request=request,
        normalized_debug={},
    )
    fallback = adapter.parse_prediction_response(
        prediction={"id": "pred_4", "status": "succeeded", "output": ["plain text only"]},
        request=request,
        normalized_debug={},
    )

    assert parsed.json_payload == {"ok": True}
    assert fallback.json_payload is None
    assert "structured_output_parse_failed" in fallback.warnings


def test_dedicated_adapter_error_normalization() -> None:
    claude = ClaudeAdapter(
        context=_build_context("claude", "claude-sonnet-4-5"),
        transport_context=MockTransportContext(MockLLMClient()),
    )
    replicate = ReplicateAdapter(
        context=_build_context("replicate", "meta/meta-llama-3-70b-instruct"),
        transport_context=MockTransportContext(MockLLMClient()),
    )

    claude_error = claude.normalize_error(
        error=RuntimeError("bad request"),
        http_status=400,
        payload={"error": {"type": "invalid_request_error", "message": "bad schema"}},
    )
    replicate_error = replicate.normalize_error(
        error=RuntimeError("prediction failed"),
        payload={"status": "failed", "error": "E1001: out of memory"},
    )

    assert claude_error.category == "llm_call_failure"
    assert claude_error.details.reason_code == "invalid_request"
    assert claude_error.details.provider_error_type == "invalid_request_error"
    assert replicate_error.category == "llm_call_failure"
    assert replicate_error.message == "E1001: out of memory"
