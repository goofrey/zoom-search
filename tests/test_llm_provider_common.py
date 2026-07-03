from __future__ import annotations

from zoom_search.models import ProviderCapability
from zoom_search.models import LLMRequestOptions
from zoom_search.models import UnifiedLLMRequest
from zoom_search.models import UnifiedMessage
from zoom_search.models import UnifiedToolDefinition
from zoom_search.providers.capabilities import BUILTIN_LLM_CAPABILITIES
from zoom_search.providers.llm import BUILTIN_LLM_REGISTRY
from zoom_search.providers.llm import BaseLLMAdapter
from zoom_search.providers.llm import apply_provider_request_patches
from zoom_search.providers.llm import apply_provider_response_patches
from zoom_search.providers.llm import build_response_format_route
from zoom_search.providers.llm import ensure_json_keyword_hint
from zoom_search.providers.llm import map_provider_error_reason
from zoom_search.providers.llm import normalize_finish_reason
from zoom_search.providers.llm import normalize_custom_native_llm_response_payload
from zoom_search.providers.llm import normalize_llm_provider_error
from zoom_search.providers.llm import normalize_llm_response_payload
from zoom_search.providers.llm import normalize_usage
from zoom_search.providers.llm import sanitize_schema_for_provider


class StubAdapter(BaseLLMAdapter):
    pass


def test_llm_capability_registry_reads_declared_capabilities() -> None:
    assert BUILTIN_LLM_REGISTRY.supports_json_schema("openai") is True
    assert BUILTIN_LLM_REGISTRY.supports_json_schema("gemini") is True
    assert BUILTIN_LLM_REGISTRY.supports_json_object("glm-china") is True
    assert BUILTIN_LLM_REGISTRY.recommended_query_rewriting_mode("minimax-global") == "prompt_only_json"
    assert BUILTIN_LLM_CAPABILITIES["claude"].adapter_class == "dedicated_native"


def test_structured_output_route_prefers_schema_then_json_object_then_prompt_only() -> None:
    schema_request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="openai",
        model="gpt-4o-mini",
        messages=[UnifiedMessage(role="user", content="return json")],
        expect_json=True,
        json_schema={"type": "object", "properties": {"ok": {"type": "string"}}},
    )
    json_object_request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="glm-china",
        model="glm-4-flash",
        messages=[UnifiedMessage(role="user", content="return json")],
        expect_json=True,
    )
    prompt_only_request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="minimax-global",
        model="MiniMax-M2.7",
        messages=[UnifiedMessage(role="user", content="return json")],
        expect_json=True,
        json_schema={"type": "object"},
    )

    schema_format, schema_debug = build_response_format_route(
        capability=BUILTIN_LLM_CAPABILITIES["openai"],
        request=schema_request,
    )
    json_object_format, json_object_debug = build_response_format_route(
        capability=BUILTIN_LLM_CAPABILITIES["glm-china"],
        request=json_object_request,
    )
    prompt_only_format, prompt_only_debug = build_response_format_route(
        capability=BUILTIN_LLM_CAPABILITIES["minimax-global"],
        request=prompt_only_request,
    )

    assert schema_format == {"type": "json_schema", "json_schema": schema_request.json_schema}
    assert schema_debug["route"] == "json_schema"
    assert json_object_format == {"type": "json_object"}
    assert json_object_debug["route"] == "json_object"
    assert prompt_only_format is None
    assert prompt_only_debug["route"] == "prompt_only_json"


def test_tool_route_and_parameter_normalization() -> None:
    capability = ProviderCapability(
        engine="glm-global",
        provider_kind="builtin",
        protocol="openai_compatible",
        adapter_class="openai_compatible_patched",
        tool_calling_level="native_tools_partial",
        tool_choice_mode="auto_only",
        structured_output_level="native_json_object_only",
        supports_seed="unsupported",
        max_temperature=1.0,
    )
    adapter = StubAdapter(capability=capability, provider_engine="glm-global", provider_model="glm-5")
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="glm-global",
        model="glm-5",
        messages=[UnifiedMessage(role="user", content="answer")],
        temperature=2.0,
        seed=9,
        tools=[UnifiedToolDefinition(name="lookup")],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )

    normalized = adapter.normalize_request(request)

    assert normalized.request.temperature == 1.0
    assert normalized.request.seed is None
    assert normalized.request.tool_choice == "auto"
    assert normalized.clamped_params == {"temperature": 1.0}
    assert "seed" in normalized.omitted_params
    assert normalized.patched_params == {"tool_choice": "auto"}


def test_request_options_can_disable_default_llm_fields() -> None:
    capability = ProviderCapability(
        engine="deepseek",
        provider_kind="builtin",
        protocol="openai_compatible",
        adapter_class="openai_compatible",
        structured_output_level="native_json_object_only",
        supports_seed="supported",
        streaming_level="sse_full_fidelity",
    )
    adapter = StubAdapter(
        capability=capability,
        provider_engine="deepseek",
        provider_model="deepseek-chat",
        request_options=LLMRequestOptions(
            temperature=False,
            response_format=False,
            seed=False,
            stream=False,
            reasoning=False,
        ),
    )
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="deepseek",
        model="deepseek-chat",
        messages=[UnifiedMessage(role="user", content="answer")],
        temperature=0.2,
        seed=9,
        stream=True,
        expect_json=True,
        json_schema={"type": "object"},
    )

    normalized = adapter.normalize_request(request)

    assert normalized.request.temperature is None
    assert normalized.request.seed is None
    assert normalized.request.stream is False
    assert normalized.response_format is None
    assert normalized.omitted_params == ["temperature", "seed", "stream", "response_format"]


def test_explicit_reasoning_false_maps_to_verified_provider_controls() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="deepseek",
        model="deepseek-v4-flash",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["deepseek"],
        request=request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert patched.request.extra_params["thinking"] == {"type": "disabled"}
    assert patched.debug["reasoning"]["status"] == "disabled"
    assert patched.debug["reasoning"]["source"] == "explicit_false"


def test_explicit_reasoning_false_maps_to_together_reasoning_toggle() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="together",
        model="zai-org/GLM-5.2",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["together"],
        request=request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert patched.request.extra_params["reasoning"] == {"enabled": False}
    assert patched.debug["reasoning"]["status"] == "disabled"
    assert patched.debug["reasoning"]["source"] == "explicit_false"


def test_explicit_reasoning_false_maps_to_additional_verified_provider_controls() -> None:
    cases = [
        ("fireworks", "accounts/fireworks/models/deepseek-r1", "reasoning", {"effort": "none"}),
        ("openrouter", "openai/gpt-5", "reasoning", {"effort": "none"}),
        ("cohere", "command-a-thinking", "thinking", {"type": "disabled"}),
        ("siliconflow", "Qwen/Qwen3-32B", "enable_thinking", False),
        ("novita", "zai-org/glm-5.1", "enable_thinking", False),
        ("deepinfra", "deepseek-ai/DeepSeek-R1", "reasoning", {"enabled": False}),
        ("hunyuan", "hy3-preview", "thinking", {"type": "disabled"}),
        ("groq", "qwen/qwen3-32b", "reasoning_effort", "none"),
        ("cerebras", "zai-glm-4.7", "reasoning_effort", "none"),
        ("grok", "grok-4.3", "reasoning", {"effort": "none"}),
        ("mistral", "magistral-medium", "reasoning_effort", "none"),
        ("mimo", "mimo-v2.5-pro", "thinking", {"type": "disabled"}),
        ("ollama", "llama3.2", "think", False),
    ]

    for provider, model, param, value in cases:
        request = UnifiedLLMRequest(
            task="answer_synthesis",
            provider=provider,
            model=model,
            messages=[UnifiedMessage(role="user", content="answer")],
        )
        patched = apply_provider_request_patches(
            capability=BUILTIN_LLM_CAPABILITIES[provider],
            request=request,
            response_format=None,
            request_options=LLMRequestOptions(reasoning=False),
        )

        assert patched.request.extra_params[param] == value
        assert patched.debug["reasoning"]["status"] == "disabled"


def test_model_scoped_reasoning_disable_reports_expected_diagnostics() -> None:
    hunyuan_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="hunyuan",
        model="hunyuan-role-latest",
        messages=[UnifiedMessage(role="user", content="answer")],
    )
    siliconflow_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="siliconflow",
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[UnifiedMessage(role="user", content="answer")],
    )
    novita_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="novita",
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[UnifiedMessage(role="user", content="answer")],
    )
    ollama_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="ollama",
        model="gpt-oss:20b",
        messages=[UnifiedMessage(role="user", content="answer")],
    )
    stepfun_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="stepfun",
        model="step-3.7-flash",
        messages=[UnifiedMessage(role="user", content="answer")],
    )
    yi_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="yi",
        model="yi-large",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    hunyuan_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["hunyuan"],
        request=hunyuan_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )
    siliconflow_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["siliconflow"],
        request=siliconflow_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )
    novita_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["novita"],
        request=novita_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )
    ollama_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["ollama"],
        request=ollama_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )
    stepfun_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["stepfun"],
        request=stepfun_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )
    yi_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["yi"],
        request=yi_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert "EnableThinking" not in hunyuan_patched.request.extra_params
    assert hunyuan_patched.debug["reasoning"]["status"] == "not_applicable"
    assert "enable_thinking" not in siliconflow_patched.request.extra_params
    assert siliconflow_patched.debug["reasoning"]["status"] == "not_applicable"
    assert "enable_thinking" not in novita_patched.request.extra_params
    assert "reasoning_control_unknown_for_model" in novita_patched.warnings
    assert novita_patched.debug["reasoning"]["status"] == "unknown_model_family"
    assert "think" not in ollama_patched.request.extra_params
    assert "reasoning_disable_unsupported_for_model" in ollama_patched.warnings
    assert ollama_patched.debug["reasoning"]["status"] == "unsupported_for_model"
    assert "reasoning_disable_unsupported_for_model" in stepfun_patched.warnings
    assert stepfun_patched.debug["reasoning"]["status"] == "unsupported_for_model"
    assert yi_patched.debug["reasoning"]["status"] == "provider_default"


def test_hunyuan_legacy_and_always_on_reasoning_models_follow_documented_controls() -> None:
    legacy_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="hunyuan",
        model="tencent/Hunyuan-A13B-Instruct",
        messages=[UnifiedMessage(role="user", content="answer")],
    )
    always_on_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="hunyuan",
        model="hunyuan-2.0-thinking-20251109",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    legacy_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["hunyuan"],
        request=legacy_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )
    always_on_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["hunyuan"],
        request=always_on_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert legacy_patched.request.extra_params["EnableThinking"] is False
    assert legacy_patched.debug["reasoning"]["status"] == "disabled"
    assert "thinking" not in legacy_patched.request.extra_params
    assert "thinking" not in always_on_patched.request.extra_params
    assert "EnableThinking" not in always_on_patched.request.extra_params
    assert "reasoning_disable_unsupported_for_model" in always_on_patched.warnings
    assert always_on_patched.debug["reasoning"]["status"] == "unsupported_for_model"


def test_novita_reasoning_disable_whitelist_covers_confirmed_families() -> None:
    models = [
        "deepseek/deepseek-r1-0528",
        "deepseek/deepseek-v3.1-terminus",
        "minimax/minimax-m3",
        "openai/gpt-oss-120b",
    ]

    for model in models:
        request = UnifiedLLMRequest(
            task="answer_synthesis",
            provider="novita",
            model=model,
            messages=[UnifiedMessage(role="user", content="answer")],
        )
        patched = apply_provider_request_patches(
            capability=BUILTIN_LLM_CAPABILITIES["novita"],
            request=request,
            response_format=None,
            request_options=LLMRequestOptions(reasoning=False),
        )

        assert patched.request.extra_params["enable_thinking"] is False
        assert patched.debug["reasoning"]["status"] == "disabled"


def test_explicit_reasoning_false_applies_outside_json_mode_for_qwen_and_glm_global() -> None:
    qwen_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="qwen-global",
        model="qwen-plus",
        messages=[UnifiedMessage(role="user", content="answer")],
    )
    glm_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="glm-global",
        model="glm-5",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    qwen_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["qwen-global"],
        request=qwen_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )
    glm_patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["glm-global"],
        request=glm_request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert qwen_patched.request.extra_params["thinking"] == {"type": "disabled"}
    assert glm_patched.request.extra_params["thinking"] == {"type": "disabled"}


def test_explicit_reasoning_false_surfaces_always_on_diagnostic_for_kimi_k27() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="kimi-global",
        model="kimi-k2.7-code",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["kimi-global"],
        request=request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert "thinking" not in patched.request.extra_params
    assert "reasoning_always_on_for_model" in patched.warnings
    assert patched.debug["reasoning"]["status"] == "always_on"


def test_explicit_reasoning_true_suppresses_default_json_disable_patch() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="doubao-global",
        model="seed-1-6",
        messages=[UnifiedMessage(role="user", content="return json")],
        expect_json=True,
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["doubao-global"],
        request=request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=True),
    )

    assert "thinking" not in patched.request.extra_params
    assert patched.debug["reasoning"]["requested"] is True
    assert patched.debug["reasoning"]["status"] == "provider_default"


def test_error_envelope_extraction_preserves_provider_fields() -> None:
    error = normalize_llm_provider_error(
        error=RuntimeError("bad request"),
        request_id="req-1",
        provider_engine="gemini",
        provider_model="gemini-2.5-flash",
        http_status=429,
        payload={
            "error": {
                "message": "Too many requests",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
                "param": "model",
            }
        },
    )

    assert error.category == "llm_call_failure"
    assert error.details.error_type == "rate_limit_error"
    assert error.details.reason_code == "rate_limited"
    assert error.details.provider_error_code == "rate_limit_exceeded"
    assert error.details.provider_error_type == "rate_limit_error"
    assert error.details.provider_error_param == "model"
    assert error.raw_diagnostics is not None
    assert error.raw_diagnostics.provider_error_code == "rate_limit_exceeded"


def test_finish_reason_normalization_covers_common_provider_variants() -> None:
    assert normalize_finish_reason("stop") == "stop"
    assert normalize_finish_reason("MAX_TOKENS") == "length"
    assert normalize_finish_reason("tool_use") == "tool_calls"
    assert normalize_finish_reason("sensitive") == "content_filter"
    assert normalize_finish_reason("network_error") == "error"
    assert normalize_finish_reason("model_context_window_exceeded") == "length"
    assert normalize_finish_reason(None) == "unknown"


def test_usage_missing_is_tolerated() -> None:
    assert normalize_usage(None) == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cached_input_tokens": None,
    }
    assert normalize_usage({"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})["total_tokens"] == 10


def test_response_normalization_handles_json_and_tool_calls() -> None:
    response = normalize_llm_response_payload(
        payload={
            "choices": [
                {
                    "message": {
                        "content": '{"ok": true}',
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "lookup", "arguments": {"q": "hotel"}},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        },
        capability=BUILTIN_LLM_CAPABILITIES["openai"],
        provider="openai",
        model="gpt-4o-mini",
    )

    assert response.json_payload == {"ok": True}
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].arguments_json_text == '{"q": "hotel"}'
    assert response.usage["total_tokens"] == 11


def test_group_a_capabilities_are_registered() -> None:
    for engine in ["gemini", "doubao-global", "doubao-china", "qwen-global", "qwen-china", "glm-china"]:
        assert engine in BUILTIN_LLM_CAPABILITIES


def test_group_b_capabilities_are_registered_with_conservative_defaults() -> None:
    assert BUILTIN_LLM_CAPABILITIES["glm-global"].tool_calling_level == "patched_tools"
    assert BUILTIN_LLM_CAPABILITIES["baichuan"].structured_output_level == "native_json_object_only"
    assert BUILTIN_LLM_CAPABILITIES["spark"].supports_finish_reason == "patched"
    assert BUILTIN_LLM_CAPABILITIES["huggingface"].tool_calling_level == "unsupported"


def test_minimax_china_capability_is_registered_with_dual_path_defaults() -> None:
    capability = BUILTIN_LLM_CAPABILITIES["minimax-china"]

    assert capability.adapter_class == "dual_path"
    assert capability.structured_output_level == "native_json_schema_subset"
    assert capability.recommended_query_rewriting_mode == "json_schema"


def test_group_c_openai_compatible_capabilities_are_registered() -> None:
    expected_base_urls = {
        "deepseek": "https://api.deepseek.com/v1",
        "kimi-china": "https://api.moonshot.cn/v1",
        "kimi-global": "https://api.moonshot.ai/v1",
        "yi": "https://api.lingyiwanwu.com/v1",
        "hunyuan": "https://tokenhub.tencentmaas.com/v1",
        "stepfun": "https://api.stepfun.com/v1",
        "siliconflow": "https://api.siliconflow.cn/v1",
        "together": "https://api.together.xyz/v1",
        "fireworks": "https://api.fireworks.ai/inference/v1",
        "groq": "https://api.groq.com/openai/v1",
        "cerebras": "https://api.cerebras.ai/v1",
        "perplexity": "https://api.perplexity.ai",
        "grok": "https://api.x.ai/v1",
        "mistral": "https://api.mistral.ai/v1",
        "cohere": "https://api.cohere.com/compatibility/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "mimo": "https://api.xiaomimimo.com/v1",
        "deepinfra": "https://api.deepinfra.com/v1/openai",
        "novita": "https://api.novita.ai/openai/v1",
        "hyperbolic": "https://api.hyperbolic.xyz/v1",
        "lepton": "",
        "ollama": "http://localhost:11434/v1",
    }

    for engine, base_url in expected_base_urls.items():
        capability = BUILTIN_LLM_CAPABILITIES[engine]
        assert capability.protocol == "openai_compatible"
        assert capability.adapter_class == "openai_compatible"
        assert capability.structured_output_level == "native_json_object_only"
        assert capability.tool_calling_level == "native_tools_partial"
        assert capability.default_base_url == base_url


def test_glm_global_request_patch_applies_capability_downgrade_and_debug_warnings() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="glm-global",
        model="glm-5",
        messages=[UnifiedMessage(role="user", content="rewrite")],
        expect_json=True,
        json_schema={"type": "object"},
        temperature=0,
        top_p=0,
        stop_sequences=["A", "B"],
        seed=1,
        tools=[UnifiedToolDefinition(name="lookup")],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["glm-global"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["glm-global"],
        request=request,
        response_format=response_format,
    )

    assert patched.response_format == {"type": "json_object"}
    assert patched.request.temperature == 0.01
    assert patched.request.top_p == 0.01
    assert patched.request.stop_sequences == ["A"]
    assert patched.request.tool_choice == "auto"
    assert patched.request.seed is None
    assert patched.request.extra_params["thinking"] == {"type": "disabled"}
    assert "json_schema_downgraded_to_json_object" in patched.warnings


def test_kimi_request_patch_omits_temperature_for_china_and_global() -> None:
    for provider, model in (("kimi-global", "kimi-k2.6"), ("kimi-china", "moonshot-v1-8k")):
        request = UnifiedLLMRequest(
            task="query_rewriting",
            provider=provider,
            model=model,
            messages=[UnifiedMessage(role="user", content="rewrite")],
            temperature=0,
        )

        patched = apply_provider_request_patches(
            capability=BUILTIN_LLM_CAPABILITIES[provider],
            request=request,
            response_format=None,
        )

        assert patched.request.temperature is None
        assert "temperature" in patched.omitted_params
        assert "temperature_omitted_for_provider_compatibility" in patched.warnings


def test_baichuan_request_patch_injects_provider_defaults_and_json_route() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="baichuan",
        model="Baichuan4",
        messages=[UnifiedMessage(role="user", content="rewrite")],
        expect_json=True,
        json_schema={"type": "object"},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["baichuan"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["baichuan"],
        request=request,
        response_format=response_format,
    )

    assert patched.response_format == {"type": "json_object"}
    assert patched.request.extra_params["top_k"] == 5
    assert patched.request.extra_params["with_search_enhance"] is False
    assert "json" in patched.request.messages[0].content.lower()


def test_spark_business_error_and_missing_finish_reason_are_patched() -> None:
    response = normalize_llm_response_payload(
        payload={
            "code": 0,
            "message": "success",
            "sid": "sid-1",
            "choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}],
        },
        capability=BUILTIN_LLM_CAPABILITIES["spark"],
        provider="spark",
        model="4.0Ultra",
    )

    assert response.finish_reason == "stop"
    assert response.json_payload == {"ok": True}

    business_error = normalize_llm_provider_error(
        error=RuntimeError("spark business error"),
        request_id="req-spark",
        provider_engine="spark",
        provider_model="4.0Ultra",
        http_status=200,
        payload={"code": 11202, "message": "rate limit", "sid": "sid-2"},
    )

    assert business_error.details.reason_code == "rate_limited"


def test_glm_global_tool_arguments_and_finish_reason_are_patched() -> None:
    response = normalize_llm_response_payload(
        payload={
            "request_id": "req-glm",
            "choices": [
                {
                    "message": {
                        "content": '{"ok": true}',
                        "reasoning_content": "hidden",
                        "tool_calls": [{"id": "t1", "function": {"name": "lookup", "arguments": {"q": "hotel"}}}],
                    },
                    "finish_reason": "model_context_window_exceeded",
                }
            ],
        },
        capability=BUILTIN_LLM_CAPABILITIES["glm-global"],
        provider="glm-global",
        model="glm-5",
    )

    assert response.finish_reason == "length"
    assert response.tool_calls[0].arguments_json_text == '{"q": "hotel"}'
    assert response.provider_metadata["reasoning_content"] == "hidden"


def test_huggingface_response_format_route_and_schema_patch() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="huggingface",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[UnifiedMessage(role="user", content="rewrite")],
        expect_json=True,
        json_schema={"type": "object", "properties": {"name": {"type": ["string", "null"]}}},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["huggingface"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["huggingface"],
        request=request,
        response_format=response_format,
    )

    assert patched.response_format == {
        "type": "json_schema",
        "value": {"type": "object", "properties": {"name": {"anyOf": [{"type": "string"}, {"type": "null"}]}}},
    }
    assert "response_format_rewritten_for_huggingface" in patched.warnings


def test_huggingface_flat_error_and_model_validation_risk_are_visible() -> None:
    error = normalize_llm_provider_error(
        error=RuntimeError("hf bad request"),
        request_id="req-hf",
        provider_engine="huggingface",
        provider_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        http_status=429,
        payload={"error": "Rate limit exceeded", "code": 429, "reason": "RATE_LIMIT_EXCEEDED"},
    )

    assert error.details.reason_code == "rate_limited"

    invalid_model_request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="huggingface",
        model="llama-3",
        messages=[UnifiedMessage(role="user", content="answer")],
    )
    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["huggingface"],
        request=invalid_model_request,
        response_format=None,
    )

    assert "huggingface_model_validation_failed" in patched.warnings


def test_huggingface_reasoning_disable_surfaces_model_specific_diagnostic() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="huggingface",
        model="deepseek-ai/DeepSeek-R1-0528:novita",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["huggingface"],
        request=request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert "reasoning_control_unknown_for_model" in patched.warnings
    assert patched.debug["reasoning"]["status"] == "unknown_model_family"


def test_perplexity_reasoning_disable_surfaces_unknown_model_diagnostic() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="perplexity",
        model="sonar-reasoning-pro",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["perplexity"],
        request=request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert "reasoning_control_unknown_for_model" in patched.warnings
    assert patched.debug["reasoning"]["status"] == "unknown_model_family"


def test_perplexity_response_preserves_citations_metadata() -> None:
    response = normalize_llm_response_payload(
        payload={
            "citations": ["https://example.com/a", "https://example.com/b"],
            "choices": [
                {
                    "message": {
                        "content": "answer",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        },
        capability=BUILTIN_LLM_CAPABILITIES["perplexity"],
        provider="perplexity",
        model="sonar-pro",
    )

    assert response.provider_metadata["citations"] == ["https://example.com/a", "https://example.com/b"]


def test_lepton_reasoning_disable_surfaces_unknown_model_diagnostic() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="lepton",
        model="mistral-7b",
        messages=[UnifiedMessage(role="user", content="answer")],
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["lepton"],
        request=request,
        response_format=None,
        request_options=LLMRequestOptions(reasoning=False),
    )

    assert "reasoning_control_unknown_for_model" in patched.warnings
    assert patched.debug["reasoning"]["status"] == "unknown_model_family"


def test_gemini_schema_subset_sanitizer_removes_unsupported_keywords() -> None:
    sanitized = sanitize_schema_for_provider(
        {
            "title": "RewriteResult",
            "type": "object",
            "$defs": {"X": {"type": "string"}},
            "properties": {
                "search_groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "patternProperties": {".*": {"type": "string"}},
                        "properties": {"group": {"type": "integer", "default": 1}},
                    },
                }
            },
        },
        provider="gemini",
    )

    assert sanitized == {
        "type": "object",
        "properties": {
            "search_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"group": {"type": "integer"}},
                },
            }
        },
    }


def test_qwen_json_schema_downgrades_and_injects_json_hint() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="qwen-global",
        model="qwen-plus",
        messages=[UnifiedMessage(role="user", content="rewrite this")],
        expect_json=True,
        json_schema={"type": "object"},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["qwen-global"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["qwen-global"],
        request=request,
        response_format=response_format,
    )

    assert patched.response_format == {"type": "json_object"}
    assert patched.request.messages[0].role == "system"
    assert "json" in patched.request.messages[0].content.lower()
    assert patched.request.extra_params["thinking"] == {"type": "disabled"}


def test_doubao_request_patch_prefers_endpoint_and_disables_thinking() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="doubao-global",
        model="seed-1-6",
        messages=[UnifiedMessage(role="user", content="return json")],
        expect_json=True,
        extra_params={"endpoint_id": "ep-123"},
        json_schema={"type": "object"},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["doubao-global"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["doubao-global"],
        request=request,
        response_format=response_format,
    )

    assert patched.request.model == "ep-123"
    assert patched.request.extra_params["thinking"] == {"type": "disabled"}
    assert patched.response_format["strict"] is True


def test_gemini_request_patch_omits_undocumented_reasoning_flag() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[UnifiedMessage(role="user", content="return json")],
        expect_json=True,
        json_schema={"type": "object"},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["gemini"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["gemini"],
        request=request,
        response_format=response_format,
    )

    assert "thinking" not in patched.request.extra_params


def test_spark_request_patch_folds_system_message_and_omits_tools_for_basic_models() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="spark",
        model="lite",
        messages=[UnifiedMessage(role="system", content="Use JSON"), UnifiedMessage(role="user", content="answer")],
        tools=[UnifiedToolDefinition(name="lookup")],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        expect_json=True,
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["spark"],
        request=request,
        response_format={"type": "json_object"},
    )

    assert patched.request.messages[0].role == "user"
    assert "System instructions:" in patched.request.messages[0].content
    assert patched.request.tools == []
    assert patched.request.tool_choice is None
    assert "spark_system_folded_into_user" in patched.warnings
    assert "spark_tools_unsupported_for_model" in patched.warnings


def test_glm_china_tool_choice_and_determinism_patch() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="glm-china",
        model="glm-5.1",
        messages=[UnifiedMessage(role="user", content="call tool")],
        temperature=0,
        tools=[UnifiedToolDefinition(name="lookup")],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        expect_json=True,
        json_schema={"type": "object"},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["glm-china"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["glm-china"],
        request=request,
        response_format=response_format,
    )

    assert patched.request.tool_choice == "auto"
    assert patched.request.extra_params["do_sample"] is False
    assert patched.response_format == {"type": "json_object"}


def test_tools_disable_stream_for_qwen() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="qwen-china",
        model="qwen-plus",
        messages=[UnifiedMessage(role="user", content="json")],
        tools=[UnifiedToolDefinition(name="lookup")],
        stream=True,
    )

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["qwen-china"],
        request=request,
        response_format=None,
    )

    assert patched.request.stream is False
    assert "tools_disable_stream" in patched.warnings


def test_minimax_global_prompt_only_json_patch_and_temperature_mapping() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="minimax-global",
        model="MiniMax-M2.7",
        messages=[UnifiedMessage(role="user", content="rewrite")],
        expect_json=True,
        json_schema={"type": "object"},
        temperature=0,
        max_tokens=128,
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["minimax-global"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["minimax-global"],
        request=request,
        response_format=response_format,
    )

    assert patched.response_format is None
    assert patched.request.temperature == 0.01
    assert patched.request.extra_params["reasoning_split"] is True
    assert patched.request.extra_params["max_completion_tokens"] == 128
    assert "minimax-global-schema-downgraded-to-prompt-json" in patched.warnings


def test_minimax_response_cleanup_parses_think_and_fenced_json() -> None:
    response = normalize_llm_response_payload(
        payload={
            "base_resp": {"status_code": 0, "status_msg": ""},
            "choices": [
                {
                    "message": {
                        "content": "<think>internal</think>```json\n{\"ok\": true}\n```",
                        "reasoning_details": [{"type": "reasoning", "text": "internal"}],
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        capability=BUILTIN_LLM_CAPABILITIES["minimax-global"],
        provider="minimax-global",
        model="MiniMax-M2.7",
    )

    assert response.text == '{"ok": true}'
    assert response.json_payload == {"ok": True}
    assert response.provider_metadata["think_tags_detected"] is True
    assert response.provider_metadata["reasoning_split_effective"] is True


def test_minimax_business_error_in_base_resp_is_normalized() -> None:
    error = normalize_llm_provider_error(
        error=RuntimeError("provider business error"),
        request_id="req-minimax",
        provider_engine="minimax-global",
        provider_model="MiniMax-M2.7",
        http_status=200,
        payload={"base_resp": {"status_code": 1002, "status_msg": "rate limit"}},
    )

    assert error.details.reason_code == "rate_limited"
    assert error.details.provider_error_code == "1002"


def test_minimax_china_routes_native_for_query_rewriting_schema() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="minimax-china",
        model="MiniMax-M2.7",
        messages=[UnifiedMessage(role="user", content="rewrite")],
        expect_json=True,
        json_schema={"type": "object"},
        extra_params={"structured_output_model": "MiniMax-Text-01"},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["minimax-china"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["minimax-china"],
        request=request,
        response_format=response_format,
    )

    assert patched.patched_params["route"] == "native"
    assert patched.request.model == "MiniMax-Text-01"
    assert patched.request.extra_params["api_route"] == "native_chatcompletion_v2"
    assert patched.request.extra_params["native_response_format"]["type"] == "json_schema"


def test_minimax_china_routes_compat_for_tools_and_downgrades_structured_output() -> None:
    request = UnifiedLLMRequest(
        task="answer_synthesis",
        provider="minimax-china",
        model="MiniMax-M2.7",
        messages=[UnifiedMessage(role="user", content="answer")],
        expect_json=True,
        json_schema={"type": "object"},
        tools=[UnifiedToolDefinition(name="lookup")],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )
    response_format, _ = build_response_format_route(capability=BUILTIN_LLM_CAPABILITIES["minimax-china"], request=request)

    patched = apply_provider_request_patches(
        capability=BUILTIN_LLM_CAPABILITIES["minimax-china"],
        request=request,
        response_format=response_format,
    )

    assert patched.patched_params["route"] == "compat"
    assert patched.response_format is None
    assert patched.request.tool_choice == "auto"
    assert patched.request.extra_params["reasoning_split"] is True
    assert "minimax-china-compat-structured-output-downgraded" in patched.warnings


def test_minimax_provider_error_mapping_handles_auth_and_content_codes() -> None:
    assert map_provider_error_reason(
        capability_engine="minimax-china",
        http_status=401,
        error_code="2049",
        error_type="auth_error",
        message="invalid api key",
    ) == ("auth_error", False)
    assert map_provider_error_reason(
        capability_engine="minimax-global",
        http_status=400,
        error_code="1026",
        error_type="content_filtered",
        message="sensitive",
    ) == ("content_filtered", False)


def test_glm_finish_reason_and_usage_are_normalized() -> None:
    response = normalize_llm_response_payload(
        payload={
            "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "model_context_window_exceeded"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "prompt_tokens_details": {"cached_tokens": 2}},
        },
        capability=BUILTIN_LLM_CAPABILITIES["glm-china"],
        provider="glm-china",
        model="glm-5.1",
    )

    assert response.finish_reason == "length"
    assert response.usage["cached_input_tokens"] == 2


def test_custom_native_response_mapping_extracts_text_json_usage_and_finish_reason() -> None:
    response = normalize_custom_native_llm_response_payload(
        payload={
            "data": {"output": {"text": '{"answer": "ripe"}', "json": {"answer": "ripe"}}},
            "stats": {"input": 11, "output": 7, "total": 18, "reasoning": 2, "cached": 3},
            "state": {"finish": "stop"},
        },
        mapping={
            "content_path": "data.output.text",
            "json_payload_path": "data.output.json",
            "usage_input_tokens_path": "stats.input",
            "usage_output_tokens_path": "stats.output",
            "usage_total_tokens_path": "stats.total",
            "usage_reasoning_tokens_path": "stats.reasoning",
            "usage_cached_input_tokens_path": "stats.cached",
            "finish_reason_path": "state.finish",
        },
        capability=ProviderCapability(engine="custom", provider_kind="custom", protocol="native", adapter_class="custom_native"),
        provider="custom",
        model="native-model",
        expect_json=True,
    )

    assert response.text == '{"answer": "ripe"}'
    assert response.json_payload == {"answer": "ripe"}
    assert response.finish_reason == "stop"
    assert response.usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "reasoning_tokens": 2,
        "cached_input_tokens": 3,
    }


def test_custom_native_response_mapping_supports_array_index_paths_and_text_json_fallback() -> None:
    response = normalize_custom_native_llm_response_payload(
        payload={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]},
        mapping={"content_path": "choices.0.message.content", "finish_reason_path": "choices.0.finish_reason"},
        capability=ProviderCapability(engine="custom", provider_kind="custom", protocol="native", adapter_class="custom_native"),
        provider="custom",
        model="native-model",
        expect_json=True,
    )

    assert response.text == '{"ok": true}'
    assert response.json_payload == {"ok": True}
    assert response.finish_reason == "stop"


def test_provider_specific_error_mapping() -> None:
    assert map_provider_error_reason(
        capability_engine="doubao-global",
        http_status=400,
        error_code="SensitiveContentDetected.PolicyViolation",
        error_type="BadRequest",
        message="blocked",
    ) == ("content_filtered", False)
    assert map_provider_error_reason(
        capability_engine="qwen-global",
        http_status=400,
        error_code="DataInspectionFailed",
        error_type="invalid_request_error",
        message="inspection failed",
    ) == ("content_filtered", False)
    assert map_provider_error_reason(
        capability_engine="glm-china",
        http_status=429,
        error_code="1302",
        error_type="rate_limit",
        message="too many requests",
    ) == ("rate_limited", True)


def test_ensure_json_keyword_hint_is_stable() -> None:
    request = UnifiedLLMRequest(
        task="query_rewriting",
        provider="qwen-china",
        model="qwen-plus",
        messages=[UnifiedMessage(role="system", content="Be precise."), UnifiedMessage(role="user", content="rewrite")],
    )

    patched_once, injected_once = ensure_json_keyword_hint(request)
    patched_twice, injected_twice = ensure_json_keyword_hint(patched_once)

    assert injected_once is True
    assert injected_twice is False
    assert patched_once.messages[0].content.count("JSON") == 1
