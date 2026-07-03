"""Core data models for zoom_search."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any
from typing import Literal

OutputMode = Literal[
    "answer",
    "answer_with_sources",
    "results_simple",
    "results_detailed",
]
SearchStreamEventType = Literal[
    "search_started",
    "search_completed",
    "answer_started",
    "answer_delta",
    "answer_completed",
    "completed",
]
ErrorCategory = Literal[
    "llm_configuration_error",
    "search_configuration_error",
    "proxy_configuration_error",
    "llm_call_failure",
    "search_call_failure",
    "network_connection_failure",
]
ErrorType = Literal[
    "configuration_error",
    "authentication_error",
    "invalid_request_error",
    "rate_limit_error",
    "quota_exceeded_error",
    "content_filtered_error",
    "network_error",
    "provider_error",
    "empty_result_error",
]
ComponentName = Literal[
    "llm",
    "search",
    "proxy",
    "transport",
    "validation",
    "orchestration",
]
ProviderKind = Literal["demo", "builtin", "custom"]
ProviderProtocol = Literal["openai_compatible", "native"]
SiteRestrictionMode = Literal["none", "provider_side", "query_side"]
QuerySiteSupport = Literal["true", "false", "unknown"]
ZoomInStrategy = Literal["provider_side", "query_side", "best_effort"]
SiteValueType = Literal["string", "string_array", "delimiter_string"]
NumResultsMode = Literal["global_limit", "per_collection_limit", "unsupported"]
CapabilityConfidence = Literal["high", "medium", "low"]
CapabilitySupport = Literal["supported", "patched", "unsupported"]
AdapterClass = Literal[
    "openai_compatible",
    "openai_compatible_patched",
    "dedicated_native",
    "custom_native",
    "dual_path",
    "async_prediction",
]
StructuredOutputLevel = Literal[
    "native_json_schema_strict",
    "native_json_schema_subset",
    "native_json_object_only",
    "prompt_only_json",
    "unsupported",
]
ToolCallingLevel = Literal[
    "native_tools_full",
    "native_tools_partial",
    "patched_tools",
    "prompt_emulated",
    "unsupported",
]
StreamingLevel = Literal[
    "sse_full_fidelity",
    "sse_text_only",
    "pseudo_stream",
    "poll_then_collect",
    "unsupported",
]
ExecutionModel = Literal[
    "sync_request_response",
    "sync_with_stream",
    "dual_path_sync_or_native",
    "async_prediction_poll",
    "hybrid_async_stream",
]
ReasoningPollutionRisk = Literal["none", "low", "medium", "high"]
ToolArgumentShape = Literal["string_json", "object_json", "mixed", "unsupported"]
ErrorShapeFamily = Literal["openai_like", "flat", "nested", "stateful_async", "custom"]
QueryRewritingMode = Literal["json_schema", "json_object", "prompt_only_json"]
RetryProfile = Literal["strict_json", "tolerant_json", "async_prediction", "streaming_patch"]
SchemaSubsetProfile = Literal["openapi_3_subset", "provider_custom_subset", "none"]
ToolChoiceMode = Literal["none", "auto_only", "auto_required_named", "provider_custom"]
ToolResultChannel = Literal["message_role", "content_block", "provider_custom", "unsupported"]
ToolArgumentEncoding = Literal["json_string", "json_object", "mixed", "unsupported"]
UnifiedResponseFormatMode = Literal["text", "json_schema", "json_object", "prompt_only_json"]
UnifiedFinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "error",
    "cancelled",
    "timeout",
    "unknown",
]


@dataclass(slots=True)
class SearchLimits:
    zoomout_num_results: int = 5
    zoomin_num_results: int = 5
    top_k_domains_per_query: int = 1


@dataclass(slots=True)
class ProxyConfig:
    http_proxy: str | None = None


@dataclass(slots=True)
class LLMRequestOptions:
    temperature: bool | None = None
    response_format: bool | None = None
    seed: bool | None = None
    stream: bool | None = None
    reasoning: bool | None = None


@dataclass(slots=True)
class LLMConfig:
    engine: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    protocol: ProviderProtocol | None = None
    http_proxy: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    request_options: LLMRequestOptions = field(default_factory=LLMRequestOptions)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchConfig:
    engine: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    http_proxy: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchRequest:
    question: str
    previous_conversation: list[str] = field(default_factory=list)
    output_mode: OutputMode = "answer"
    search_limits: SearchLimits = field(default_factory=SearchLimits)
    llm: LLMConfig | None = None
    search: SearchConfig | None = None
    demo_mode: bool = False
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    seed: int | None = None


@dataclass(slots=True)
class TraceabilityInfo:
    previous_conversation: list[str] = field(default_factory=list)
    original_input: str | None = None
    comparison_question: str | None = None
    split_question: str | None = None
    phase: str | None = None
    search_query_used: str | None = None
    rank: int | None = None
    split_question_id: int | None = None
    query_variant_id: Literal[1, 2] | None = None

    def enriched(self, **changes: Any) -> "TraceabilityInfo":
        data = {
            "previous_conversation": list(self.previous_conversation),
            "original_input": self.original_input,
            "comparison_question": self.comparison_question,
            "split_question": self.split_question,
            "phase": self.phase,
            "search_query_used": self.search_query_used,
            "rank": self.rank,
            "split_question_id": self.split_question_id,
            "query_variant_id": self.query_variant_id,
        }
        data.update(changes)
        return TraceabilityInfo(**data)


@dataclass(slots=True)
class DuplicateTraceabilityInfo:
    phase: str | None = None
    split_question: str | None = None
    split_question_id: int | None = None
    query_variant_id: Literal[1, 2] | None = None
    search_query_used: str | None = None
    rank: int | None = None
    source_domain: str | None = None
    title: str | None = None
    url: str | None = None


@dataclass(slots=True)
class SearchGroup:
    group: int
    original_input: str
    comparison_question: str | None
    split_question: str
    main_term: str
    key_noun: str
    alias1: str
    alias2: str
    query1: str
    query2: str


@dataclass(slots=True)
class ZoomOutSearchRequest:
    split_question_id: int
    split_question: str
    query: str
    query_variant_id: Literal[1, 2]
    num_results: int
    traceability: TraceabilityInfo


@dataclass(slots=True)
class ZoomOutSearchResult:
    title: str
    snippet: str
    url: str
    rank: int
    traceability: TraceabilityInfo
    source_domain: str | None = None
    provider_score: float | None = None


@dataclass(slots=True)
class SourceDomainRecord:
    source_domain: str
    split_question: str
    query1: str
    rank: int
    traceability: TraceabilityInfo
    provider_score: float | None = None
    duplicate_traceabilities: list[DuplicateTraceabilityInfo] = field(default_factory=list)


@dataclass(slots=True)
class ZoomInSearchRequest:
    source_domain: str
    split_question: str
    query: str
    num_results: int
    site_restriction_mode: SiteRestrictionMode
    traceability: TraceabilityInfo
    provider_site_value: str | None = None


@dataclass(slots=True)
class ZoomInSearchResult:
    title: str
    snippet: str
    url: str
    rank: int
    traceability: TraceabilityInfo
    source_domain: str | None = None


@dataclass(slots=True)
class FinalSearchResult:
    title: str
    snippet: str
    url: str
    source_domain: str | None
    traceability: TraceabilityInfo
    duplicate_traceabilities: list[DuplicateTraceabilityInfo] = field(default_factory=list)


@dataclass(slots=True)
class SimpleSearchResult:
    title: str
    snippet: str
    url: str


@dataclass(slots=True)
class WarningInfo:
    code: str
    message: str
    phase: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RawDiagnostics:
    http_status: int | None = None
    provider_error_body: dict[str, Any] | list[Any] | str | None = None
    provider_error_headers: dict[str, str] | None = None
    provider_error_code: str | None = None
    provider_error_type: str | None = None
    provider_error_param: str | None = None
    transport_exception_class: str | None = None
    transport_exception_message: str | None = None


@dataclass(slots=True)
class ErrorDetails:
    error_type: ErrorType | None = None
    reason_code: str | None = None
    invalid_fields: list[str] = field(default_factory=list)
    http_status: int | None = None
    provider_error_code: str | None = None
    provider_error_type: str | None = None
    provider_error_param: str | None = None


@dataclass(slots=True)
class ZoomSearchError(Exception):
    category: ErrorCategory
    message: str
    user_message: str
    component: ComponentName
    request_id: str = "pending"
    provider_engine: str | None = None
    provider_model: str | None = None
    retryable: bool = False
    retry_attempted: bool = False
    details: ErrorDetails = field(default_factory=ErrorDetails)
    raw_diagnostics: RawDiagnostics | None = None

    def __str__(self) -> str:
        return self.message


@dataclass(slots=True)
class ProviderCapability:
    engine: str
    provider_kind: ProviderKind
    protocol: ProviderProtocol | None = None
    adapter_class: AdapterClass = "openai_compatible"
    structured_output_level: StructuredOutputLevel = "unsupported"
    tool_calling_level: ToolCallingLevel = "unsupported"
    streaming_level: StreamingLevel = "unsupported"
    execution_model: ExecutionModel = "sync_request_response"
    supports_system_message: bool = True
    supports_multi_message_roles: bool = True
    supports_structured_output: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = False
    supports_temperature: CapabilitySupport = "supported"
    supports_seed: CapabilitySupport = "unsupported"
    supports_stop_sequences: CapabilitySupport = "supported"
    supports_max_tokens: CapabilitySupport = "supported"
    supports_parallel_tool_calls: CapabilitySupport = "unsupported"
    supports_reasoning_control: CapabilitySupport = "unsupported"
    supports_usage_reporting: CapabilitySupport = "supported"
    supports_finish_reason: CapabilitySupport = "supported"
    supports_native_json_schema: CapabilitySupport = "unsupported"
    supports_json_object_mode: CapabilitySupport = "unsupported"
    requires_json_keyword_hint: bool = False
    reasoning_pollution_risk: ReasoningPollutionRisk = "none"
    tool_argument_shape: ToolArgumentShape = "unsupported"
    error_shape_family: ErrorShapeFamily = "openai_like"
    recommended_query_rewriting_mode: QueryRewritingMode = "prompt_only_json"
    recommended_retry_profile: RetryProfile = "strict_json"
    schema_subset_profile: SchemaSubsetProfile = "none"
    client_validation_required: bool = False
    tool_choice_mode: ToolChoiceMode = "none"
    tool_result_channel: ToolResultChannel = "unsupported"
    tool_argument_encoding: ToolArgumentEncoding = "unsupported"
    request_timeout_seconds: int = 60
    min_temperature: float | None = None
    max_temperature: float | None = None
    max_output_tokens_param: str = "max_tokens"
    notes: list[str] = field(default_factory=list)
    research_sources: list[str] = field(default_factory=list)
    supports_site_restriction: bool = False
    site_restriction_mode: SiteRestrictionMode = "none"
    default_base_url: str | None = None
    supports_query_site_operator: QuerySiteSupport = "false"
    recommended_zoom_in_strategy: ZoomInStrategy = "query_side"
    query_param_path: str = "query"
    site_param_path: str | None = None
    site_value_type: SiteValueType = "string"
    supports_provider_side_num_results: bool = False
    num_results_mode: NumResultsMode = "unsupported"
    num_results_param: str | None = None
    result_collection_path: str | None = None
    field_candidates: dict[str, list[str]] = field(default_factory=dict)
    capability_confidence: CapabilityConfidence = "high"


@dataclass(slots=True)
class ResolvedProvider:
    engine: str
    provider_kind: ProviderKind
    component: Literal["llm", "search"]
    capability: ProviderCapability
    model: str | None = None
    protocol: ProviderProtocol | None = None
    base_url: str | None = None
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    http_proxy: str | None = None
    request_options: LLMRequestOptions = field(default_factory=LLMRequestOptions)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResolvedProviders:
    llm: ResolvedProvider
    search: ResolvedProvider


@dataclass(slots=True)
class RuntimeContext:
    request_id: str
    request: SearchRequest
    llm_provider: ResolvedProvider
    search_provider: ResolvedProvider
    warnings: list[WarningInfo] = field(default_factory=list)
    retry_budget: dict[str, int] = field(default_factory=dict)
    transport_context: dict[str, Any] = field(default_factory=dict)
    semaphore_limits: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False


@dataclass(slots=True)
class SearchResponse:
    request_id: str
    warnings: list[WarningInfo] = field(default_factory=list)
    metrics: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    answer: str | None = field(default=None, repr=False, compare=False)
    results: list[SimpleSearchResult | FinalSearchResult] | None = field(default=None, repr=False, compare=False)
    search_context: str | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_fields(cls, *, request_id: str, warnings: list[WarningInfo] | None = None, **fields: Any) -> "SearchResponse":
        response = cls(request_id=request_id, warnings=list(warnings or []))
        for name in ("metrics", "answer", "results", "search_context"):
            if name not in fields:
                delattr(response, name)
        for name, value in fields.items():
            setattr(response, name, value)
        return response

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"request_id": self.request_id}
        for name in ("metrics", "answer", "results", "search_context"):
            if hasattr(self, name):
                data[name] = getattr(self, name)
        data["warnings"] = self.warnings
        return data

    def dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(slots=True)
class SearchStreamEvent:
    type: SearchStreamEventType
    request_id: str
    metrics: dict[str, Any] | None = None
    text: str | None = None
    answer: str | None = None
    response: SearchResponse | None = None
    results: list[FinalSearchResult] | None = None
    search_context: str | None = None
    warnings: list[WarningInfo] = field(default_factory=list)

    @classmethod
    def from_fields(cls, *, type: SearchStreamEventType, request_id: str, warnings: list[WarningInfo] | None = None, **fields: Any) -> "SearchStreamEvent":
        event = cls(type=type, request_id=request_id, warnings=list(warnings or []))
        for name in ("metrics", "text", "answer", "response", "results", "search_context"):
            if name not in fields:
                delattr(event, name)
        for name, value in fields.items():
            setattr(event, name, value)
        return event

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type, "request_id": self.request_id}
        for name in ("metrics", "text", "answer", "response", "results", "search_context"):
            if hasattr(self, name):
                data[name] = getattr(self, name)
        data["warnings"] = self.warnings
        return data

    def dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(slots=True)
class UnifiedMessage:
    role: Literal["system", "user", "assistant"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UnifiedToolDefinition:
    name: str
    description: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UnifiedToolCall:
    id: str
    type: Literal["function"] = "function"
    name: str = ""
    arguments_json_text: str = "{}"
    arguments_object: dict[str, Any] | None = None
    provider_call_index: int | None = None
    raw_tool_call: Any = None


@dataclass(slots=True)
class UnifiedLLMRequest:
    task: Literal["query_rewriting", "answer_synthesis"]
    messages: list[UnifiedMessage]
    model: str
    provider: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None
    expect_json: bool = False
    json_schema: dict[str, Any] | None = None
    json_object: bool = False
    tools: list[UnifiedToolDefinition] = field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    request_timeout_seconds: int = 60
    trace_context: dict[str, Any] = field(default_factory=dict)

    def copy_with(self, **changes: Any) -> "UnifiedLLMRequest":
        return replace(self, **changes)


@dataclass(slots=True)
class UnifiedLLMResponse:
    provider: str | None = None
    model: str | None = None
    adapter_class: AdapterClass | None = None
    text: str | None = None
    json_payload: dict[str, Any] | list[Any] | None = None
    tool_calls: list[UnifiedToolCall] = field(default_factory=list)
    finish_reason: UnifiedFinishReason = "unknown"
    usage: dict[str, int | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UnifiedSearchRequest:
    task: Literal["zoomout_search", "zoomin_search"]
    query: str
    num_results: int
    site_restriction_mode: SiteRestrictionMode = "none"
    site_restriction_domain: str | None = None
    method: Literal["GET", "POST"] | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    request_timeout_seconds: int = 60
    trace_context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UnifiedSearchResult:
    title: str
    snippet: str
    url: str
    provider_result_index: int = 0
    raw_item: dict[str, Any] | str | None = None


@dataclass(slots=True)
class UnifiedSearchResponse:
    results: list[UnifiedSearchResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | list[Any] | str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    normalized_warnings: list[WarningInfo] = field(default_factory=list)
    debug_events: list[dict[str, Any]] = field(default_factory=list)
