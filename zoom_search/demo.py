"""Deterministic demo providers and fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import AsyncIterator

from zoom_search.models import UnifiedLLMRequest
from zoom_search.models import UnifiedLLMResponse
from zoom_search.models import UnifiedSearchRequest
from zoom_search.models import UnifiedSearchResponse
from zoom_search.models import UnifiedSearchResult


def create_demo_llm_provider(*, seed: int | None = None) -> "DemoLLMProvider":
    return DemoLLMProvider(seed=seed)


def create_demo_search_provider(*, seed: int | None = None) -> "DemoSearchProvider":
    return DemoSearchProvider(seed=seed)


@dataclass(slots=True)
class DemoLLMProvider:
    seed: int | None = None

    async def generate(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        if request.task == "query_rewriting":
            return UnifiedLLMResponse(
                json_payload=_default_rewrite_payload(),
                usage={"input_tokens": 120, "output_tokens": 80, "total_tokens": 200, "reasoning_tokens": 0, "cached_input_tokens": 0},
                warnings=[],
                provider_metadata={"seed": self.seed},
            )
        if request.expect_json:
            return UnifiedLLMResponse(
                json_payload={
                    "answer": "Shenzhen hotels with in-room exercise bikes appear in several travel sources.[1][2]",
                    "sources": [
                        {"id": "[1]", "url": "https://www.trip.com/hotels/shenzhen-fitness-bike-suite"},
                        {"id": "[2]", "url": "https://www.booking.com/shenzhen/spin-room"},
                    ],
                },
                usage={"input_tokens": 240, "output_tokens": 110, "total_tokens": 350, "reasoning_tokens": 0, "cached_input_tokens": 0},
                warnings=[],
                provider_metadata={"seed": self.seed},
            )
        return UnifiedLLMResponse(
            text=(
                "Shenzhen hotels with in-room exercise bikes appear in several travel sources.[1][2]\n\n"
                "Sources:\n"
                "[1] https://www.trip.com/hotels/shenzhen-fitness-bike-suite\n"
                "[2] https://www.booking.com/shenzhen/spin-room"
            ),
            usage={"input_tokens": 240, "output_tokens": 110, "total_tokens": 350, "reasoning_tokens": 0, "cached_input_tokens": 0},
            warnings=[],
            provider_metadata={"seed": self.seed},
        )

    async def stream_generate(self, request: UnifiedLLMRequest) -> AsyncIterator[dict[str, Any]]:
        response = await self.generate(request)
        if response.text:
            yield {"event": "token", "delta_text": response.text}
        yield {"event": "done", "response": response}

    async def generate_json(self, *, prompt: str, context, json_strategy: str) -> object:
        return await self.generate(
            UnifiedLLMRequest(
                task="query_rewriting",
                messages=[],
                model="demo",
            )
        )


@dataclass(slots=True)
class DemoSearchProvider:
    seed: int | None = None

    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        results = _SEARCH_FIXTURES.get(request.query, [])
        return UnifiedSearchResponse(
            results=[
                UnifiedSearchResult(
                    title=item["title"],
                    snippet=item["snippet"],
                    url=item["url"],
                    provider_result_index=index,
                )
                for index, item in enumerate(results)
            ],
            warnings=[],
            provider_metadata={"seed": self.seed},
        )


def _default_rewrite_payload() -> dict:
    return {
        "previous_conversation": [],
        "search_groups": [
            {
                "group": 1,
                "original_input": "What hotels in Shenzhen have rooms with exercise bikes?",
                "comparison_question": "",
                "split_question": "What hotels in Shenzhen have rooms with exercise bikes?",
                "main_term": "Shenzhen hotel",
                "key_noun": "exercise bike room",
                "alias1": "spin bike room",
                "alias2": "fitness guest room",
                "query1": "Shenzhen hotel exercise bike room",
                "query2": "Shenzhen hotel spin bike room fitness guest room",
            }
        ],
    }


_SEARCH_FIXTURES = {
    "Shenzhen hotel exercise bike room": [
        {
            "title": "Trip.com Shenzhen Fitness Bike Suite",
            "snippet": "A Shenzhen hotel room with an in-room exercise bike.",
            "url": "https://www.trip.com/hotels/shenzhen-fitness-bike-suite",
        },
        {
            "title": "Invalid Listing",
            "snippet": "Missing URL normalization edge case.",
            "url": "not-a-valid-url",
        },
    ],
    "Shenzhen hotel spin bike room fitness guest room": [
        {
            "title": "Booking Shenzhen Spin Room",
            "snippet": "Booking page mentioning a spin bike room in Shenzhen.",
            "url": "https://www.booking.com/shenzhen/spin-room",
        },
        {
            "title": "Trip duplicate",
            "snippet": "Duplicate domain to verify deterministic deduplication.",
            "url": "https://m.trip.com/hotels/shenzhen-fitness-bike-suite",
        },
    ],
    "What hotels in Shenzhen have rooms with exercise bikes?": [
        {
            "title": "Trip.com Detailed Room Page",
            "snippet": "Room details with exercise bike amenities.",
            "url": "https://www.trip.com/hotels/shenzhen-fitness-bike-suite/details",
        }
    ],
}
