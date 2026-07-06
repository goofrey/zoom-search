import asyncio

from zoom_search import astream_search


async def main() -> None:
    async for event in astream_search(
        question="What hotels in Shenzhen have rooms with exercise bikes?",
        demo_mode=True,
        output_mode="answer_with_sources",
        seed=7,
    ):
        if event.type == "answer_delta":
            print(event.text, end="")
        elif event.type == "completed":
            print(f"\n\nRequest ID: {event.response.request_id}")


if __name__ == "__main__":
    asyncio.run(main())
