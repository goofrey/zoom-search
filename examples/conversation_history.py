import asyncio

from zoom_search import search


async def main() -> None:
    response = await search(
        question="What about hotels with in-room fitness equipment?",
        previous_conversation=[
            "I am planning a business trip to Shenzhen.",
            "I prefer hotels with wellness facilities.",
        ],
        demo_mode=True,
        output_mode="answer_with_sources",
        seed=7,
    )
    print("Answer:\n", response.answer)
    print("\nSearch context:\n", response.search_context)


if __name__ == "__main__":
    asyncio.run(main())
