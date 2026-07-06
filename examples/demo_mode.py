import asyncio

from zoom_search import search


async def main() -> None:
    response = await search(
        question="What hotels in Shenzhen have rooms with exercise bikes?",
        demo_mode=True,
        output_mode="answer_with_sources",
        seed=7,
    )
    print("Answer:\n", response.answer)
    print("\nResults:")
    for index, result in enumerate(response.results, start=1):
        print(f"{index}. {result.title} - {result.url}")
    print("\nMetrics:\n", response.metrics)


if __name__ == "__main__":
    asyncio.run(main())
