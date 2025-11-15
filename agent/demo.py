from __future__ import annotations

import asyncio

from .factory import ask


async def main() -> None:
    print("Glass LangGraph ReAct agent demo")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not user_input:
            print("Please enter a prompt or Ctrl+C to exit.")
            continue

        response = await asyncio.to_thread(ask, user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    asyncio.run(main())
