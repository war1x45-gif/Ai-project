"""Interactive command-line chat: python -m simple_ai"""

import argparse

from .bot import ChatBot


def main():
    parser = argparse.ArgumentParser(description="Chat with SimpleAI.")
    parser.add_argument("--debug", action="store_true", help="show detected intent and confidence")
    args = parser.parse_args()

    bot = ChatBot()
    print("SimpleAI: Hi! Type a message, or 'quit' to exit.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            print("SimpleAI: Goodbye!")
            break
        if args.debug:
            intent, confidence = bot.classify(message)
            print(f"  [intent={intent} confidence={confidence:.2f}]")
        print(f"SimpleAI: {bot.respond(message)}")


if __name__ == "__main__":
    main()
