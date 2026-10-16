# SimpleAI

A tiny chatbot written in plain Python (no dependencies). It learns from example
phrases in `simple_ai/intents.json` using a **Naive Bayes classifier**, figures out
what you mean (the "intent"), and replies.

## Run it

```bash
python3 -m simple_ai          # start chatting
python3 -m simple_ai --debug  # also show detected intent + confidence
```

```
SimpleAI: Hi! Type a message, or 'quit' to exit.
You: tell me a joke
SimpleAI: I would tell you a UDP joke, but you might not get it.
```

## How it works

1. Your message is split into lowercase words (`tokenize`).
2. For each intent, the classifier computes how likely those words are, based on
   how often they appeared in that intent's training patterns (with Laplace
   smoothing so unseen words don't zero things out).
3. The most likely intent wins. If none of your words were ever seen, or the
   confidence is below the threshold, the bot says it didn't understand.

## Teach it something new

Add an entry to `simple_ai/intents.json`:

```json
{
  "tag": "weather",
  "patterns": ["what's the weather", "is it raining", "will it be sunny"],
  "responses": ["I can't see outside, but I hope it's nice!"]
}
```

More (and more varied) patterns make it smarter.

## Use it from code

```python
from simple_ai import ChatBot

bot = ChatBot()
print(bot.classify("hello there"))  # ('greeting', 0.6...)
print(bot.respond("thank you!"))
```

## Tests

```bash
python3 -m unittest -v
```
