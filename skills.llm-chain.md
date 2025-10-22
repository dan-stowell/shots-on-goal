---
name: LLM Chain with Tools
description: Create conversations with tool execution using the llm library. Chain tool calls together, pass functions or toolboxes, control execution limits, and handle errors. Use when working with LLM tool calling, agentic workflows, or multi-step reasoning with Python.
---

# LLM Chain with Tools

## Quick start

```python
import llm

def upper(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

model = llm.get_model("gpt-4o-mini")
response = model.chain("Convert hello to upper", tools=[upper])
print(response.text())
```

The `.text()` method automatically executes all tool calls and returns the final result.

## Core concepts

**model.chain()** - Creates new conversation each time (stateless)
**conversation.chain()** - Preserves history across calls (stateful)
**chain_limit** - Maximum tool call iterations (no limit by default; set explicitly to prevent runaway tool loops)

## Conversation with tools

```python
import llm

def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

model = llm.get_model("gpt-4o-mini")

# Create conversation with tools
conversation = model.conversation(
    tools=[multiply, add],
    chain_limit=5
)

# First call - conversation remembers this
result1 = conversation.chain("What is 12 times 34?").text()
print(result1)

# Second call - has access to previous context
result2 = conversation.chain("Now add 100 to that result").text()
print(result2)
```

## Toolbox pattern for shared state

Use `llm.Toolbox` when tools need to share state:

```python
import llm

class Memory(llm.Toolbox):
    def __init__(self):
        self._memory = {}

    def set(self, key: str, value: str):
        """Set a value by key."""
        self._memory[key] = value

    def get(self, key: str) -> str:
        """Get a value by key."""
        return self._memory.get(key, "")

    def list_keys(self) -> str:
        """List all keys."""
        return ", ".join(self._memory.keys())

model = llm.get_model("gpt-4o-mini")
memory = Memory()

conversation = model.conversation(tools=[memory])
conversation.chain("Set name to Alice and age to 30").text()
conversation.chain("What name did I store?").text()

print(memory._memory)  # {'name': 'Alice', 'age': '30'}
```

**Key points**:
- Methods starting with `_` are private (not exposed as tools)
- Public methods become tools automatically
- Instance variables persist across tool calls
- Docstrings become tool descriptions

## Before/after call hooks

Monitor or control tool execution:

```python
import llm
from typing import Optional

def before_call(tool: Optional[llm.Tool], tool_call: llm.ToolCall):
    """Called before each tool execution."""
    print(f"About to call: {tool.name if tool else 'unknown'}")
    print(f"Arguments: {tool_call.arguments}")

    # Cancel dangerous operations
    if tool and "delete" in tool.name.lower():
        raise llm.CancelToolCall("Deletion not allowed")

def after_call(tool: llm.Tool, tool_call: llm.ToolCall, tool_result: llm.ToolResult):
    """Called after each tool execution."""
    print(f"Completed: {tool.name}")
    print(f"Result: {tool_result.output}")

class FileSystem(llm.Toolbox):
    def __init__(self):
        self.access_log = []

    def read_file(self, filename: str) -> str:
        """Read a file."""
        return f"Contents of {filename}"

model = llm.get_model("gpt-4o-mini")
fs = FileSystem()

# Set hooks on conversation
conversation = model.conversation(
    tools=[fs],
    before_call=before_call,
    after_call=after_call
)

result = conversation.chain("Read the config.json file").text()

# Or set hooks per call
result = conversation.chain(
    "Read the data.csv file",
    before_call=before_call,
    after_call=after_call
).text()
```

## Chain limit control

```python
import llm

def step1(x: int) -> int:
    """First step."""
    return x * 2

def step2(x: int) -> int:
    """Second step."""
    return x + 10

model = llm.get_model("gpt-4o-mini")

# Set default limit on conversation
conversation = model.conversation(
    tools=[step1, step2],
    chain_limit=3
)

# Uses conversation's limit (3)
result = conversation.chain("Process the number 5").text()

# Override with higher limit for this call
result = conversation.chain(
    "Do a complex calculation",
    chain_limit=10
).text()
```

When chain_limit is exceeded, `ValueError` is raised with message "Chain limit of N exceeded."

## Error handling

```python
import llm
from llm.errors import ModelError, NeedsKeyException

def safe_chain(prompt: str, tools=None, conversation=None, chain_limit=10):
    """Execute chain with comprehensive error handling."""
    try:
        if conversation:
            result = conversation.chain(
                prompt,
                tools=tools,
                chain_limit=chain_limit
            ).text()
        else:
            # Create conversation with limit, then chain
            model = llm.get_model("gpt-4o-mini")
            conv = model.conversation(tools=tools, chain_limit=chain_limit)
            result = conv.chain(prompt).text()
        return result

    except NeedsKeyException as e:
        print(f"API key required: {e}")
        return None

    except ValueError as e:
        if "Chain limit" in str(e):
            print(f"Too many tool calls, retrying with higher limit...")
            # Retry with higher limit
            return safe_chain(prompt, tools, conversation, chain_limit * 2)
        elif "does not support tools" in str(e):
            print(f"Model doesn't support tools: {e}")
        else:
            print(f"Validation error: {e}")
        return None

    except Exception as e:
        # API errors (rate limits, context window, etc.)
        error_msg = str(e).lower()
        if "maximum context length" in error_msg:
            print("Context window exceeded")
        elif "rate limit" in error_msg:
            print("Rate limit hit")
        else:
            print(f"Unexpected error: {type(e).__name__}: {e}")
        return None

# Usage
result = safe_chain("What is 2+2?", tools=[...])
```

### Common errors

**ValueError scenarios**:
- Chain limit exceeded
- Model doesn't support tools
- Model doesn't support schemas
- Tool has no implementation
- Invalid tool definition
- Lambda function without name
- Attachment type not supported

**ModelError/NeedsKeyException**:
- Missing API key
- Model configuration issues

**API provider errors** (bubble up from OpenAI, Anthropic, etc.):
- Context window exceeded
- Rate limit exceeded
- Network errors

## Advanced toolbox features

### Setup with prepare()

```python
import llm

class DatabaseTools(llm.Toolbox):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.query_count = 0

    def prepare(self):
        """Called once before first tool use."""
        # Setup database connection
        print(f"Connecting to {self.db_path}")
        # self.connection = connect(self.db_path)

    def query(self, sql: str) -> str:
        """Execute SQL query."""
        self.query_count += 1
        return f"Query result for: {sql}"

    def stats(self) -> str:
        """Get query statistics."""
        return f"Executed {self.query_count} queries"

model = llm.get_model("gpt-4o-mini")
db = DatabaseTools("/path/to/db")

conversation = model.conversation(tools=[db])
# prepare() is called automatically on first tool use
result = conversation.chain("Query the users table").text()
```

### Multiple tools and toolboxes

```python
import llm

def format_date(date_str: str) -> str:
    """Format a date string."""
    return f"Formatted: {date_str}"

class Calculator(llm.Toolbox):
    def add(self, x: int, y: int) -> int:
        """Add numbers."""
        return x + y

    def multiply(self, x: int, y: int) -> int:
        """Multiply numbers."""
        return x * y

class TextTools(llm.Toolbox):
    def upper(self, text: str) -> str:
        """Convert to uppercase."""
        return text.upper()

    def lower(self, text: str) -> str:
        """Convert to lowercase."""
        return text.lower()

model = llm.get_model("gpt-4o-mini")
calc = Calculator()
text = TextTools()

# Mix functions and toolboxes
conversation = model.conversation(
    tools=[format_date, calc, text]
)

result = conversation.chain(
    "Add 5 and 10, then convert the result to uppercase"
).text()
```

## Async usage

```python
import llm
import asyncio

class Memory(llm.Toolbox):
    def __init__(self):
        self._data = {}

    async def prepare_async(self):
        """Async setup hook."""
        print("Async preparation complete")

    def store(self, key: str, value: str):
        """Store a value."""
        self._data[key] = value

async def before_call(tool, tool_call):
    """Async before hook."""
    print(f"Before: {tool.name if tool else 'unknown'}")
    await asyncio.sleep(0.01)

async def after_call(tool, tool_call, tool_result):
    """Async after hook."""
    print(f"After: {tool.name}")
    await asyncio.sleep(0.01)

async def main():
    model = llm.get_async_model("gpt-4o-mini")
    memory = Memory()

    conversation = model.conversation(
        tools=[memory],
        before_call=before_call,
        after_call=after_call,
        chain_limit=5
    )

    response = conversation.chain("Store my name as Alice")
    text = await response.text()
    print(text)

asyncio.run(main())
```

## Best practices

1. **Use conversations for stateful workflows**: When you need history or shared state
2. **Set chain_limit explicitly**: Prevents runaway tool loops (no default limit without it)
3. **Wrap chains in error handling**: Catch ValueError for limits, Exception for API errors
4. **Use toolboxes for related tools**: Group tools that share state or configuration
5. **Provide clear docstrings**: They become tool descriptions for the LLM
6. **Use hooks for monitoring**: Log tool calls, validate arguments, prevent dangerous operations

## Troubleshooting

**Chain stops unexpectedly**:
- Check if chain_limit was hit (will raise ValueError)
- Verify tools return appropriate values (not None unless intended)

**Tool not being called**:
- Check docstring is clear and descriptive
- Verify tool name doesn't start with underscore
- Ensure model supports tools (not all do)

**"does not support tools" error**:
- Model doesn't have tool calling capability
- Switch to a model that supports tools (gpt-4o, claude-3-5-sonnet, etc.)
