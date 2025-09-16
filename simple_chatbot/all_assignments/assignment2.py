from agents import Agent, Runner, set_tracing_disabled, function_tool
from rich import print
from model_config.model_setting import llm_model


# -------------------
# Step 1: Define tool
# -------------------
@function_tool
def add(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b

@function_tool
def multiply(a: int, b: int) -> int:
    """Return the product of two numbers."""
    return a * b

# ---------------------
# Step 2: Agent setup
# ---------------------
math_agent = Agent(
    name="math_agent",
    instructions="""
    You are a helpful math assistant.
    If the user asks a math question (like addition or multiplication), 
    use the provided tools to calculate the answer. 
    Otherwise, reply normally.
    """,
    model=llm_model,
    tools=[add, multiply],
)

# ---------------------
# Step 3: Run examples
# ---------------------
questions = [
    "What is 5 + 7?",
    "Can you multiply 6 and 9?",
    "If I add 20 and 30, what do I get?",
]

for q in questions:
    result = Runner.run_sync(math_agent, q)
    print(f"\n[bold green]Q:[/bold green] {q}")
    print(f"[bold yellow]A:[/bold yellow] {result.final_output}")
