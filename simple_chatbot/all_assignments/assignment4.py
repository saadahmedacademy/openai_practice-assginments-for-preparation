import requests
from agents import Agent, Runner, function_tool,set_trace_processors
from rich import print
from model_config.model_setting import llm_model
import os
import asyncio

# -----------------------
# Math function
# -----------------------
@function_tool
def add(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b

# -----------------------
# Weather function
# -----------------------
@function_tool
def get_weather(city: str) -> str:
    """Fetch real weather info for a city using WeatherAPI."""
    API_KEY = os.getenv("WEATHER_API_KEY")  # your key
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Extract needed info
        temp = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        city_name = data["location"]["name"]
        country = data["location"]["country"]

        return f"The weather in {city_name}, {country} is {temp}°C with {condition}."
    except Exception as e:
        return f"Sorry, I could not fetch weather for {city}. Error: {e}"

async def main():
    
# -----------------------
#  Agent setup
# -----------------------
 multi_tool_agent = Agent(
    name="multi_tool_agent",
    instructions="""
    You are a versatile assistant.
    - If the user asks a math question (like addition), use the add tool.
    - If the user asks about the weather, use the weather tool.
    - Otherwise, answer normally.
    """,
    model=llm_model,
    tools=[add, get_weather],
)


# -----------------------
#  Run examples
# -----------------------
 questions = [
    "What is 15 + 27?",
    "What’s the weather in Karachi?",
    "Add 200 and 300.",
    "Tell me the weather in Lahore.",
]
    
 for q in questions:
    result = Runner.run_sync(multi_tool_agent, q)
    print(f"\n[bold green]Q:[/bold green] {q}")
    print(f"[bold yellow]A:[/bold yellow] {result.final_output}")


asyncio.run(main())


# -----------------------
#  Run This command in terminal
# -----------------------
# uv run -m all_assignments.assignment3

# -------------
# Final Output
# -------------


# Q: What is 15 + 27?
# A: The sum of 15 and 27 is 42.

# Q: What’s the weather in Karachi?
# A: The weather in Karachi, Pakistan is 29.0°C with Mist.

# Q: Add 200 and 300.
# A: The sum of 200 and 300 is 500.

# Q: Tell me the weather in Lahore.
# A: The weather in Lahore, Pakistan is 25.1°C with Overcast.