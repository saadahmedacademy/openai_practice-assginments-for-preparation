import requests
from agents import Agent, Runner, set_tracing_disabled, function_tool
from rich import print
from model_config.model_setting import llm_model
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
# -----------------------
# Step 1: Weather function
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


# -----------------------
# Step 2: Agent setup
# -----------------------
weather_agent = Agent(
    name="weather_agent",
    instructions="""
    You are a weather assistant.
    If the user asks about the weather in a city, 
    use the weather tool to fetch the answer.
    Otherwise, respond normally.
    """,
    model=llm_model,
    tools=[get_weather],
)

# -----------------------
# Step 3: Run examples
# -----------------------
questions = [
    "What’s the weather in Karachi?",
    "Tell me the weather in Lahore.",
    "How’s the weather in New York?",
]

for q in questions:
    result = Runner.run_sync(weather_agent, q)
    print(f"\n[bold green]Q:[/bold green] {q}")
    print(f"[bold yellow]A:[/bold yellow] {result.final_output}")


# -----------------------
# Step 4: Run This command in terminal
# -----------------------
# uv run -m all_assignments.assignment3

# -------------
# Final Output
# -------------

# Q: What’s the weather in Karachi?
# A: The weather in Karachi, Pakistan is 29.3°C with Partly cloudy.

# Q: Tell me the weather in Lahore.
# A: The weather in Lahore, Pakistan is 25.0°C with Overcast.

# Q: How’s the weather in New York?
# A: The weather in New York, United States of America is 22.2°C with Partly cloudy.
