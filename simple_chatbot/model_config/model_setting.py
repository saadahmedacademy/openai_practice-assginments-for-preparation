from dotenv import load_dotenv, find_dotenv
import os
from agents import AsyncOpenAI, OpenAIChatCompletionsModel 


# Load environment variables
load_dotenv(find_dotenv())

# ✅ Get the GEMINI_API_KEY from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Make sure it's set in your .env file.")

# External client
externel_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Chat model
llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=externel_client
)
