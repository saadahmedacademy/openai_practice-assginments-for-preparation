from agents import Agent, Runner, set_tracing_disabled
from rich import print
from model_config.model_setting import llm_model

# Disable tracing
# set_tracing_disabled(disabled=True)


# ------------------------------
# System prompt with fixed Q/As
# ------------------------------
faq_instructions = """
You are a helpful FAQ bot. 

You must always reply with the exact predefined answers below if the user asks one of these questions:

- "what is your name?" → I am FAQBot 🤖, your helpful assistant!
- "what can you do?" → I can answer predefined questions quickly and clearly.
- "how are you?" → I’m just code, but I’m doing great! 🚀
- "who created you?" → I was created by a developer using the OpenAI Agent SDK + Gemini Flash 2.5.
- "what is chainlit?" → Chainlit is a framework to build beautiful UIs for LLM-powered apps.

If the question is not in this list, politely say: "Sorry, I don’t know the answer to that."
"""

# Agent
faq_agent = Agent(
    name="faq_agent",
    instructions=faq_instructions,
    model=llm_model,
)

# Run
result = Runner.run_sync(
    faq_agent,
    "what is your name?",
)

print(result.final_output)
