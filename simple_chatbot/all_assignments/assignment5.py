"""
Assignment 6: Smart Customer Support Bot
----------------------------------------
Features:
✅ BotAgent answers FAQs & fetches order status
✅ HumanAgent handles escalation
✅ Guardrail to block/rephrase offensive inputs
✅ Function tools with is_enabled + failure_error_function
✅ Agent handoff on complex queries or negative sentiment
✅ ModelSettings: tool_choice, metadata
✅ Logging for traceability
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any
from pydantic import BaseModel
from model_config.model_setting import llm_model
from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    function_tool,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    ModelSettings,
    AgentBase,
    TResponseInputItem,
)

# ------------------------
# Context
# ------------------------


class SentimentOutput(BaseModel):
    is_negative: bool
    reasoning: str


@dataclass
class SupportContext:
    customer_id: str
    faq_answers: Dict[str, str] = field(default_factory=lambda: {
        "what is your name?": "I am FAQBot 🤖, your helpful assistant!",
        "what can you do?": "I can answer predefined questions quickly and clearly.",
        "how are you?": "I’m just code, but I’m doing great! 🚀",
        "who created you?": "I was created using the OpenAI Agent SDK + Gemini Flash 2.5.",
        "what is chainlit?": "Chainlit is a framework to build beautiful UIs for LLM-powered apps.",
    })
    orders: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "ORD-1001": {"status": "Shipped", "eta": "3-5 days", "carrier": "DHL"},
        "ORD-1002": {"status": "Processing", "eta": "TBD", "carrier": None},
    })

    last_guardrail_output: SentimentOutput | None | Any = None


# ------------------------
# is_enabled Functions
# ------------------------
def faq_is_enabled(ctx: RunContextWrapper[SupportContext], agent: AgentBase[Any]) -> bool:
    """Enable FAQ tool always."""
    return True


def order_status_is_enabled(ctx: RunContextWrapper[SupportContext], agent: AgentBase[Any]) -> bool:
    """Enable order status tool always (later can add smarter conditions)."""
    return True


# ------------------------
# Custom Error Function
# ------------------------
def order_status_error(context: RunContextWrapper[SupportContext], error: Exception) -> str:
    """Custom error message if tool fails."""
    print(f"[ERROR] Order status tool failed: {error}")
    return f"⚠️ Order not found. Please double-check your order ID."


# ------------------------
# Function Tools
# ------------------------
@function_tool(is_enabled=faq_is_enabled,strict_mode=False)
def faq_tool(question: str, context: SupportContext) -> str:
    return context.faq_answers.get(question.lower(), "❓ Sorry, I don’t know that one.")


@function_tool(
    is_enabled=order_status_is_enabled,
    failure_error_function=order_status_error,
    strict_mode=False
)
def get_order_status(order_id: str, context: SupportContext) -> str:
    order = context.orders.get(order_id)
    if not order:
        raise ValueError("Order not found")

    return f"📦 Order {order_id}: {order['status']} | ETA: {order['eta']} | Carrier: {order['carrier']}"

# ------------------------
# Guardrail
# ------------------------


guardrail_agent = Agent[SupportContext](
    name="GuardrailAgent",
    instructions="Classify if the user input is offensive, rude, or negative. "
                 "Return is_negative=True if it should be blocked or escalated.",
    output_type=SentimentOutput,
    model=llm_model 
)


@input_guardrail
async def sentiment_guardrail(
    ctx: RunContextWrapper[SupportContext],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    # Run guardrail agent
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    # Store the reasoning for later use
    ctx.context.last_guardrail_output = result.final_output  

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_negative,
    )

# ------------------------
# Agents
# ------------------------

# To solve the complex query or negative sentiment
HumanAgent = Agent[SupportContext](
    name="HumanAgent",
    instructions=(
        "You are a human support agent. Be empathetic and resolve complex issues. "
        "You may receive context from the guardrail (negative sentiment classification). "
        "Use this information to adapt your tone and response."
    ),
    model=llm_model,
    handoffs=[],  # will be set after BotAgent is created
    handoff_description="Escalate to me when the issue requires empathy, judgment, or cannot be solved by tools."
)


# To solve the simple query and to use the tools to get the order status and order_id data
BotAgent = Agent[SupportContext](
    name="BotAgent",
    instructions="You are a helpful support bot. Use faq_tool for FAQs and get_order_status for orders message ,if you don't find the order_id  so please answer with appropriatly polityly . "
                 "Escalate to the HumanAgent if the query is too complex or requires empathy.",
    model=llm_model,
    tools=[faq_tool, get_order_status],
    handoffs=[HumanAgent],
    input_guardrails=[sentiment_guardrail],
    model_settings=ModelSettings(tool_choice="auto"),
    handoff_description="Escalate to me when the query is structured, factual, or needs tool usage (FAQ/order lookup)."
)

HumanAgent.handoffs.append(BotAgent)

# ------------------------
# Logging Helper
# ------------------------
def log_event(event: str):
    print(f"[LOG] {event}")


# ------------------------
# Context
# ------------------------
context = SupportContext(customer_id="CUST-1234")

# ------------------------
# Runner Demo
# ------------------------
async def demo():

    # User enters the query once
    user_query = input("Enter your message: ")

    try:
        result = await Runner.run(BotAgent, user_query, context=context)
        print("Bot Response:", result.final_output)

    except InputGuardrailTripwireTriggered:
        log_event("Guardrail triggered → escalating to HumanAgent")
        guardrail_reason = context.last_guardrail_output

        escalation_message = (
        f"Escalated query: {user_query}\n"
        f"Guardrail reasoning: {guardrail_reason.reasoning}"
        )

        result = await Runner.run(HumanAgent, escalation_message, context=context)
        print("HumanAgent Response:", result.final_output)


if __name__ == "__main__":
    asyncio.run(demo())
