# =============================================================================
# Mental Wellness Practice Suggester -- A LangGraph Learning Project
# =============================================================================
#
# This project teaches you how LangGraph works by building a mental wellness
# assistant that suggests personalized calming practices.
#
# WHAT THIS DOES:
# A user enters how they are feeling (e.g. "I feel stressed", "I can't sleep",
# "I feel anxious and overwhelmed"). The system runs 3 suggestion engines in
# PARALLEL (breathing, mindfulness, movement), then a decision node picks the
# best approach and routes to either a QUICK practice (under 5 minutes) or a
# DEEPER session (10-15 minutes) based on severity.
#
# LANGGRAPH CONCEPTS COVERED:
# 1. State Management (Pydantic) -- user feeling flows through the graph
# 2. Nodes -- each function does one job (suggest breathing, mindfulness, etc.)
# 3. Parallel Execution -- 3 suggestion nodes run at the same time
# 4. Fan-in -- waiting for all 3 suggestions before picking the best
# 5. Conditional Edges -- routing to quick vs deep based on severity
# 6. Graph Compilation -- turning the graph definition into a runnable app
#
# GRAPH STRUCTURE:
#
#   START
#     |
#   understand_mood
#     |
#     +---> suggest_breathing --------+
#     |                               |
#     +---> suggest_mindfulness ------+---> pick_best_practice
#     |                               |         |
#     +---> suggest_movement ---------+    (conditional)
#                                        /          \
#                                   quick?         deep?
#                                     |               |
#                               quick_practice   deep_practice
#                                     |               |
#                                    END             END
#
# HOW TO RUN:
#   python mental_wellness_graph.py
#
# DEPENDENCIES (same as requirements.txt):
#   langgraph, langchain-openai, python-dotenv, pydantic
#
# =============================================================================

import sys
import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


class InterviewCoach(BaseModel):
    job_role: str = ""
    specialist_suggestions: str = ""
    technical_suggestion: str = ""
    confidence_suggestion: str = ""
    needs_deep_prep: bool = False
    behavioral_suggestion : str = ""
    urgency_level : str = ""
    final_plan: str = ""
    messages: Annotated[list, operator.add] = []


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def job_role(state: InterviewCoach) -> dict:
    response = llm.invoke(
        f"You are a Interview Coach assistant you will help the students to crack their job. "
        f"A user says: '{state.job_role}'. "
        f"Acknowledge their job role and give a one-sentence empathetic response that shows you understand how they might be feeling about their job search. "
        f"Then classify the Inteview as LOW, MODERATE, or HIGH in one word on a new line like: Interview: MODERATE. "
    )
    return {
        "messages": [f"[job_role] {response.content}"]
    }


def specialist_suggestions(state: InterviewCoach) -> dict:
    response = llm.invoke(
        f"You are Technical Specailst in all the technologies. Where you will give the best suggestion to the user to crack the interview. "
        f"The user Says: '{state.job_role}'. "
        f"Guide the user step-by-step instructions (3-4 steps), and how long it takes. "
        f"Keep it under 5 sentences."
    )
    return {
        "specialist_suggestions": response.content,
        "messages": [f"[specialist_suggestions] Done"]
    }


def technical_suggestion(state: InterviewCoach) -> dict:
    response = llm.invoke(
        f"You are a technical interview preparation expert. "
        f"The user is preparing for a {state.job_role} interview. "
        f"Provide specific advice and strategies to help them succeed. "
        f"Include common questions they might face and how to approach them. "
        f"Keep it under 5 sentences."
    )
    return {
        "technical_suggestion": response.content,
        "messages": [f"[technical_suggestion] Done"]
    }

def confidence_suggestion(state: InterviewCoach) -> dict:
    response = llm.invoke(
        f"You are a confidence and self-esteem building expert. "
        f"The user is preparing for a {state.job_role} interview. "
        f"Provide specific advice and strategies to help them build confidence. "
        f"Include tips on how to handle nervousness and present themselves effectively. "
        f"Keep it under 5 sentences."
    )
    return {
        "confidence_suggestion": response.content,
        "messages": [f"[confidence_suggestion] Done"]
    }

def behavioral_suggestion (state: InterviewCoach) -> dict:
    response = llm.invoke(
        f"You are a behavioral interview preparation expert. "
        f"The user is preparing for a {state.job_role} interview. "
        f"Provide specific advice and strategies to help them succeed in behavioral questions. "
        f"Include examples of how to structure their responses using the STAR method (Situation, Task, Action, Result). "
        f"Keep it under 5 sentences."
    )
    return {
        "behavioral_suggestion ": response.content,
        "messages": [f"[behavioral_suggestion ] Done"]
    }


def pick_best_practice(state: InterviewCoach) -> dict:
    response = llm.invoke(
        f"You are a Interview Coach decision system. The user is preparing for a: '{state.job_role}'.\n\n"
        f"Here are three suggestions from specialists:\n\n"
        f"Technical:\n{state.technical_suggestion}\n\n"
        f"Confidence:\n{state.confidence_suggestion}\n\n"
        f"Behavioral:\n{state.behavioral_suggestion }\n\n"
        f"Decide: does this person need a QUICK practice guide (under 5 min, for interview as LOW/MODERATE/HIGH) "
        f"or a DEEPER Practice (10-15 min, for interview as HIGH)? "
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"needs_deep_prep": true/false, "reason": "one sentence explanation"}}'
    )
    try:
        result = json.loads(response.content)
        needs_deep = result["needs_deep_prep"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        needs_deep = False
        reason = "Could not parse decision, defaulting to quick practice."

    return {
        "needs_deep_prep": needs_deep,
        "urgency_level": reason,
        "messages": [f"[pick_best_practice] deep_session={needs_deep}"]
    }


def quick_practice(state: InterviewCoach) -> dict:
    response = llm.invoke(
        f"You are a Interview Coach. he user is preparing for a: '{state.job_role}'.\n\n"
        f"Based on these specialist suggestions, create a SHORT practice guide (under 5 minutes) that gives the most essential advice and steps to prepare for the interview, "
        f"that combines the best elements:\n\n"
        f"Technical:\n{state.technical_suggestion}\n\n"
        f"Confidence:\n{state.confidence_suggestion}\n\n"
        f"Behavioral:\n{state.behavioral_suggestion }\n\n"
        f"Format it as a simple numbered list of steps. "
        f"Keep it concise, encouraging, and easy to follow. End with a kind closing line."
    )
    return {
        "final_plan": f"QUICK Practice for the interview\n{'='*45}\n{response.content}",
        "messages": [f"[quick_practice] Generated quick prep plan based on urgency"]
    }


def deep_practice(state: InterviewCoach) -> dict:
    response = llm.invoke(
        f"You are a Interview Coach. The user is preparing for a: '{state.job_role}'.\n\n"
        f"Based on these specialist suggestions, create a DEEPER session (10-15 minutes) "
        f"that thoughtfully combines all three approaches:\n\n"
        f"Technical: {state.technical_suggestion}\n"
        f"Confidence: {state.confidence_suggestion}\n"
        f"Behavioral: {state.behavioral_suggestion }\n"
        f"Specialist: {state.specialist_suggestions}\n"
        f"Structure it in 4 phases: technical topics to review (data structures, system design), behavioral story prompts (STAR method, past wins), confidence-building habits (power pose, breathing before the call)"
        f"Give clear step-by-step instructions for each phase with timing. "
        f"Keep it concise and actionable. End with a kind closing message."
    )
    return {
        "final_plan": f"BEST Interview PRACTICE (10-15 min)\n{'='*45}\n{response.content}",
        "messages": [f"[deep_practice] Generated deep session"]
    }


def route_after_decision(state: InterviewCoach) -> str:
    if state.needs_deep_prep:
        return "deep"
    else:
        return "quick"


graph = StateGraph(InterviewCoach)

graph.add_node("specialist_suggestions", specialist_suggestions)
graph.add_node("technical_suggestion", technical_suggestion)
graph.add_node("confidence_suggestion", confidence_suggestion)
graph.add_node("behavioral_suggestion", behavioral_suggestion)
graph.add_node("pick_best_practice", pick_best_practice)
graph.add_node("quick_practice", quick_practice)
graph.add_node("deep_practice", deep_practice)

graph.add_edge(START, "specialist_suggestions")

graph.add_edge("specialist_suggestions", "technical_suggestion")
graph.add_edge("specialist_suggestions", "confidence_suggestion")
graph.add_edge("specialist_suggestions", "behavioral_suggestion")


graph.add_edge("technical_suggestion", "pick_best_practice")
graph.add_edge("confidence_suggestion", "pick_best_practice")
graph.add_edge("behavioral_suggestion", "pick_best_practice")

graph.add_conditional_edges(
    "pick_best_practice",
    route_after_decision,
    {
        "quick": "quick_practice",
        "deep": "deep_practice",
    }
)

graph.add_edge("quick_practice", END)
graph.add_edge("deep_practice", END)

app = graph.compile()


def Interview_Prep_Coach(User_Question: str):
    print("=" * 55)
    print("  Interview Coach Suggestions and guider")
    print(f"  You said: \"{User_Question}\"")
    print("=" * 55)

    result = app.invoke({
        "job_role": User_Question,
        "messages": [],
    })

    print("\n" + "=" * 55)
    print("  YOUR PERSONALIZED Interview PRACTICE SUGGESTION")
    print("=" * 55)
    print(f"\n{result['final_plan']}")

    print("\n" + "-" * 55)
    print("  MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Interview Specialist and Preparation Coach")
    print("=" * 55)
    print("\n  Tell me about the interview you're preparing for and I'll suggest a")
    print("  personalized preparation plan just for you.")
    print("  Type 'quit' to exit.\n")

    while True:
        User_Query = input("  How May I Help You? > ").strip()

        if User_Query.lower() in ("quit", "exit", "q"):
            print("\n  Take care of yourself. Goodbye!\n")
            break

        if not User_Query:
            continue

        Interview_Prep_Coach(User_Query)
        print("\n")
