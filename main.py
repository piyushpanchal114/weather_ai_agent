import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

API_KEY = os.environ.get("API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def get_weather(city: str):
    resp = requests.get(f"https://wttr.in/{city}?format=%C+%t")
    if resp.status_code == 200:
        return f"The weather in {city} is {resp.text}"
    return "Something went wrong."


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes the city name as input and returns the current weather of the city."  # noqa: E501
    },
}

sys_prompt = """
    You are a helpful AI assistant who is specialised in resolving user query.
    You work on start, plan, action and observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevent tool from the available tools. and based on the tool selection you perform an action to call the tool.
    wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the strict JSON format for output.
    - Always perform one step at a time and wait for the next input.
    - Carefully analyze the user query.

    Output JSON Format:
    {{
        "step": string,
        "content": string,
        "function": The name of the function if the step is action,
        "input": The input parameter of the function
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city.

    Example:
    User Query: What is the weather of the chandigarh?
    output: {{"step": "plan", "content": "The user is intrested in weather data of new york."}}
    output: {{"step": "plan", "content": "From the available tools I should call get_weather."}}
    output: {{"step": "action", "function": "get_weather", "input": "chandigarh"}}
    output: {{"step": "observe", "output": "21 Degree Celcius"}}
    output: {{"step": "output", "content": "The weather of chandigarh seems to be 21 degree celcius"}}
"""  # noqa: E501


messages = [
    {"role": "system", "content": sys_prompt}
]


def ask_llm(messages):
    response = client.chat.completions\
        .create(model="gemini-2.0-flash",
                response_format={"type": "json_object"},
                messages=messages)
    return response.choices[0].message.content


def ask_agent(messages: list[str]):
    while True:
        resp = ask_llm(messages)
        res_dict = json.loads(resp)
        messages.append({"role": "assistant", "content": json.dumps(res_dict)})

        if res_dict.get("step") == "plan":
            print(f"Plan: {res_dict.get('content')}")
            continue
        if res_dict.get("step") == "action":
            tool_name = res_dict.get("fn")
            tool_input = res_dict.get("input")

            if available_tools.get(tool_name):
                output = available_tools["tool_name"]["fn"](tool_input)
                messages.append(
                    {"role": "assistant", "content": json.dumps(
                        {"step": "observe", "output": output})})
                continue

        if res_dict.get("step") == "output":
            print(f"bot: {res_dict.get('content')}")
            break


if __name__ == "__main__":
    while True:
        user_query = input("> ")
        messages.append({"role": "user", "content": user_query})
        ask_agent(messages)
