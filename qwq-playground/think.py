import os
import requests
import warnings
import argparse
from typing import Dict

# Configuration for reasoning and instruction models
REASONING_CONFIG = {
    "model": "qwq",
    "temperature": 1.2,
    "min_p": 0.1,
    "system_prompt": """You are a helpful and harmless assistant. You should think step-by-step."""
}

INSTRUCTION_CONFIG = {
    "model": "qwen2.5:32b",
    "temperature": 0.2,
    "min_p": 0.0,
    "system_prompt": """You are a helpful assistant that structures thinking processes.
Your task is to analyze a response and structure it into two clear sections:
1. A thinking process section wrapped in <think></think> tags
2. A final answer section wrapped in <answer></answer> tags

Respect the original content, DO NOT supplement it. Ensure it's well-structured in these sections."""
}

# Shared configuration
SHARED_CONFIG = {
    "ollama_endpoint": "http://localhost:11434"
}

def query_reasoning(prompt: str, config: Dict = REASONING_CONFIG) -> str:
    """
    Query the reasoning model for initial thoughts.
    """
    if os.getenv("PROMPT_DEBUG"):
        print(f"🤖 Reasoning system prompt:\n{config['system_prompt']}")
        print(f"🤖 User prompt:\n{prompt}")
    
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": config["temperature"],
            "min_p": config["min_p"]
        }
    }
    
    response = requests.post(f"{SHARED_CONFIG['ollama_endpoint']}/api/chat", json=payload)
    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "")
    
    return f"Error: {response.status_code} - {response.text}"

def query_instruction(content: str, config: Dict = INSTRUCTION_CONFIG) -> str:
    """
    Query the instruction model to structure the output.
    """
    if os.getenv("PROMPT_DEBUG"):
        print(f"🤖 Instruction system prompt:\n{config['system_prompt']}")
        print(f"🤖 Content to structure:\n{content}")
    
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": f"Please structure this response into thinking and answer sections:\n\n{content}"}
        ],
        "stream": False,
        "options": {
            "temperature": config["temperature"],
            "min_p": config["min_p"]
        }
    }
    
    response = requests.post(f"{SHARED_CONFIG['ollama_endpoint']}/api/chat", json=payload)
    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "")
    
    return f"Error: {response.status_code} - {response.text}"

def extract_sections(text: str) -> tuple[str, str]:
    """
    Extract the content between <think> and <answer> tags.
    Returns a tuple of (thinking_content, answer_content).
    If no think block is found, issues a warning.
    """
    thinking_block = ""
    answer_block = ""
    
    if "<think>" in text and "</think>" in text:
        thinking_block = text.split("<think>")[1].split("</think>")[0].strip()
    else:
        warnings.warn("No <think></think> block found in the response.")
    
    if "<answer>" in text and "</answer>" in text:
        answer_block = text.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        warnings.warn("No <answer></answer> block found in the response.")
        answer_block = text.strip()
    
    return thinking_block, answer_block

def structured_thinking(
    prompt: str,
    reasoning_config: Dict = REASONING_CONFIG,
    instruction_config: Dict = INSTRUCTION_CONFIG
) -> tuple[str, str, str]:
    """
    Generate a structured thinking response using a two-stage process:
    1. Get initial reasoning from the reasoning model
    2. Have the instruction model structure the output
    Returns a tuple of (thinking_content, final_output).
    """
    # Stage 1: Get reasoning from primary model
    initial_response = query_reasoning(prompt, reasoning_config)

    # Stage 2: Have instruction model structure the output
    structured_response = query_instruction(initial_response, instruction_config)

    # Extract the structured sections
    thinking_block, answer = extract_sections(structured_response)
    
    # Format final output as <think> block followed by answer
    return initial_response, thinking_block, answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate structured thinking responses using Ollama')
    parser.add_argument('--prompt', '-p', type=str, help='Input prompt for the model')
    args = parser.parse_args()

    # Use provided prompt or fall back to example
    test_prompt = args.prompt or "What's one way to help reduce my impact on microplastics?"
    
    initial_response, thinking, output = structured_thinking(test_prompt)
    
    print("🧠 Raw Reasoning:")
    print("-" * 40)
    print(initial_response)
    print("\n🤔 Extracted Thinking:")
    print("-" * 40)
    print(thinking)
    print("\n📝 Extracted Answer:")
    print("-" * 40)
    print(output) 