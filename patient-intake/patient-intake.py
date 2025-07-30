import json
import os
import re

import ollama
from dotenv import load_dotenv

load_dotenv()


OLLAMA_HOST = os.getenv("OLLAMA_HOST")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = 0.1


def parse_thinking_and_response(content: str) -> dict:
    """
    Parse content to separate thinking process from the actual response.

    Args:
        content: Raw content from the model

    Returns:
        Dictionary with 'thinking' and 'response' keys
    """

    thinking_patterns = [
        r"<think>(.*?)</think>",
        r"<thinking>(.*?)</thinking>",
        r"\[THINKING\](.*?)\[/THINKING\]",
        r"\*\*Thinking:\*\*(.*?)(?=\n\n|\*\*|$)",
        r"Think:(.*?)(?=\n\n|Response:|$)",
    ]

    thinking_content = ""
    response_content = content

    for pattern in thinking_patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            thinking_content = "\n".join(matches).strip()
            response_content = re.sub(
                pattern, "", content, flags=re.DOTALL | re.IGNORECASE
            ).strip()
            break

    return {"thinking": thinking_content, "response": response_content}


def generate_next_question_stream(conversation_history: list[dict]):
    """
    Generates a single, focused follow-up question with streaming support.

    Args:
        conversation_history: A list of dictionaries representing the chat so far.

    Yields:
        Dictionary containing streaming chunks with thinking and response content.
    """
    system_prompt = """
    You are an expert AI medical intake assistant. Your role is to be empathetic, professional,
    and thorough. Your goal is to guide a user through a comprehensive medical intake interview.

    **Your Task:**
    Based on the entire conversation history provided, determine the **single most important question**
    to ask next. Ask only ONE question at a time.

    **Guiding Framework (Your toolkit):**
    Use the following clinical framework to guide your questions. The order is not strict, but your goal is to build a complete picture of the patient's current illness.

    - **Chief Complaint Exploration (OPQRSTU):**
        - **O (Onset):** When did it start? What were you doing?
        - **P (Provocation/Palliation):** What makes it better or worse?
        - **Q (Quality):** What does it feel like (e.g., sharp, dull, burning)?
        - **R (Region/Radiation):** Where is it located? Does the feeling spread anywhere?
        - **S (Severity):** On a scale of 1 to 10, how bad is it?
        - **T (Timing):** How long does it last? Is it constant or intermittent?
        - **U (Understanding):** What do you think is causing it?

    - **Associated Symptoms:**
        - **Crucially, after understanding the main symptom, ask about others.** For example: "Along with the fever, have you experienced any other symptoms like sneezing, coughing, or a sore throat?" or "Does anything else accompany the stomach pain, like nausea or dizziness?"

    - **Broader History (after the main illness is detailed):**
        - Past Medical History
        - Medications and Allergies
        - Social or Family History, gender, age, occupation, etc.
        - Lab Results

    **Crucial Rules:**
    1.  **NEVER** provide a diagnosis, medical advice, or suggestions.
    2.  Ask only **ONE** question.
    3.  Your entire response must be a single JSON object in the format: {"question": "Your question here."}
    4.  Do not add any text outside of the JSON object.
    """

    try:
        client = ollama.Client(host=OLLAMA_HOST)

        accumulated_content = ""
        thinking_content = ""
        response_content = ""

        response_stream = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(conversation_history)},
            ],
            options={"temperature": TEMPERATURE},
            stream=True,
        )

        for chunk in response_stream:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                accumulated_content += content

                parsed = parse_thinking_and_response(accumulated_content)

                if (
                    parsed["thinking"] != thinking_content
                    or parsed["response"] != response_content
                ):
                    thinking_content = parsed["thinking"]
                    response_content = parsed["response"]

                    yield {
                        "type": "chunk",
                        "thinking": thinking_content,
                        "response": response_content,
                        "raw_content": content,
                    }

        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if json_match:
            try:
                final_json = json.loads(json_match.group())
                yield {
                    "type": "complete",
                    "thinking": thinking_content,
                    "response": response_content,
                    "parsed_json": final_json,
                }
            except json.JSONDecodeError:
                yield {
                    "type": "error",
                    "error": "Could not parse JSON from response",
                    "thinking": thinking_content,
                    "response": response_content,
                }
        else:
            yield {
                "type": "error",
                "error": "Could not find JSON in response",
                "thinking": thinking_content,
                "response": response_content,
            }

    except Exception as e:
        yield {"type": "error", "error": f"An unexpected error occurred: {e}"}


def generate_next_question(conversation_history: list[dict]) -> dict | None:
    try:
        chunks = list(generate_next_question_stream(conversation_history))

        for chunk in reversed(chunks):
            if chunk.get("type") == "complete":
                return chunk.get("parsed_json")
            elif chunk.get("type") == "error":
                print(f"Error: {chunk.get('error')}")
                return None

        return None
    except Exception as e:
        print(f"An unexpected error occurred in generate_next_question: {e}")
        return None


def check_completion_status_stream(conversation_history: list[dict]):
    """
    Check the completion status of the conversation history.

    Args:
        conversation_history: A list of dictionaries representing the chat so far.

    Yields:
        Dictionary containing streaming chunks with thinking and response content.
    """
    system_prompt = """
    You are an AI medical intake auditor. Your task is to review a conversation and
    determine if enough information has been gathered for a complete patient report.

    A complete intake MUST cover these areas sufficiently:
    1.  **Chief Complaint:** Fully explored using the OPQRSTU model (Onset, Quality, Location, etc.).
    2.  **Associated Symptoms:** You must check if other related symptoms have been asked about (e.g., if the complaint is a cough, has the AI asked about fever or shortness of breath?).
    3.  **Past Medical History:** Any significant past illnesses or surgeries.
    4.  **Medications & Allergies:** What the patient takes and is allergic to.
    5.  **Family History:** Any significant medical conditions in the patient's family.
    6.  **Social History:** Any significant lifestyle factors (smoking, alcohol use, occupation).
    7.  **Lab Results:** Any significant lab results provided by the patient.

    Review the history. If these areas are covered, you can conclude. If not, the intake must continue.

    **Your Response:**
    Respond with ONLY a JSON object in this format:
    {"is_complete": boolean, "reason": "A brief explanation for your decision."}
    
    Example 1: {"is_complete": false, "reason": "I still need to ask about associated symptoms and past medical history."}
    Example 2: {"is_complete": true, "reason": "The chief complaint and associated symptoms have been fully explored, and we have covered medical history, medications, and allergies."}
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)

        accumulated_content = ""
        thinking_content = ""
        response_content = ""

        response_stream = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(conversation_history)},
            ],
            options={"temperature": 0.0},
            stream=True,
        )

        for chunk in response_stream:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                accumulated_content += content

                parsed = parse_thinking_and_response(accumulated_content)

                if (
                    parsed["thinking"] != thinking_content
                    or parsed["response"] != response_content
                ):
                    thinking_content = parsed["thinking"]
                    response_content = parsed["response"]

                    yield {
                        "type": "chunk",
                        "thinking": thinking_content,
                        "response": response_content,
                        "raw_content": content,
                    }

        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if json_match:
            try:
                final_json = json.loads(json_match.group())
                yield {
                    "type": "complete",
                    "thinking": thinking_content,
                    "response": response_content,
                    "parsed_json": final_json,
                }
            except json.JSONDecodeError:
                yield {
                    "type": "complete",
                    "thinking": thinking_content,
                    "response": response_content,
                    "parsed_json": {
                        "is_complete": False,
                        "reason": "Response parsing error.",
                    },
                }
        else:
            yield {
                "type": "complete",
                "thinking": thinking_content,
                "response": response_content,
                "parsed_json": {
                    "is_complete": False,
                    "reason": "Could not find JSON in response.",
                },
            }

    except Exception as e:
        yield {
            "type": "error",
            "error": f"An error occurred: {e}",
            "parsed_json": {
                "is_complete": False,
                "reason": "System error during check.",
            },
        }


def check_completion_status(conversation_history: list[dict]) -> dict:
    try:
        chunks = list(check_completion_status_stream(conversation_history))

        for chunk in reversed(chunks):
            if chunk.get("type") in ["complete", "error"]:
                return chunk.get(
                    "parsed_json", {"is_complete": False, "reason": "Unknown error."}
                )

        return {"is_complete": False, "reason": "No response received."}
    except Exception as e:
        print(f"An error occurred in check_completion_status: {e}")
        return {"is_complete": False, "reason": "System error during check."}


def generate_intake_report_stream(conversation_history: list[dict]):
    """
    Generate a structured intake report from the conversation history.

    Args:
        conversation_history: A list of dictionaries representing the chat so far.

    Yields:
        Dictionary containing streaming chunks with thinking and response content.
    """

    system_prompt = """
    You are an AI medical scribe. Your sole purpose is to convert a conversation transcript 
    into a formal, structured, and comprehensive patient intake report. Use the provided
    transcript to fill in every section accurately. If information is not available, state "Not mentioned."

    **Output Format (Strict):**

    # Patient Intake Report

    ## Patient Information
    - **Patient Name:** [Not Collected]
    - **Date of Birth:** [Not Collected]
    - **Date of Intake:** [Current Date]

    ## 1. Chief Complaint
    *The primary reason for the visit as stated by the patient.*
    - 

    ## 2. History of Present Illness (HPI)
    *A detailed narrative of the chief complaint from onset to present.*
    - **Onset:** 
    - **Location & Radiation:** 
    - **Duration & Timing:** 
    - **Character/Quality:** 
    - **Severity:** 
    - **Aggravating Factors:** 
    - **Relieving Factors:** 
    - **Associated Symptoms:** 
    - **Patient's Understanding:** 

    ## 3. Review of Systems (ROS)
    *A summary of other symptoms mentioned during the conversation.*
    - **General:** (e.g., fever, weight loss)
    - **Gastrointestinal:** (e.g., nausea, vomiting, diarrhea)
    - **Neurological:** (e.g., headache, dizziness)
    - *(Add other systems as mentioned)*

    ## 4. Past Medical History (PMH)
    - **Past Illnesses:** 
    - **Surgeries:** 
    - **Hospitalizations:** 

    ## 5. Medications
    *List of current medications and dosages.*
    - 

    ## 6. Allergies
    *List of drug, food, and environmental allergies.*
    - 

    ## 7. Family History
    *Medical conditions in immediate family members.*
    - 

    ## 8. Social History
    *Details on lifestyle such as smoking, alcohol use, and occupation.*
    - 

    ## 9. Lab Results
    *Summary of any lab results provided by the user.*
    - 

    ## 10. Possible Diagnoses
    *List of possible diagnoses based on the information provided.*
    - 

    --- END OF REPORT ---
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)

        accumulated_content = ""
        thinking_content = ""
        response_content = ""

        response_stream = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(conversation_history)},
            ],
            options={"temperature": 0.0},
            stream=True,
        )

        for chunk in response_stream:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                accumulated_content += content

                parsed = parse_thinking_and_response(accumulated_content)

                if (
                    parsed["thinking"] != thinking_content
                    or parsed["response"] != response_content
                ):
                    thinking_content = parsed["thinking"]
                    response_content = parsed["response"]

                    yield {
                        "type": "chunk",
                        "thinking": thinking_content,
                        "response": response_content,
                        "raw_content": content,
                    }

        yield {
            "type": "complete",
            "thinking": thinking_content,
            "response": response_content,
            "final_report": response_content,
        }

    except Exception as e:
        yield {"type": "error", "error": f"Error generating report: {e}"}


def generate_intake_report(conversation_history: list[dict]) -> str:
    try:
        chunks = list(generate_intake_report_stream(conversation_history))

        for chunk in reversed(chunks):
            if chunk.get("type") == "complete":
                return chunk.get("final_report", "Error generating report")
            elif chunk.get("type") == "error":
                return chunk.get("error", "Unknown error generating report")

        return "No response received"
    except Exception as e:
        return f"Error generating report: {e}"


def chat():
    pass


if __name__ == "__main__":
    chat()
