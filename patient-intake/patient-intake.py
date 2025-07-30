import os

from dotenv import load_dotenv

load_dotenv()


OLLAMA_HOST = os.getenv("OLLAMA_HOST")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = 0.1


def generate_next_question(conversation_history: list[dict]) -> dict | None:
    pass


def generate_next_question_stream(conversation_history: list[dict]):
    pass


def check_completion_status(conversation_history: list[dict]) -> dict:
    pass


def check_completion_status_stream(conversation_history: list[dict]):
    pass


def generate_intake_report(conversation_history: list[dict]) -> str:
    pass


def generate_intake_report_stream(conversation_history: list[dict]):
    pass


def chat():
    pass


if __name__ == "__main__":
    chat()
