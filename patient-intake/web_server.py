import json
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, session
from flask_session import Session
from patient_intake import (
    check_completion_status,
    check_completion_status_stream,
    generate_intake_report,
    generate_intake_report_stream,
    generate_next_question,
    generate_next_question_stream,
)

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/")
def index():
    """Serve the main chat interface"""
    return render_template("index.html")


@app.route("/start_chat", methods=["POST"])
def start_chat():
    """Initialize a new chat session with initial symptoms"""
    data = request.get_json()
    initial_symptoms = data.get("symptoms", "")

    if not initial_symptoms:
        return jsonify({"error": "Please provide initial symptoms"}), 400

    session["chat_id"] = str(uuid.uuid4())
    session["conversation_history"] = [{"role": "user", "content": initial_symptoms}]

    next_question_data = generate_next_question(session["conversation_history"])

    if not next_question_data or not next_question_data.get("question"):
        return jsonify(
            {"error": "Could not generate intake questions. Please try again."}
        ), 500

    question = next_question_data.get("question")

    session["conversation_history"].append({"role": "assistant", "content": question})

    response = {
        "chat_id": session["chat_id"],
        "clarifying_questions": [question],
        "current_symptoms": [initial_symptoms],
        "conversation_step": 1,
    }

    return jsonify(response)


@app.route("/start_chat_stream", methods=["POST"])
def start_chat_stream():
    """Initialize a new chat session with streaming response"""
    data = request.get_json()
    initial_symptoms = data.get("symptoms", "")

    if not initial_symptoms:
        return jsonify({"error": "Please provide initial symptoms"}), 400

    chat_id = str(uuid.uuid4())
    conversation_history = [{"role": "user", "content": initial_symptoms}]
    session["chat_id"] = chat_id
    session["conversation_history"] = conversation_history

    def generate():
        yield f"event: start\ndata: {json.dumps({'chat_id': chat_id, 'status': 'generating_question'})}\n\n"

        try:
            for chunk in generate_next_question_stream(conversation_history):
                if chunk.get("type") == "chunk":
                    yield f"event: thinking\ndata: {json.dumps({'thinking': chunk.get('thinking', ''), 'response': chunk.get('response', '')})}\n\n"
                elif chunk.get("type") == "complete":
                    question_data = chunk.get("parsed_json", {})
                    if question_data.get("question"):
                        question = question_data.get("question")
                        conversation_history.append(
                            {"role": "assistant", "content": question}
                        )

                        response_data = {
                            "chat_id": chat_id,
                            "clarifying_questions": [question],
                            "current_symptoms": [initial_symptoms],
                            "conversation_step": 1,
                            "thinking": chunk.get("thinking", ""),
                            "session_update": {
                                "conversation_history": conversation_history
                            },
                        }
                        yield f"event: complete\ndata: {json.dumps(response_data)}\n\n"
                    else:
                        yield f"event: error\ndata: {json.dumps({'error': 'Could not generate question'})}\n\n"
                elif chunk.get("type") == "error":
                    yield f"event: error\ndata: {json.dumps({'error': chunk.get('error', 'Unknown error')})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")
