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


@app.route("/respond_to_question", methods=["POST"])
def respond_to_question():
    """Handle user's response to clarifying questions"""
    data = request.get_json()
    user_response = data.get("response", "")

    if "conversation_history" not in session:
        return jsonify({"error": "No active chat session"}), 400

    conversation_history = session["conversation_history"]

    if user_response.strip():
        conversation_history.append({"role": "user", "content": user_response})

    session["conversation_history"] = conversation_history

    if user_response.lower() in ["done", "finished", "no more", "that's all"]:
        final_report = generate_intake_report(conversation_history)
        response = {
            "intake_complete": True,
            "final_report": final_report,
            "current_symptoms": [conversation_history[0]["content"]],
        }
        return jsonify(response)

    completion_status = check_completion_status(conversation_history)

    if completion_status.get("is_complete", False):
        final_report = generate_intake_report(conversation_history)
        response = {
            "intake_complete": True,
            "final_report": final_report,
            "current_symptoms": [conversation_history[0]["content"]],
        }
        return jsonify(response)

    next_question_data = generate_next_question(conversation_history)

    if not next_question_data or not next_question_data.get("question"):
        final_report = generate_intake_report(conversation_history)
        response = {
            "intake_complete": True,
            "final_report": final_report,
            "current_symptoms": [conversation_history[0]["content"]],
        }
        return jsonify(response)

    question = next_question_data.get("question")

    conversation_history.append({"role": "assistant", "content": question})
    session["conversation_history"] = conversation_history

    response = {
        "intake_complete": False,
        "clarifying_questions": [question],
        "current_symptoms": [conversation_history[0]["content"]],
        "conversation_step": len(
            [msg for msg in conversation_history if msg["role"] == "user"]
        ),
        "has_more_questions": True,
    }

    return jsonify(response)


@app.route("/respond_to_question_stream", methods=["POST"])
def respond_to_question_stream():
    """Handle user's response with streaming"""
    data = request.get_json()
    user_response = data.get("response", "")

    if "conversation_history" not in session:
        return jsonify({"error": "No active chat session"}), 400

    conversation_history = list(session["conversation_history"])

    if user_response.strip():
        conversation_history.append({"role": "user", "content": user_response})

    session["conversation_history"] = conversation_history

    def generate():
        yield f"event: start\ndata: {json.dumps({'status': 'processing_response'})}\n\n"

        try:
            if user_response.lower() in ["done", "finished", "no more", "that's all"]:
                yield f"event: status\ndata: {json.dumps({'status': 'generating_report'})}\n\n"

                for chunk in generate_intake_report_stream(conversation_history):
                    if chunk.get("type") == "chunk":
                        yield f"event: thinking\ndata: {json.dumps({'thinking': chunk.get('thinking', ''), 'response': chunk.get('response', '')})}\n\n"
                    elif chunk.get("type") == "complete":
                        response_data = {
                            "intake_complete": True,
                            "final_report": chunk.get("final_report", ""),
                            "current_symptoms": [conversation_history[0]["content"]],
                            "thinking": chunk.get("thinking", ""),
                            "session_update": {
                                "conversation_history": conversation_history
                            },
                        }
                        yield f"event: complete\ndata: {json.dumps(response_data)}\n\n"
                    elif chunk.get("type") == "error":
                        yield f"event: error\ndata: {json.dumps({'error': chunk.get('error', 'Unknown error')})}\n\n"
                return

            yield f"event: status\ndata: {json.dumps({'status': 'checking_completion'})}\n\n"

            completion_status = None
            for chunk in check_completion_status_stream(conversation_history):
                if chunk.get("type") == "chunk":
                    yield f"event: thinking\ndata: {json.dumps({'thinking': chunk.get('thinking', ''), 'response': chunk.get('response', '')})}\n\n"
                elif chunk.get("type") in ["complete", "error"]:
                    completion_status = chunk.get("parsed_json", {"is_complete": False})
                    break

            if completion_status and completion_status.get("is_complete", False):
                yield f"event: status\ndata: {json.dumps({'status': 'generating_report'})}\n\n"

                for chunk in generate_intake_report_stream(conversation_history):
                    if chunk.get("type") == "chunk":
                        yield f"event: thinking\ndata: {json.dumps({'thinking': chunk.get('thinking', ''), 'response': chunk.get('response', '')})}\n\n"
                    elif chunk.get("type") == "complete":
                        response_data = {
                            "intake_complete": True,
                            "final_report": chunk.get("final_report", ""),
                            "current_symptoms": [conversation_history[0]["content"]],
                            "thinking": chunk.get("thinking", ""),
                            "session_update": {
                                "conversation_history": conversation_history
                            },
                        }
                        yield f"event: complete\ndata: {json.dumps(response_data)}\n\n"
                    elif chunk.get("type") == "error":
                        yield f"event: error\ndata: {json.dumps({'error': chunk.get('error', 'Unknown error')})}\n\n"
                return

            yield f"event: status\ndata: {json.dumps({'status': 'generating_question'})}\n\n"

            question_generated = False
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
                            "intake_complete": False,
                            "clarifying_questions": [question],
                            "current_symptoms": [conversation_history[0]["content"]],
                            "conversation_step": len(
                                [
                                    msg
                                    for msg in conversation_history
                                    if msg["role"] == "user"
                                ]
                            ),
                            "has_more_questions": True,
                            "thinking": chunk.get("thinking", ""),
                            "session_update": {
                                "conversation_history": conversation_history
                            },
                        }
                        yield f"event: complete\ndata: {json.dumps(response_data)}\n\n"
                        question_generated = True
                    break
                elif chunk.get("type") == "error":
                    yield f"event: error\ndata: {json.dumps({'error': chunk.get('error', 'Unknown error')})}\n\n"
                    break

            if not question_generated:
                yield f"event: status\ndata: {json.dumps({'status': 'generating_report'})}\n\n"

                for chunk in generate_intake_report_stream(conversation_history):
                    if chunk.get("type") == "chunk":
                        yield f"event: thinking\ndata: {json.dumps({'thinking': chunk.get('thinking', ''), 'response': chunk.get('response', '')})}\n\n"
                    elif chunk.get("type") == "complete":
                        response_data = {
                            "intake_complete": True,
                            "final_report": chunk.get("final_report", ""),
                            "current_symptoms": [conversation_history[0]["content"]],
                            "thinking": chunk.get("thinking", ""),
                            "session_update": {
                                "conversation_history": conversation_history
                            },
                        }
                        yield f"event: complete\ndata: {json.dumps(response_data)}\n\n"
                    elif chunk.get("type") == "error":
                        yield f"event: error\ndata: {json.dumps({'error': chunk.get('error', 'Unknown error')})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/update_session", methods=["POST"])
def update_session():
    """Update session state after streaming completes"""
    data = request.get_json()

    if "conversation_history" in data:
        session["conversation_history"] = data["conversation_history"]

    return jsonify({"status": "session updated"})


@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    """Reset the current chat session"""
    session.clear()
    return jsonify({"message": "Chat session reset successfully"})


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Medical Intake Assistant"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
