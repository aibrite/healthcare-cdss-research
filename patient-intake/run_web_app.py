#!/usr/bin/env python3
"""
Medical Intake Assistant - Web Application Launcher
Run this script to start the web interface for medical intake assistant.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import flask_session
        import ollama

        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Installing required dependencies...")

        try:
            if Path("pyproject.toml").exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
            else:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "flask",
                        "ollama",
                        "flask-session",
                    ]
                )
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install dependencies")
            print("Please run: pip install flask ollama flask-session")
            return False


def check_ollama_connection():
    """Check if Ollama server is accessible"""
    try:
        import ollama
        from patient_intake import MODEL_NAME, OLLAMA_HOST

        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
        print(f"✓ Ollama server is accessible at {OLLAMA_HOST}")

        model_names = [model["name"] for model in models["models"]]
        if MODEL_NAME in model_names:
            print(f"✓ Required model '{MODEL_NAME}' is available")
        else:
            print(f"⚠️  Warning: Required model '{MODEL_NAME}' not found")
            print("Available models:", model_names)
            print("You may need to pull the model first:")
            print(f"  ollama pull {MODEL_NAME}")

        return True
    except Exception as e:
        print(f"✗ Cannot connect to Ollama server: {e}")
        print("Please ensure Ollama is running at the configured address")
        print(
            "Check the OLLAMA_HOST setting in patient-intake.py if using a different address"
        )
        return False


def test_llm_functions():
    """Test the core LLM functions to ensure they work"""
    try:
        from patient_intake import (
            check_completion_status,
            generate_intake_report,
            generate_next_question,
        )

        test_conversation = [{"role": "user", "content": "I have a headache"}]

        next_question = generate_next_question(test_conversation)
        if next_question and "question" in next_question:
            print("✓ Question generation is working")
        else:
            print("✗ Question generation failed")
            return False

        completion_status = check_completion_status(test_conversation)
        if isinstance(completion_status, dict) and "is_complete" in completion_status:
            print("✓ Completion status checking is working")
        else:
            print("✗ Completion status checking failed")
            return False

        print("✓ All LLM functions are working correctly")
        return True

    except Exception as e:
        print(f"✗ Error testing LLM functions: {e}")
        return False


def main():
    print("🏥 Medical Intake Assistant - Web Interface")
    print("=" * 50)

    if not Path("patient-intake.py").exists():
        print(
            "✗ patient-intake.py not found. Please run this script from the project directory."
        )
        sys.exit(1)

    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    print("\nChecking Ollama connection...")
    if not check_ollama_connection():
        print("\n⚠️  Warning: Ollama server not accessible.")
        print("The web interface will start, but AI functionality may not work.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)
    else:
        print("\nTesting LLM functions...")
        if not test_llm_functions():
            print("\n⚠️  Warning: LLM functions are not working properly.")
            print("The web interface will start, but AI functionality may not work.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                sys.exit(1)

    print("\n🚀 Starting web server...")
    print("Web interface will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    os.environ["FLASK_APP"] = "web_server.py"
    os.environ["FLASK_ENV"] = "development"

    try:
        import threading

        def open_browser():
            import time

            time.sleep(2)
            webbrowser.open("http://localhost:5000")

        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()

        from web_server import app

        app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)

    except KeyboardInterrupt:
        print("\n\n👋 Web server stopped. Goodbye!")
    except Exception as e:
        print(f"\n✗ Error starting web server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
