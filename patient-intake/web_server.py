import json
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, session
from flask_session import Session

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
