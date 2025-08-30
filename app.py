import os
import io
import fitz  # PyMuPDF for PDF
import docx  # python-docx
import pytesseract  # OCR for images
from PIL import Image
import requests as req
from bs4 import BeautifulSoup

from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================================
# Load environment variables
# ==========================================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")   # from .env
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

# ==========================================================
# Flask setup
# ==========================================================
app = Flask(__name__)
app.secret_key = "supersecret"
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================================
# OpenAI-compatible client (Groq API)
# ==========================================================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
MODEL_NAME = "llama3-8b-8192"  # Options: "llama3-70b-8192", "gpt-4o-mini", etc.


# ==========================================================
# Helper: Ask LLaMA (Groq API)
# ==========================================================
def ask_llm(prompt, model=MODEL_NAME):
    """
    Sends a prompt to the API and returns the response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM API error: {str(e)}"


# ==========================================================
# Helpers: Extract text from different sources
# ==========================================================
def extract_text_from_pdf(file):
    """
    Extract text from uploaded PDF using PyMuPDF
    """
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in pdf])
        return text.strip()
    except Exception as e:
        return f"⚠️ PDF extraction error: {str(e)}"


def extract_text_from_docx(file):
    """
    Extract text from uploaded DOCX using python-docx
    """
    try:
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"⚠️ DOCX extraction error: {str(e)}"


def extract_text_from_image(file):
    """
    Extract text from uploaded Image using Tesseract OCR
    """
    try:
        img = Image.open(io.BytesIO(file.read()))
        return pytesseract.image_to_string(img)
    except Exception as e:
        return f"⚠️ Image OCR error: {str(e)}"


def extract_text_from_url(url):
    """
    Scrape raw text from a given website URL
    """
    try:
        res = req.get(url, timeout=8)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        return f"⚠️ URL fetch error: {str(e)}"


# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def home():
    """
    Homepage → renders frontend (index.html)
    """
    return render_template("index.html")


# ----------------------- File Upload -----------------------
@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles PDF, Word, Image uploads
    """
    if "file" not in request.files:
        return jsonify({"summary": "⚠️ No file uploaded"})

    file = request.files["file"]
    filename = file.filename.lower()

    # Extract text
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(file)
    else:
        return jsonify({"summary": "⚠️ Unsupported file type"})

    # Validate text
    if not text or text.startswith("⚠️"):
        return jsonify({"summary": "⚠️ Could not extract text from file."})

    # Store & summarize
    session["document"] = text
    summary = ask_llm(f"Summarize this text in 3-4 lines:\n\n{text}")
    return jsonify({"summary": summary})


# ----------------------- URL Input -----------------------
@app.route("/url", methods=["POST"])
def url_upload():
    """
    Handles URL input
    """
    data = request.json
    url = data.get("url", "")
    if not url:
        return jsonify({"summary": "⚠️ No URL provided"})

    text = extract_text_from_url(url)
    if not text or text.startswith("⚠️"):
        return jsonify({"summary": "⚠️ Could not fetch text from URL."})

    # Store & summarize
    session["document"] = text
    summary = ask_llm(f"Summarize this text in 3-4 lines:\n\n{text}")
    return jsonify({"summary": summary})


# ----------------------- Summarize -----------------------
@app.route("/summarize", methods=["POST", "GET"])
def summarize():
    """
    Handles paste text + summary modes (short/detailed/bullet)
    """
    # POST → summarize pasted text
    if request.method == "POST":
        data = request.json
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"summary": "⚠️ No text provided."})

        session["document"] = text
        summary = ask_llm(f"Summarize this text in 3-4 lines:\n\n{text}")
        return jsonify({"summary": summary})

    # GET → summary modes
    mode = request.args.get("mode", "short")
    text = session.get("document", "")

    if not text:
        return jsonify({"summary": "⚠️ Please upload or paste text first."})

    # Mode-specific prompts
    if mode == "short":
        prompt = f"Give me a short summary (2-3 lines):\n\n{text}"
    elif mode == "detailed":
        prompt = f"Write a detailed summary of this document:\n\n{text}"
    elif mode == "bullet":
        prompt = f"Summarize this document into clear bullet points:\n\n{text}"
    else:
        prompt = f"Summarize this text:\n\n{text}"

    summary = ask_llm(prompt)
    return jsonify({"summary": summary})


# ----------------------- Ask Questions -----------------------
@app.route("/ask", methods=["POST"])
def ask():
    """
    Handles Q&A based on uploaded/pasted document
    """
    data = request.json
    question = data.get("question", "")
    text = session.get("document", "")

    if not text:
        return jsonify({"answer": "⚠️ Please upload or paste text first."})
    if not question.strip():
        return jsonify({"answer": "⚠️ Please enter a question."})

    prompt = f"Document:\n{text}\n\nQuestion: {question}\nAnswer clearly:"
    answer = ask_llm(prompt)
    return jsonify({"answer": answer})


# ==========================================================
# Run server
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)
