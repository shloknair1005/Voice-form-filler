# ============================================================================
# BACKEND: voice_form_backend.py
# ============================================================================

import spacy
import speech_recognition as sr
from spacy.matcher import Matcher
import re
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import queue

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)


class VoiceFormFiller:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize form data
        self.form_data = {
            "name": "",
            "age": "",
            "city": "",
            "college_year": "",
            "college_name": "",
            "skills": []
        }

        # Initialize matcher
        self.matcher = Matcher(nlp.vocab)
        self._setup_patterns()

        # Track what's been spoken
        self.transcript_lines = []
        self.is_listening = False
        self.listen_thread = None

        # Skill keywords
        self.skill_keywords = [
            "python", "java", "javascript", "c++", "cpp", "react", "angular", "vue",
            "machine learning", "deep learning", "data science", "ai", "artificial intelligence",
            "web development", "app development", "mobile development",
            "sentiment analysis", "face recognition", "chatbot", "nlp",
            "computer vision", "neural networks", "tensorflow", "pytorch",
            "sql", "mongodb", "database", "node.js", "nodejs", "express", "django", "flask",
            "html", "css", "bootstrap", "tailwind", "git", "github",
            "data analysis", "pandas", "numpy", "matplotlib", "scikit-learn", "sklearn",
            "cloud computing", "aws", "azure", "docker", "kubernetes"
        ]

    def _setup_patterns(self):
        year_patterns = [
            [{"LOWER": {"IN": ["first", "1st"]}}, {"LOWER": "year"}],
            [{"LOWER": {"IN": ["second", "2nd"]}}, {"LOWER": "year"}],
            [{"LOWER": {"IN": ["third", "3rd"]}}, {"LOWER": "year"}],
            [{"LOWER": {"IN": ["fourth", "4th", "final"]}}, {"LOWER": "year"}]
        ]
        self.matcher.add("COLLEGE_YEAR", year_patterns)

    def extract_name(self, doc, text):
        if self.form_data["name"]:
            return
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                self.form_data["name"] = ent.text
                return
        name_patterns = [
            r"(?:my name is|i am|i'm|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                self.form_data["name"] = match.group(1)
                return

    def extract_age(self, text):
        if self.form_data["age"]:
            return
        age_patterns = [
            r"(?:i am|i'm|age is|age)\s*(\d{1,2})\s*(?:years?\s*old)?",
            r"(\d{1,2})\s*years?\s*old",
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 15 <= age <= 99:
                    self.form_data["age"] = str(age)
                    return

    def extract_city(self, doc, text):
        if self.form_data["city"]:
            return
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                self.form_data["city"] = ent.text
                return
        indian_cities = [
            "mumbai", "delhi", "bangalore", "bengaluru", "pune", "hyderabad",
            "chennai", "kolkata", "ahmedabad", "jaipur", "lucknow", "kanpur"
        ]
        text_lower = text.lower()
        for city in indian_cities:
            if city in text_lower:
                self.form_data["city"] = city.title()
                return

    def extract_college_year(self, doc):
        if self.form_data["college_year"]:
            return
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if nlp.vocab.strings[match_id] == "COLLEGE_YEAR":
                year_text = doc[start:end].text.lower()
                year_map = {
                    "first": "1st Year", "1st": "1st Year",
                    "second": "2nd Year", "2nd": "2nd Year",
                    "third": "3rd Year", "3rd": "3rd Year",
                    "fourth": "4th Year", "4th": "4th Year",
                    "final": "4th Year"
                }
                for key, value in year_map.items():
                    if key in year_text:
                        self.form_data["college_year"] = value
                        return

    def extract_college_name(self, doc, text):
        if self.form_data["college_name"]:
            return
        college_patterns = [
            r"(?:studying at|student at|from|at)\s+([A-Z][A-Za-z\s&]+(?:university|college|institute|iit|nit|bits))",
            r"(iit|nit|bits)\s+([a-z]+)",
        ]
        for pattern in college_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    self.form_data["college_name"] = f"{match.group(1).upper()} {match.group(2).title()}"
                else:
                    self.form_data["college_name"] = match.group(1).strip()
                return

    def extract_skills(self, text):
        text_lower = text.lower()
        for skill in self.skill_keywords:
            if skill in text_lower and skill not in [s.lower() for s in self.form_data["skills"]]:
                formatted_skill = skill.title() if ' ' in skill else skill.upper() if len(
                    skill) <= 4 else skill.capitalize()
                self.form_data["skills"].append(formatted_skill)

    def process_speech(self, text):
        self.transcript_lines.append(text)
        doc = nlp(text)
        self.extract_name(doc, text)
        self.extract_age(text)
        self.extract_city(doc, text)
        self.extract_college_year(doc)
        self.extract_college_name(doc, text)
        self.extract_skills(text)

    def listen_loop(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
                try:
                    text = self.recognizer.recognize_google(audio)
                    self.process_speech(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
            except sr.WaitTimeoutError:
                continue
            except Exception:
                break

    def start_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.listen_thread = threading.Thread(target=self.listen_loop)
            self.listen_thread.start()

    def stop_listening(self):
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)

    def reset(self):
        self.form_data = {
            "name": "", "age": "", "city": "",
            "college_year": "", "college_name": "", "skills": []
        }
        self.transcript_lines = []


# Global instance
form_filler = VoiceFormFiller()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start_listening():
    form_filler.start_listening()
    return jsonify({"status": "started"})


@app.route('/stop', methods=['POST'])
def stop_listening():
    form_filler.stop_listening()
    return jsonify({"status": "stopped"})


@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "form_data": form_filler.form_data,
        "transcript": form_filler.transcript_lines,
        "is_listening": form_filler.is_listening
    })


@app.route('/reset', methods=['POST'])
def reset():
    form_filler.reset()
    return jsonify({"status": "reset"})


if __name__ == '__main__':
    print("Starting Voice Form Filler Server...")
    print("Visit http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)