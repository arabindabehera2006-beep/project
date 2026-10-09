"""
Multimodal Healthcare Triage Assistant — Flask backend.

Patient input → Streamlit → this API → Python processing
→ fusion → LangChain orchestration → structured triage
→ gTTS audio + healthcare professional review.

The AI organizes information and raises an urgency flag.
It does not diagnose or prescribe. A qualified professional
makes the final clinical decision.

Run:
    python backend.py
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

try:
    from langchain_core.runnables import RunnableLambda

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - demo still runs without the package
    RunnableLambda = None
    LANGCHAIN_AVAILABLE = False


ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

UPLOADS = ROOT / "uploads"
DATA = ROOT / "data"
CASES = DATA / "cases"
MODEL_PATH = ROOT / "models" / "visual_cnn.pt"

IMAGE_SIZE = 64
MAX_TEXT = 5000
PORT = 5050

DISCLAIMER = (
    "This suggestion is produced from the text, voice, report, and image you submitted. "
    "Groq names a likely condition and a suggested treatment from those inputs. "
    "Confirm both with a doctor, nurse, or medical officer before taking any medicine."
)

LANGUAGES = (
    "English",
    "Hindi",
    "Bengali",
    "Tamil",
    "Telugu",
    "Marathi",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Punjabi",
    "Urdu",
    "Other regional language",
)
REVIEWER_ROLES = ("Doctor", "Nurse", "Medical Officer")

# Prototype visual flags only. These are not diagnoses.
VISUAL_LABELS = [
    "no_obvious_visual_concern",
    "visible_redness_flag",
    "visible_swelling_flag",
    "wound_or_surface_break_flag",
    "unclear_image_flag",
]

VISUAL_TEXT = {
    "no_obvious_visual_concern": "Visual flag: no obvious visual concern in the uploaded image.",
    "visible_redness_flag": "Visual flag: redness pattern in the uploaded image.",
    "visible_swelling_flag": "Visual flag: swelling-like pattern in the uploaded image.",
    "wound_or_surface_break_flag": "Visual flag: wound or surface-break pattern in the uploaded image.",
    "unclear_image_flag": "Visual flag: the image is unclear. Retake it if you can.",
}

SYMPTOMS = [
    ("Fever", ["fever", "बुखार"]),
    ("Weakness", ["weakness", "fatigue", "कमजोरी"]),
    ("Cough", ["cough", "खांसी"]),
    ("Cold", ["cold", "जुकाम"]),
    ("Headache", ["headache", "सिरदर्द"]),
    ("Body ache", ["body ache", "body pain", "बदन दर्द"]),
    ("Sore throat", ["sore throat", "गले में दर्द"]),
    ("Vomiting", ["vomiting", "vomit", "उल्टी"]),
    ("Nausea", ["nausea", "मतली"]),
    ("Dizziness", ["dizziness", "dizzy", "चक्कर"]),
    ("Rash", ["rash", "रैश", "दाने"]),
    ("Pain", ["pain", "दर्द"]),
    ("Swelling", ["swelling", "swollen", "सूजन"]),
    ("Bleeding", ["bleeding", "खून"]),
    ("Wound", ["wound", "cut", "घाव"]),
    ("Breathlessness", ["breathlessness", "shortness of breath", "सांस फूल"]),
    ("Diarrhea", ["diarrhea", "diarrhoea", "दस्त"]),
]

# Higher urgency is checked first. These route the case to a person.
# They do not name a disease.
URGENT_PATTERNS = [
    r"chest pain",
    r"सीने में दर्द",
    r"difficulty breathing",
    r"shortness of breath",
    r"breathlessness",
    r"can(?:not|'t) breathe",
    r"सांस फूल",
    r"सांस लेने में",
    r"unconscious",
    r"unresponsive",
    r"not breathing",
    r"बेहोश",
    r"behosh",
    r"severe bleeding",
    r"heavy bleeding",
    r"seizure",
    r"convulsion",
    r"दौरा",
    r"face drooping",
    r"slurred speech",
    r"sudden confusion",
    r"blue lips",
    r"anaphylaxis",
    r"severe allergic",
    r"coughing blood",
    r"vomiting blood",
    r"suicidal",
    r"suicide",
]

PRIORITY_PATTERNS = [
    r"high fever",
    r"severe pain",
    r"\bsevere\b",
    r"persistent vomiting",
    r"fainting",
    r"fainted",
    r"dehydrat",
    r"pregnant",
    r"pregnancy",
    r"गर्भवती",
    r"गर्भावस्था",
    r"infant",
    r"newborn",
    r"\bbaby\b",
    r"शिशु",
    r"blood in stool",
    r"blood in urine",
]

MILD_ONLY = {"Cough", "Cold", "Sore throat"}

PATIENT_MESSAGE_EN = (
    "Your information has been organized. Please wait for professional review. "
    "This tool does not diagnose or prescribe treatment."
)
EMERGENCY_LINE_EN = "If this is an emergency, contact local emergency services now."

# Speech and translation codes for the languages offered on the page.
# Other names are resolved through googletrans at request time.
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
}
SPEECH_LANG = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Punjabi": "pa-IN",
    "Urdu": "ur-IN",
}

ASSERTION_PATTERNS = [
    r"\byou have\b.{0,40}\b(disease|infection|condition)\b",
    r"\bthe diagnosis is\b",
    r"\bi diagnose\b",
    r"\bprescribed medication\b",
    r"\btake\s+\d+\s*mg\b",
    r"\brecommended dosage\b",
]

LOCK = threading.Lock()
CHAIN = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def ensure_dirs() -> None:
    for path in (UPLOADS, DATA, CASES, MODEL_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


ensure_dirs()


def now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_language(value: str | None) -> str:
    if not value:
        return "English"
    cleaned = value.strip().lower()
    for language in LANGUAGES:
        if cleaned == language.lower():
            return language
    return "English"


def clip(text: str, limit: int = 2000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def mentioned(text: str, key: str) -> bool:
    if re.search(r"[^\x00-\x7f]", key):
        return key in text
    return re.search(rf"\b{re.escape(key.lower())}\b", text) is not None


def find_symptoms(text: str) -> list[str]:
    lower = text.lower()
    found: list[tuple[int, str]] = []
    for label, keys in SYMPTOMS:
        positions = []
        for key in keys:
            if not mentioned(lower, key):
                continue
            positions.append(lower.find(key.lower()))
        if positions:
            found.append((min(positions), label))
    found.sort()
    labels = []
    for _, label in found:
        if label not in labels:
            labels.append(label)
    return labels


def find_duration(text: str) -> str:
    match = re.search(
        r"\b(\d+)\s*(day|days|week|weeks|hour|hours|month|months|"
        r"दिन|घंटे|घंटा|हफ्ते|हफ्ता)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)} {match.group(2)}"
    for phrase, label in (
        (r"\byesterday\b|\bकल\b", "since yesterday"),
        (r"\btoday\b|\bआज\b", "since today"),
        (r"since morning|सुबह से", "since morning"),
    ):
        if re.search(phrase, text, flags=re.IGNORECASE):
            return label
    return ""


def has_age(text: str) -> bool:
    return bool(
        re.search(r"\b(age|उम्र)\b[^.\n]{0,16}\d{1,3}", text, flags=re.IGNORECASE)
        or re.search(
            r"\b\d{1,3}\s*(years?\s*old|yrs?\b|yo\b|years\b|saal|साल)",
            text,
            flags=re.IGNORECASE,
        )
    )


def has_temperature(text: str) -> bool:
    return bool(
        re.search(r"\b(temp|temperature|तापमान)\b", text, flags=re.IGNORECASE)
        or re.search(
            r"\b\d{2,3}(?:\.\d)?\s*(?:°|degrees?\s*[fc]?|[fc]\b)",
            text,
            flags=re.IGNORECASE,
        )
    )


def has_oxygen(text: str) -> bool:
    return bool(
        re.search(
            r"\b(oxygen|spo2|o2|saturation|ऑक्सीजन)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def high_temperature(text: str) -> bool:
    pattern = re.compile(
        r"\b(\d{2,3}(?:\.\d)?)\s*(?:°?\s*([fc])|degrees?\s*([fc])?)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        value = float(match.group(1))
        unit = (match.group(2) or match.group(3) or "f").lower()
        if unit == "c" and value >= 39:
            return True
        if unit != "c" and 102 <= value <= 110:
            return True
    return False


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def save_upload(storage, prefix: str) -> Path | None:
    if storage is None or not storage.filename:
        return None
    suffix = Path(storage.filename).suffix.lower() or ".bin"
    mime = (storage.mimetype or "").lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".aiff", ".aif", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    if suffix not in allowed and "wav" in mime:
        suffix = ".wav"
    elif suffix not in allowed and mime.startswith("audio/"):
        suffix = ".mp3" if "mpeg" in mime or "mp3" in mime else ".wav"
    elif suffix not in allowed:
        suffix = ".bin"
    path = UPLOADS / f"{prefix}-{uuid.uuid4().hex}{suffix}"
    storage.save(path)
    return path


def to_wav(src: Path) -> Path:
    if src.suffix.lower() == ".wav":
        return src
    dst = src.with_suffix(".converted.wav")
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16", str(src), str(dst)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dst


# ---------------------------------------------------------------------------
# Layer 3 — processing modules
# ---------------------------------------------------------------------------

def process_voice(path: Path | None, language: str) -> dict:
    """Voice → speech-to-text. Failure does not stop the rest of the triage."""
    if path is None:
        return {"status": "skipped", "text": "", "detail": "No voice file provided."}
    try:
        import speech_recognition as sr
    except ImportError:
        return {
            "status": "error",
            "text": "",
            "detail": "SpeechRecognition is not installed.",
        }
    try:
        wav_path = to_wav(path)
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(wav_path)) as source:
            audio = recognizer.record(source)
        speech_lang = SPEECH_LANG.get(language, "en-IN")
        text = recognizer.recognize_google(audio, language=speech_lang)
        return {"status": "ok", "text": clip(text, MAX_TEXT), "detail": ""}
    except Exception as exc:
        unknown = exc.__class__.__name__ == "UnknownValueError"
        detail = "No clear speech was detected." if unknown else str(exc)
        return {"status": "error", "text": "", "detail": detail}


def process_report(path: Path | None) -> dict:
    """PDF or image → extracted text. The text is not interpreted as a diagnosis."""
    if path is None:
        return {"status": "skipped", "text": "", "detail": "No report uploaded."}
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages[:8]:
                pages.append(page.extract_text() or "")
            text = "\n".join(pages).strip()
            if not text:
                return {
                    "status": "empty",
                    "text": "",
                    "detail": "No selectable text found. Upload a photo of the report for OCR.",
                }
            return {"status": "ok", "text": clip(text), "detail": "Text extracted from PDF."}
        image = Image.open(path).convert("RGB")
        import pytesseract

        text = pytesseract.image_to_string(image).strip()
        if not text:
            return {
                "status": "empty",
                "text": "",
                "detail": "OCR found no text in the report image.",
            }
        return {"status": "ok", "text": clip(text), "detail": "OCR extracted report text."}
    except Exception as exc:
        return {"status": "error", "text": "", "detail": str(exc)}


class SmallVisualCNN:
    """Same Conv → ReLU → Pool → Conv → ReLU → Pool → Dense network as cnn.ipynb.

    Torch is imported only when an image is classified, so the rest of the
    API can start even if the notebook environment is separate.
    """

    def __init__(self, num_classes: int = 5):
        import torch
        from torch import nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc = nn.Linear(16 * 16 * 16, num_classes)

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = torch.flatten(x, 1)
                return self.fc(x)

        self.torch = torch
        self.net = _Net()

    def load(self, path: Path) -> dict:
        try:
            checkpoint = self.torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = self.torch.load(path, map_location="cpu")
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        return checkpoint

    def predict(self, tensor) -> tuple[str, float, list[float]]:
        with self.torch.no_grad():
            logits = self.net(tensor)
            probs = self.torch.softmax(logits, dim=1)[0]
        scores = [round(float(value), 4) for value in probs]
        index = int(self.torch.argmax(probs).item())
        return VISUAL_LABELS[index], scores[index], scores


def preprocess_image(path: Path):
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def process_image(path: Path | None) -> dict:
    """Resize → normalize → small CNN → visual flag. Never a diagnosis."""
    if path is None:
        return {
            "status": "skipped",
            "label": "",
            "finding": "No image provided.",
            "confidence": None,
            "detail": "No image uploaded.",
        }
    try:
        array = preprocess_image(path)
    except Exception as exc:
        return {
            "status": "error",
            "label": "unclear_image_flag",
            "finding": VISUAL_TEXT["unclear_image_flag"],
            "confidence": None,
            "detail": str(exc),
        }
    preprocess = {
        "resized_to": [IMAGE_SIZE, IMAGE_SIZE],
        "normalized": "pixel values divided by 255",
        "mean_pixel": round(float(array.mean()), 4),
    }
    if not MODEL_PATH.exists():
        return {
            "status": "untrained",
            "label": "unclear_image_flag",
            "finding": (
                "The image was resized and normalized. CNN weights are not trained yet. "
                "Run all cells in cnn.ipynb, then classify the image again."
            ),
            "confidence": None,
            "preprocess": preprocess,
            "detail": "models/visual_cnn.pt is missing.",
        }
    try:
        import torch

        model = SmallVisualCNN(num_classes=len(VISUAL_LABELS))
        model.load(MODEL_PATH)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        label, confidence, scores = model.predict(tensor)
        if confidence < 0.55:
            label = "unclear_image_flag"
        return {
            "status": "ok",
            "label": label,
            "finding": VISUAL_TEXT[label],
            "confidence": confidence,
            "scores": dict(zip(VISUAL_LABELS, scores)),
            "preprocess": preprocess,
            "detail": "Prototype CNN visual flag. Not a diagnosis.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "label": "unclear_image_flag",
            "finding": VISUAL_TEXT["unclear_image_flag"],
            "confidence": None,
            "preprocess": preprocess,
            "detail": str(exc),
        }


# ---------------------------------------------------------------------------
# Layer 4 — multimodal fusion
# ---------------------------------------------------------------------------

def fuse(symptoms: str, voice_text: str, report_text: str, visual: dict, language: str, regional_name: str, english_text: str = "") -> dict:
    combined = "\n".join(part.strip() for part in (symptoms, voice_text) if part and part.strip())
    return {
        "symptoms_text": symptoms.strip(),
        "voice_text": voice_text.strip(),
        "report_text": report_text.strip(),
        "visual": visual,
        "language": language,
        "regional_language_name": regional_name.strip(),
        "combined_text": combined,
        "english_text": english_text.strip(),
    }


def translation_dest(language: str, regional_name: str = "") -> str | None:
    if language in LANGUAGE_CODES:
        return LANGUAGE_CODES[language]
    name = (regional_name or "").strip().lower()
    if not name:
        return None
    try:
        from googletrans import LANGUAGES
    except ImportError:
        return None
    for code, label in LANGUAGES.items():
        if label.lower() == name:
            return code
    return None


def translate_text(text: str, dest: str) -> str:
    """Translate with googletrans. Returns the original text if translation fails."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    import httpcore

    # googletrans 4.0.0rc1 still references an httpcore name removed in current httpx.
    if not hasattr(httpcore, "SyncHTTPTransport"):
        httpcore.SyncHTTPTransport = object
    from googletrans import Translator

    translator = Translator(http2=False)
    result = translator.translate(cleaned, dest=dest)
    if hasattr(result, "__await__"):
        import asyncio

        result = asyncio.run(result)
    translated = (getattr(result, "text", "") or "").strip()
    return translated or cleaned


def to_english(text: str, language: str, regional_name: str = "") -> dict:
    original = (text or "").strip()
    if not original or language == "English":
        return {"text": original, "status": "skipped", "engine": "googletrans", "detail": "Already English."}
    try:
        english = translate_text(original, "en")
        return {"text": english, "status": "ok", "engine": "googletrans", "detail": ""}
    except Exception as exc:
        return {
            "text": original,
            "status": "error",
            "engine": "googletrans",
            "detail": exc.__class__.__name__,
        }


def translate_patient_message(message: str, language: str, regional_name: str = "") -> tuple[str, dict]:
    dest = translation_dest(language, regional_name)
    if not dest or dest == "en":
        return message, {"status": "skipped", "engine": "googletrans", "detail": "Patient message stays in English."}
    try:
        translated = translate_text(message, dest)
        return translated, {"status": "ok", "engine": "googletrans", "dest": dest, "detail": ""}
    except Exception as exc:
        return message, {
            "status": "error",
            "engine": "googletrans",
            "detail": exc.__class__.__name__,
        }


def groq_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def groq_answer_model() -> str:
    chosen = os.getenv("GROQ_ANSWER_MODEL", "").strip()
    if chosen:
        return chosen
    configured = os.getenv("GROQ_MODEL", "").strip()
    blocked = ("prompt-guard", "whisper", "orpheus", "safeguard")
    if configured and not any(part in configured for part in blocked):
        return configured
    return "openai/gpt-oss-20b"


def parse_json_object(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    required = ("summary_text", "referral_note", "patient_message", "likely_condition")
    if not all(isinstance(data.get(key), str) and data.get(key).strip() for key in required):
        return None
    if not isinstance(data.get("prescription"), str):
        return None
    data["brand_medicines"] = clean_brand_medicines(data.get("brand_medicines"))
    return data


def drop_unverified_dose(text: str, ctx: dict) -> str:
    """Remove a made-up strength when the case does not include an age."""
    if has_age(patient_source_text(ctx)):
        return text
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|iu)\b", "", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\bevery\s+\d+\s*(?:to|-)?\s*\d*\s*(?:hours?|hrs?)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    return cleaned.strip()


def clean_brand_medicines(raw) -> list[dict]:
    if not isinstance(raw, list):
        return []
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        generic = str(item.get("generic") or "").strip()
        brands = item.get("brands") or []
        if isinstance(brands, str):
            brands = [part.strip() for part in brands.split(",") if part.strip()]
        brands = [str(name).strip() for name in brands if str(name).strip()]
        generic_names = {generic.lower(), "paracetamol", "acetaminophen"}
        if generic.lower() not in {"paracetamol", "acetaminophen"}:
            generic_names = {generic.lower()}
        brands = [name for name in brands if name.lower() not in generic_names]
        use = str(item.get("use") or "").strip()
        if generic and brands:
            cleaned.append({"generic": generic, "brands": brands[:6], "use": use})
    return cleaned


def groq_answer(ctx: dict) -> dict | None:
    """Ask Groq for a condition and treatment grounded in this case."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    from groq import Groq

    extracted = ctx.get("extracted") or {}
    visual = ctx.get("visual") or {}
    voice = (ctx.get("modules") or {}).get("voice") or {}
    facts = {
        "patient_words": ctx.get("combined_text") or "",
        "voice_transcript": voice.get("text") or ctx.get("voice_text") or "",
        "english_rendering": ctx.get("english_text") or "",
        "complaint": extracted.get("complaint") or "Not provided",
        "symptoms": extracted.get("symptoms") or [],
        "additional_symptoms": extracted.get("additional_symptoms") or [],
        "duration": extracted.get("duration") or "Not provided",
        "report_text": ctx.get("report_text") or "No report text extracted",
        "visual_label": visual.get("label") or "",
        "visual_finding": visual.get("finding") or "No image provided",
        "missing_information": ctx.get("missing_information") or [],
        "urgency_flag": ctx.get("urgency_flag"),
        "language": ctx.get("language") or "English",
    }
    system = (
        "You are the clinical response model for a multimodal triage assistant. "
        "Read the JSON. It may include typed symptoms, a voice transcript, report text, and an image flag. "
        "Use every source that is present. Ignore a source only when it says it was not provided. "
        "Name the most likely condition supported by those inputs, plus a short reason. "
        "Then give a suggested treatment. Name medicines only when the symptoms support them. "
        "Do not invent age, weight, allergies, temperature, oxygen, duration, or lab values. "
        "Never write a milligram amount, a tablet strength, or a dosing interval when age is missing. "
        "If you cannot name a medicine, prescription must still say what to do, such as cleaning, monitoring, or emergency care. "
        "Do not leave prescription empty. "
        "If urgency_flag is Urgent Professional Review, the treatment is emergency care now, not a home medicine list, "
        "and brand_medicines must be an empty array. "
        "Otherwise name the generic medicines that fit these symptoms, and brand_medicines must contain one object per medicine. "
        "Each object has generic, brands, and use. brands is three to five different product names, not the generic name itself. "
        "For fever without emergency signs, include paracetamol and its brand names. "
        "Do not invent a milligram dose, and do not put a dose in brand_medicines. "
        "The image flag is a visual clue. "
        "Keep urgency_flag unchanged. "
        "summary_text, referral_note, likely_condition, and prescription are in English. "
        "patient_message is two or three short English sentences naming the likely condition and what to do. "
        "Return only JSON with keys summary_text, referral_note, patient_message, likely_condition, prescription, and brand_medicines."
    )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=groq_answer_model(),
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
        ],
    )
    raw = response.choices[0].message.content or ""
    drafted = parse_json_object(raw)
    if drafted and prompt_guard_flagged(drafted["summary_text"]):
        return None
    return drafted


def prompt_guard_flagged(text: str) -> bool:
    """Optional Groq prompt-guard check. It takes one user message and returns a score."""
    model = os.getenv("GROQ_MODEL", "").strip()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or "prompt-guard" not in model:
        return False
    from groq import Groq

    response = Groq(api_key=api_key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": clip(text, 1500)}],
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        score = float(raw.split()[0])
    except (ValueError, IndexError):
        return False
    return score >= 0.5


def scrape_page_text(url: str) -> str:
    """Read visible text from a page with Selenium. Not used for clinical answers."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(20)
        driver.get(url)
        return (driver.find_element("tag name", "body").text or "").strip()
    finally:
        driver.quit()


# ---------------------------------------------------------------------------
# Layer 5 — LangChain orchestration
# Each function is one chain step from the architecture diagram.
# ---------------------------------------------------------------------------

def patient_source_text(ctx: dict) -> str:
    return "\n".join(
        part
        for part in (
            ctx.get("combined_text") or "",
            ctx.get("english_text") or "",
            ctx.get("report_text") or "",
        )
        if part
    )


def extract_information(ctx: dict) -> dict:
    text = patient_source_text(ctx)
    symptoms = find_symptoms(text)
    duration = find_duration(text) or find_duration(ctx.get("english_text") or "")
    if symptoms:
        complaint = symptoms[0]
        additional = symptoms[1:]
    elif text.strip():
        complaint = re.split(r"[\n.]", text.strip())[0][:180]
        additional = []
    elif (ctx.get("report_text") or "").strip():
        complaint = "Information provided mainly through an uploaded report"
        additional = []
    elif ctx.get("visual", {}).get("status") not in {None, "skipped"}:
        complaint = "Information provided mainly through an image"
        additional = []
    else:
        complaint = "Not provided"
        additional = []
    return {
        "complaint": complaint,
        "symptoms": symptoms,
        "additional_symptoms": additional,
        "duration": duration,
    }


def detect_missing(ctx: dict) -> list[str]:
    """List gaps for the healthcare worker. Do not invent the missing values."""
    searchable = patient_source_text(ctx)
    missing = []
    if not has_age(searchable):
        missing.append("Age")
    if not has_temperature(searchable):
        missing.append("Temperature")
    if not has_oxygen(searchable):
        missing.append("Oxygen level")
    if not (ctx.get("extracted") or {}).get("duration"):
        missing.append("Duration")
    return missing


def analyze_urgency(ctx: dict) -> str:
    """Map the case to a review priority. This is not a diagnosis."""
    text = "\n".join(
        [
            patient_source_text(ctx),
            ctx.get("visual", {}).get("finding") or "",
        ]
    )
    if matches_any(text, URGENT_PATTERNS):
        return "Urgent Professional Review"
    if matches_any(text, PRIORITY_PATTERNS) or high_temperature(text):
        return "Priority Review"
    symptoms = set((ctx.get("extracted") or {}).get("symptoms") or [])
    duration = (ctx.get("extracted") or {}).get("duration") or ""
    short = bool(re.search(r"\b[1-2]\s*(day|days|hour|hours|दिन)\b", duration, flags=re.IGNORECASE))
    if symptoms and symptoms.issubset(MILD_ONLY) and short:
        return "Routine Review"
    return "Needs Review"


def clinical_from_findings(ctx: dict) -> dict:
    """Used only when Groq does not answer. Built from this case's extracted inputs."""
    extracted = ctx.get("extracted") or {}
    symptoms = extracted.get("symptoms") or []
    visual = (ctx.get("visual") or {}).get("label") or ""
    urgency = ctx.get("urgency_flag") or ""
    report = (ctx.get("report_text") or "").strip()
    missing = ctx.get("missing_information") or []
    if urgency == "Urgent Professional Review":
        return {
            "likely_condition": "Possible emergency based on the symptoms that were reported",
            "prescription": "Contact local emergency services now. Do not rely on a home medicine list.",
            "brand_medicines": [],
        }
    if "Fever" in symptoms and ("Cough" in symptoms or "Cold" in symptoms):
        condition = "Possible viral respiratory infection"
    elif "Fever" in symptoms:
        condition = "Possible febrile illness"
    elif "Headache" in symptoms:
        condition = "Possible primary headache"
    elif visual == "visible_redness_flag":
        condition = "Possible local skin inflammation, based on the image flag"
    elif visual == "wound_or_surface_break_flag":
        condition = "Possible open wound or surface break, based on the image flag"
    elif visual == "visible_swelling_flag":
        condition = "Possible localized swelling, based on the image flag"
    elif symptoms:
        condition = "Possible illness related to " + ", ".join(symptoms).lower()
    elif report:
        condition = "The uploaded report needs a clinician to interpret the values"
    else:
        condition = "Not enough information to name a condition"
    care = ["Rest and take fluids."]
    if any(item in symptoms for item in ("Fever", "Pain", "Headache", "Body ache")):
        care.append("A clinician can consider an antipyretic or pain reliever.")
    if "Age" in missing:
        care.append("No milligram dose is set because age was not provided.")
    if visual == "wound_or_surface_break_flag":
        care.append("Clean the area and ask a clinician whether it needs closure or an antibiotic.")
    if visual == "visible_redness_flag":
        care.append("Keep the area clean. A clinician should decide if a skin medicine is needed.")
    if report:
        care.append("Include the report values in that review.")
    return {"likely_condition": condition, "prescription": " ".join(care), "brand_medicines": []}


def render_summary(ctx: dict) -> str:
    extracted = ctx["extracted"]
    additional = extracted["additional_symptoms"] or ["None reported"]
    missing = ctx["missing_information"] or ["None noted from the provided text"]
    report = ctx.get("report_text") or "No report text extracted"
    visual = (ctx.get("visual") or {}).get("finding") or "No image provided"
    lines = [
        "PATIENT SUMMARY",
        "",
        "Complaint:",
        extracted["complaint"],
        "",
        "Duration:",
        extracted["duration"] or "Not provided",
        "",
        "Additional symptoms:",
        *[f"- {item}" for item in additional],
        "",
        "Report information:",
        clip(report, 700),
        "",
        "Visual information:",
        visual,
        "",
        "Missing information:",
        *[f"- {item}" for item in missing],
        "",
        "Urgency:",
        ctx["urgency_flag"],
        "",
        "This summary organizes the information provided. It is not a diagnosis or a prescription.",
    ]
    return "\n".join(lines)


def render_triage_note(ctx: dict) -> str:
    extracted = ctx["extracted"]
    language = ctx.get("language") or "English"
    if language == "Other regional language" and ctx.get("regional_language_name"):
        language = ctx["regional_language_name"]
    missing = ctx["missing_information"] or ["None noted from the provided text"]
    return "\n".join(
        [
            "REFERRAL / TRIAGE NOTE",
            "",
            "Prepared for: Doctor, nurse, or medical officer",
            f"Patient language: {language}",
            f"Urgency flag: {ctx['urgency_flag']}",
            "",
            "What the patient reported:",
            clip(ctx.get("combined_text") or extracted["complaint"], 700),
            "",
            "Voice transcript:",
            ctx.get("voice_text") or "Not provided",
            "",
            "Report text extracted:",
            clip(ctx.get("report_text") or "Not provided", 700),
            "",
            "Visual flag:",
            (ctx.get("visual") or {}).get("finding") or "Not provided",
            "",
            "Missing information to collect:",
            *[f"- {item}" for item in missing],
            "",
            "Request:",
            "Please review this organized information and make the clinical decision.",
            "This note is not a diagnosis and does not recommend treatment.",
        ]
    )


def step_extract(ctx: dict) -> dict:
    ctx["extracted"] = extract_information(ctx)
    return ctx


def step_missing(ctx: dict) -> dict:
    ctx["missing_information"] = detect_missing(ctx)
    return ctx


def step_urgency(ctx: dict) -> dict:
    ctx["urgency_flag"] = analyze_urgency(ctx)
    return ctx


def step_write(ctx: dict) -> dict:
    drafted = None
    try:
        drafted = groq_answer(ctx)
    except Exception as exc:
        ctx["answer_error"] = exc.__class__.__name__
    if drafted:
        ctx["summary_text"] = drafted["summary_text"].strip()
        ctx["referral_note"] = drafted["referral_note"].strip()
        ctx["likely_condition"] = drafted["likely_condition"].strip()
        ctx["prescription"] = drafted["prescription"].strip()
        if not ctx["prescription"]:
            ctx["prescription"] = clinical_from_findings(ctx)["prescription"]
        ctx["prescription"] = drop_unverified_dose(ctx["prescription"], ctx)
        ctx["brand_medicines"] = clean_brand_medicines(drafted.get("brand_medicines"))
        english_message = drop_unverified_dose(drafted["patient_message"].strip(), ctx)
        ctx["answer_engine"] = "Groq"
    else:
        clinical = clinical_from_findings(ctx)
        ctx["summary_text"] = render_summary(ctx)
        ctx["referral_note"] = render_triage_note(ctx)
        ctx["likely_condition"] = clinical["likely_condition"]
        ctx["prescription"] = clinical["prescription"]
        ctx["brand_medicines"] = clinical["brand_medicines"]
        english_message = (
            f"{clinical['likely_condition']}. {clinical['prescription']}"
        )
        if ctx.get("urgency_flag") == "Urgent Professional Review":
            english_message = f"{english_message} {EMERGENCY_LINE_EN}"
        ctx["answer_engine"] = "template"
    translated, translation = translate_patient_message(
        english_message,
        ctx.get("language") or "English",
        ctx.get("regional_language_name") or "",
    )
    ctx["patient_message"] = translated
    ctx["patient_translation"] = translation
    return ctx


def finalize(ctx: dict) -> dict:
    extracted = ctx["extracted"]
    visual = ctx.get("visual") or {}
    report_text = ctx.get("report_text") or ""
    result = {
        "id": ctx["id"],
        "created_at": ctx["created_at"],
        "language": ctx.get("language") or "English",
        "regional_language_name": ctx.get("regional_language_name") or "",
        "patient_summary": {
            "complaint": extracted["complaint"],
            "duration": extracted["duration"] or "Not provided",
            "additional_symptoms": extracted["additional_symptoms"],
            "report_information": report_text or "No report text extracted",
            "visual_information": visual.get("finding") or "No image provided",
            "missing_information": ctx["missing_information"],
            "urgency": ctx["urgency_flag"],
        },
        "symptoms": extracted["symptoms"],
        "duration": extracted["duration"] or "Not provided",
        "report_findings": report_text or "No report text extracted",
        "visual_findings": visual.get("finding") or "No image provided",
        "visual_detail": visual,
        "missing_information": ctx["missing_information"],
        "urgency_flag": ctx["urgency_flag"],
        "summary_text": ctx["summary_text"],
        "referral_note": ctx["referral_note"],
        "likely_condition": ctx.get("likely_condition") or "",
        "prescription": ctx.get("prescription") or "",
        "brand_medicines": ctx.get("brand_medicines") or [],
        "patient_message": ctx["patient_message"],
        "answer_engine": ctx.get("answer_engine") or "template",
        "answer_error": ctx.get("answer_error") or "",
        "english_text": ctx.get("english_text") or "",
        "patient_translation": ctx.get("patient_translation") or {},
        "orchestration": "LangChain + Groq" if LANGCHAIN_AVAILABLE and ctx.get("answer_engine") == "Groq" else (
            "LangChain" if LANGCHAIN_AVAILABLE else "local chain fallback"
        ),
        "orchestration_steps": [
            "Information Extraction",
            "Missing Information Detection",
            "Urgency Analysis",
        "Summary Generation",
        "Condition and treatment",
        "Triage Note Generation",
        ],
        "modules": ctx.get("modules") or {},
        "professional_review": {
            "status": "pending",
            "role": None,
            "note": None,
        },
        "final_decision": "Pending. A qualified professional makes the final clinical decision.",
    }
    return apply_guardrails(result)


def apply_guardrails(result: dict) -> dict:
    """Attach the disclaimer. Do not replace the Groq condition or treatment."""
    result["disclaimer"] = DISCLAIMER
    result["decision_owner"] = "Confirm with a qualified healthcare professional before taking medicine"
    result["ai_role"] = (
        "Reads the submitted text, voice, report, and image, then asks Groq for a likely "
        "condition and a suggested treatment."
    )
    result["guardrail_triggered"] = False
    return result


def get_chain():
    global CHAIN
    if CHAIN is not None:
        return CHAIN
    steps = [step_extract, step_missing, step_urgency, step_write, finalize]
    if LANGCHAIN_AVAILABLE:
        chain = RunnableLambda(steps[0])
        for step in steps[1:]:
            chain = chain | RunnableLambda(step)
        CHAIN = chain
    else:
        class LocalChain:
            def invoke(self, payload):
                ctx = payload
                for step in steps:
                    ctx = step(ctx)
                return ctx

        CHAIN = LocalChain()
    return CHAIN


def speech_code(language: str, regional_name: str, translation_status: str) -> tuple[str, str]:
    """Pick a gTTS voice for the message that was actually produced."""
    if translation_status == "error":
        return "en", "Translation failed, so the audio is English."
    if language in LANGUAGE_CODES:
        return LANGUAGE_CODES[language], ""
    code = translation_dest(language, regional_name)
    if code and code != "en":
        return code, ""
    if language == "English":
        return "en", ""
    return "en", "Audio uses English. The selected language is shown as text."


def synthesize_speech(message: str, language: str, regional_name: str = "", translation_status: str = "") -> dict:
    """Structured text → gTTS audio for accessibility."""
    from gtts import gTTS

    spoken = message
    lang, note = speech_code(language, regional_name, translation_status)
    buffer = BytesIO()
    gTTS(text=spoken, lang=lang).write_to_fp(buffer)
    return {
        "audio_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "lang": lang,
        "spoken_text": spoken,
        "note": note,
    }


def attach_audio(result: dict) -> dict:
    try:
        translation = result.get("patient_translation") or {}
        audio = synthesize_speech(
            result["patient_message"],
            result["language"],
            result.get("regional_language_name") or "",
            translation.get("status") or "",
        )
        result["audio_base64"] = audio["audio_base64"]
        result["audio_mime"] = "audio/mp3"
        result["tts_engine"] = "gTTS"
        result["tts_language"] = audio["lang"]
        result["audio_note"] = audio["note"]
    except Exception as exc:
        result["audio_base64"] = ""
        result["audio_mime"] = "audio/mp3"
        result["tts_engine"] = "gTTS"
        result["tts_language"] = ""
        result["audio_note"] = (
            "Audio could not be created. The text summary is still available. "
            f"({exc.__class__.__name__})"
        )
    return result


def case_path(case_id: str) -> Path:
    safe = re.sub(r"[^a-f0-9]", "", case_id.lower())
    return CASES / f"{safe}.json"


def history_path() -> Path:
    return DATA / "history.json"


def read_history() -> list[dict]:
    path = history_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_case(result: dict) -> None:
    stored = dict(result)
    case_path(result["id"]).write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    index = {
        "id": result["id"],
        "created_at": result["created_at"],
        "urgency_flag": result["urgency_flag"],
        "complaint": result["patient_summary"]["complaint"],
        "language": result["language"],
        "review_status": result["professional_review"]["status"],
    }
    with LOCK:
        history = [item for item in read_history() if item["id"] != result["id"]]
        history.insert(0, index)
        history_path().write_text(
            json.dumps(history[:20], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (DATA / "latest_result.json").write_text(
            json.dumps(stored, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_case(case_id: str) -> dict | None:
    path = case_path(case_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_triage(symptoms: str, language: str, regional_name: str, voice_path, report_path, image_path) -> dict:
    voice = process_voice(voice_path, language)
    report = process_report(report_path)
    image = process_image(image_path)
    voice_text = voice.get("text") or ""
    symptom_text = clip(symptoms, MAX_TEXT)
    original_words = "\n".join(part for part in (symptom_text, voice_text) if part.strip())
    rendered = to_english(original_words, language, regional_name)
    context = fuse(
        symptoms=symptom_text,
        voice_text=voice_text,
        report_text=report.get("text") or "",
        visual=image,
        language=language,
        regional_name=regional_name,
        english_text="" if language == "English" else rendered.get("text") or "",
    )
    context["id"] = uuid.uuid4().hex
    context["created_at"] = now_stamp()
    context["modules"] = {"voice": voice, "report": report, "image": image, "translation": rendered}
    result = get_chain().invoke(context)
    result = attach_audio(result)
    write_case(result)
    return result


def inputs_present(symptoms: str, voice, report, image) -> bool:
    return bool((symptoms or "").strip() or voice or report or image)


# ---------------------------------------------------------------------------
# Flask API
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return jsonify(
        {
            "service": "Multimodal Healthcare Triage Assistant",
            "role": "Flask bridge between Streamlit and the Python processing modules",
            "endpoints": [
                "POST /ingest",
                "POST /voice",
                "POST /report",
                "POST /image",
                "POST /triage",
                "GET /result",
                "GET /history",
                "POST /review",
            ],
            "langchain": LANGCHAIN_AVAILABLE,
            "disclaimer": DISCLAIMER,
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "langchain": LANGCHAIN_AVAILABLE,
            "groq": groq_configured(),
            "cnn_weights": MODEL_PATH.exists(),
        }
    )


@app.post("/ingest")
def ingest():
    payload = request.get_json(silent=True) or request.form
    symptoms = clip(payload.get("symptoms") or "", MAX_TEXT)
    language = normalize_language(payload.get("language"))
    if not symptoms:
        return jsonify({"error": "Enter symptom text before calling /ingest."}), 400
    return jsonify({"status": "received", "language": language, "symptoms": symptoms})


@app.post("/voice")
def voice_route():
    path = save_upload(request.files.get("voice") or request.files.get("file"), "voice")
    language = normalize_language(request.form.get("language"))
    return jsonify(process_voice(path, language))


@app.post("/report")
def report_route():
    path = save_upload(request.files.get("report") or request.files.get("file"), "report")
    return jsonify(process_report(path))


@app.post("/image")
def image_route():
    path = save_upload(request.files.get("image") or request.files.get("file"), "image")
    return jsonify(process_image(path))


@app.post("/triage")
def triage():
    payload = request.form if request.form else (request.get_json(silent=True) or {})
    symptoms = payload.get("symptoms") or ""
    language = normalize_language(payload.get("language"))
    regional_name = (payload.get("regional_language_name") or "").strip()
    voice_path = save_upload(request.files.get("voice"), "voice") if request.files else None
    report_path = save_upload(request.files.get("report"), "report") if request.files else None
    image_path = save_upload(request.files.get("image"), "image") if request.files else None
    if not inputs_present(symptoms, voice_path, report_path, image_path):
        return jsonify({"error": "Add symptoms, a voice note, a report, or an image."}), 400
    result = build_triage(symptoms, language, regional_name, voice_path, report_path, image_path)
    return jsonify(result)


@app.get("/result")
def latest_result():
    path = DATA / "latest_result.json"
    if not path.exists():
        return jsonify({"error": "No triage result yet."}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.get("/result/<case_id>")
def one_result(case_id: str):
    case = load_case(case_id)
    if case is None:
        return jsonify({"error": "Case not found."}), 404
    return jsonify(case)


@app.get("/history")
def history():
    return jsonify(read_history())


@app.post("/review")
def review():
    payload = request.get_json(silent=True) or {}
    case_id = (payload.get("id") or "").strip()
    role = (payload.get("role") or "").strip()
    note = clip(payload.get("note") or "", 2000)
    if role not in REVIEWER_ROLES:
        return jsonify({"error": "Reviewer role must be Doctor, Nurse, or Medical Officer."}), 400
    if len(note) < 3:
        return jsonify({"error": "Enter the professional's own clinical note."}), 400
    case = load_case(case_id)
    if case is None:
        return jsonify({"error": "Case not found."}), 404
    case["professional_review"] = {
        "status": "reviewed",
        "role": role,
        "note": note,
        "reviewed_at": now_stamp(),
    }
    case["final_decision"] = note
    case["decision_owner"] = role
    write_case(case)
    return jsonify(case)


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "File is too large. Please upload a file under 16 MB."}), 413


@app.errorhandler(Exception)
def unexpected(exc):
    from werkzeug.exceptions import HTTPException

    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("triage request failed")
    return jsonify({"error": "The request could not be processed.", "detail": exc.__class__.__name__}), 500


_SERVER_THREAD = None


def backend_is_up(port: int = PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def serve_in_background() -> None:
    """Start Flask once, in this process, so Streamlit and the API share one command."""
    global _SERVER_THREAD
    ensure_dirs()
    if backend_is_up() or (_SERVER_THREAD is not None and _SERVER_THREAD.is_alive()):
        return

    def _run() -> None:
        app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False, threaded=True)

    _SERVER_THREAD = threading.Thread(target=_run, name="triage-flask", daemon=True)
    _SERVER_THREAD.start()
    for _ in range(50):
        if backend_is_up():
            return
        time.sleep(0.1)


if __name__ == "__main__":
    ensure_dirs()
    print("Multimodal Healthcare Triage Assistant")
    print(f"Flask API: http://127.0.0.1:{PORT}")
    print("One command for the full app: python app.py")
    print(DISCLAIMER)
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)
