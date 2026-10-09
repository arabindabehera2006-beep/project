# Multimodal Healthcare Triage Assistant

Organizes text, voice, reports, and images, then asks Groq for a likely condition and a suggested treatment. Confirm both with a clinician before taking medicine.

Patient → Streamlit → Flask → speech-to-text / OCR / CNN → fusion → LangChain → Groq → googletrans → gTTS → professional review

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit page. Also starts Flask. |
| `backend.py` | API and processing. |
| `cnn.ipynb` | Trains the CNN. 50 epochs, 95% held-out target. |
| `.env` | Groq key. Do not commit. |
| `.streamlit/config.toml` | Theme and local address. |
| `models/visual_cnn.pt` | CNN weights. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install "googletrans==4.0.0rc1" --no-deps
brew install tesseract
cp .env.example .env
```

`.env`:

```bash
GROQ_API_KEY=your_groq_key_here
GROQ_ANSWER_MODEL=openai/gpt-oss-20b
GROQ_MODEL=meta-llama/llama-prompt-guard-2-22m
```

`GROQ_ANSWER_MODEL` writes the summary. `GROQ_MODEL` is the prompt-guard check. Install googletrans with `--no-deps` so it does not downgrade `httpx`.

## Run

```bash
source .venv/bin/activate
python app.py
```

| Service | URL |
|---|---|
| Streamlit | http://127.0.0.1:8501 |
| Flask | http://127.0.0.1:5050 |

Stop with `Ctrl+C`.

API only:

```bash
python backend.py
```

Streamlit only (Flask still starts from `app.py`):

```bash
streamlit run app.py
```

## Streamlit page

Sidebar: Flask URL, backend status, CNN weights, LangChain, Groq key.

**Patient / Healthcare Worker**

- Language: English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu, or another regional language
- Text symptoms
- Record or upload voice
- Upload report (PDF or image)
- Upload image
- **Use sample case** — “Fever and weakness for 3 days”, Hindi
- **Analyze** — sends everything to `POST /triage`

Result: urgency, complaint, duration, symptoms, missing information, report text, visual flag, summary, referral note, patient message, audio.

Urgency colors: red Urgent, yellow Priority, blue Needs Review, green Routine.

**Professional Review**

- Pick a case
- Role: Doctor, Nurse, or Medical Officer
- Write the clinical note
- **Save professional review**

## Processing

| Input | Step |
|---|---|
| Voice | Speech-to-text |
| Report | PDF text or Tesseract OCR |
| Image | Resize 64×64, divide by 255, small CNN |
| All | Fusion, then LangChain: extract, missing info, urgency, summary, triage note, guardrail |
| Text | Groq writes the English answer from the submitted case. googletrans renders the patient message in the selected language. gTTS speaks that message. |

CNN flags: no obvious concern, redness, swelling, wound or surface break, unclear image. Not a diagnosis.

Urgency: Routine Review, Needs Review, Priority Review, Urgent Professional Review.

Missing fields, when not in the text: Age, Temperature, Oxygen level, Duration.

Sample case result: Fever, 3 days, Weakness, missing Age / Temperature / Oxygen level, Needs Review.

## Train CNN

```bash
source .venv/bin/activate
jupyter notebook cnn.ipynb
```

Run all cells. Saves `models/visual_cnn.pt`. Restart `python app.py` after training.

## API

Base: `http://127.0.0.1:5050`

| Method | Path | Use |
|---|---|---|
| GET | `/health` | Status |
| POST | `/ingest` | Text and language |
| POST | `/voice` | Audio file |
| POST | `/report` | PDF or image |
| POST | `/image` | CNN flag |
| POST | `/triage` | Full case |
| GET | `/result` | Latest case |
| GET | `/result/<id>` | One case |
| GET | `/history` | Recent cases |
| POST | `/review` | Professional note |

```bash
curl http://127.0.0.1:5050/health

curl -X POST http://127.0.0.1:5050/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms":"Fever and weakness for 3 days","language":"Hindi"}'

curl -X POST http://127.0.0.1:5050/review \
  -H "Content-Type: application/json" \
  -d '{"id":"CASE_ID","role":"Doctor","note":"Reviewed."}'
```

Upload limit: 16 MB.

## Fixes

| Problem | Fix |
|---|---|
| Page will not open | Keep `python app.py` running. Open port 8501. |
| Backend offline | Restart `python app.py`. |
| Port in use | Free 5050 or 8501, then restart. |
| No Groq key | Set `GROQ_API_KEY` in `.env` and restart. |
| Empty report text | `brew install tesseract`, or upload a photo. |
| No CNN weights | Run `cnn.ipynb`. |
| No audio | gTTS needs internet. An unsupported regional language is spoken in English. |
