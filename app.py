"""
Multimodal Healthcare Triage Assistant.

One command starts the Streamlit page and the Flask backend:

    python app.py
"""

import base64
import html
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st


def inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and os.environ.get("TRIAGE_LAUNCHED") != "1" and not inside_streamlit():
    os.environ["TRIAGE_LAUNCHED"] = "1"
    script_path = Path(__file__).resolve()
    script = str(script_path)
    venv_python = script_path.parent / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    os.execv(
        python,
        [
            python,
            "-m",
            "streamlit",
            "run",
            script,
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
        ],
    )

import backend

backend.serve_in_background()

API_DEFAULT = "http://127.0.0.1:5050"
LOCAL_APIS = {API_DEFAULT, "http://localhost:5050"}
LANGUAGES = list(backend.LANGUAGES)

st.set_page_config(
    page_title="Multimodal Healthcare Triage Assistant",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #0f172a;
        --muted: #475569;
        --line: #dbe3ef;
        --panel: #ffffff;
        --soft: #f4f7fb;
        --accent: #0f766e;
        --accent-soft: #ccfbf1;
    }
    .stApp {
        background: linear-gradient(180deg, #eef3f8 0%, #f8fafc 220px, #ffffff 100%);
        color: var(--ink);
    }
    h1, h2, h3, h4 {
        color: var(--ink) !important;
        letter-spacing: -0.02em;
        font-weight: 700 !important;
    }
    div[data-testid="stWidgetLabel"] p {
        font-weight: 700 !important;
        color: var(--ink) !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        font-weight: 600;
        border: 1px solid var(--line) !important;
        background: var(--panel) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border: 1px solid var(--line) !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }
    .hero-bar {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        padding: 0.2rem 0 1rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1rem;
    }
    .hero-bar h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 760;
    }
    .hero-bar p {
        margin: 0.35rem 0 0 0;
        color: var(--muted);
        font-weight: 600;
    }
    .input-shell {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem 0.4rem 1.1rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
    }
    .section-title {
        font-size: 0.78rem;
        font-weight: 760;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0 0 0.75rem 0;
    }
    .metric-card {
        background: var(--soft);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        min-height: 96px;
    }
    .metric-card .label {
        font-size: 0.74rem;
        font-weight: 760;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }
    .metric-card .value {
        font-size: 1.02rem;
        font-weight: 720;
        color: var(--ink);
        line-height: 1.45;
    }
    .brand-row {
        background: var(--soft);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .brand-row .generic {
        font-weight: 760;
        color: var(--ink);
        margin-bottom: 0.2rem;
    }
    .brand-row .brands {
        font-weight: 650;
        color: var(--accent);
    }
    .brand-row .use {
        color: var(--muted);
        font-weight: 600;
        margin-top: 0.15rem;
    }
    .status-chip {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-weight: 740;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }
    .status-urgent { background: #fee2e2; color: #991b1b; }
    .status-priority { background: #ffedd5; color: #9a3412; }
    .status-routine { background: #dcfce7; color: #166534; }
    .status-needs { background: #e0f2fe; color: #075985; }
    .empty-panel {
        color: var(--muted);
        font-weight: 650;
        padding: 1.4rem 0.2rem;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: var(--soft);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.35rem;
    }
    button[data-baseweb="tab"] {
        font-weight: 720 !important;
        border-radius: 9px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "symptoms" not in st.session_state:
    st.session_state.symptoms = ""
if "language" not in st.session_state or st.session_state.language not in LANGUAGES:
    st.session_state.language = "English"
if "regional_language_name" not in st.session_state:
    st.session_state.regional_language_name = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None


class ApiResult:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if not isinstance(self._payload, (dict, list)):
            raise ValueError("unreadable")
        return self._payload


def api_base() -> str:
    return st.session_state.get("api_url", API_DEFAULT).rstrip("/")


def use_local_backend() -> bool:
    return api_base() in LOCAL_APIS


def http_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _from_http(response: requests.Response) -> ApiResult:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return ApiResult(response.status_code, payload)


def api_get(path: str, timeout: float = 10) -> ApiResult:
    if use_local_backend():
        response = backend.app.test_client().get(path)
        return ApiResult(response.status_code, response.get_json(silent=True))
    return _from_http(http_session().get(f"{api_base()}{path}", timeout=timeout))


def api_post_json(path: str, payload: dict, timeout: float = 20) -> ApiResult:
    if use_local_backend():
        response = backend.app.test_client().post(path, json=payload)
        return ApiResult(response.status_code, response.get_json(silent=True))
    return _from_http(
        http_session().post(f"{api_base()}{path}", json=payload, timeout=timeout)
    )


def api_post_triage(data: dict, files: dict | None) -> ApiResult:
    if use_local_backend():
        form = dict(data)
        if files:
            for key, (name, raw, mime) in files.items():
                form[key] = (BytesIO(raw), name, mime)
        response = backend.app.test_client().post("/triage", data=form)
        return ApiResult(response.status_code, response.get_json(silent=True))
    return _from_http(
        http_session().post(
            f"{api_base()}/triage",
            data=data,
            files=files or None,
            timeout=180,
        )
    )


def file_payload(uploaded, fallback_name: str):
    if uploaded is None:
        return None
    name = getattr(uploaded, "name", "") or fallback_name
    if "." not in Path(name).name:
        name = fallback_name
    mime = getattr(uploaded, "type", None) or "application/octet-stream"
    return Path(name).name, uploaded.getvalue(), mime


def pdf_text(value: str) -> str:
    text = str(value or "").replace("\r", "\n")
    text = text.replace("\\", "/").replace("(", "[").replace(")", "]")
    text = text.encode("ascii", "replace").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip() or "-"


def wrap_pdf(text: str, width: int = 88) -> list[str]:
    cleaned = pdf_text(text)
    lines: list[str] = []
    for paragraph in cleaned.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("-")
            continue
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > width and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines or ["-"]


def build_case_pdf(result: dict) -> bytes:
    brands = result.get("brand_medicines") or []
    brand_lines = []
    for item in brands:
        if not isinstance(item, dict):
            continue
        names = ", ".join(str(name) for name in (item.get("brands") or []) if str(name).strip())
        use = item.get("use") or ""
        generic = item.get("generic") or ""
        brand_lines.append(f"{generic}: {names}" + (f" - {use}" if use else ""))
    if not brand_lines:
        brand_lines = ["None listed"]
    summary = result.get("patient_summary") or {}
    symptoms = result.get("symptoms") or summary.get("additional_symptoms") or []
    if isinstance(symptoms, list):
        symptom_text = ", ".join(str(item) for item in symptoms) if symptoms else "None reported"
    else:
        symptom_text = str(symptoms)
    missing = result.get("missing_information") or []
    missing_text = ", ".join(str(item) for item in missing) if missing else "None"
    language = result.get("language") or "English"
    regional = (result.get("regional_language_name") or "").strip()
    language_label = f"{language} - {regional}" if regional else language
    detail = result.get("visual_detail") or {}
    confidence = detail.get("confidence")
    visual = result.get("visual_findings") or "No image provided"
    if isinstance(confidence, (int, float)):
        visual = f"{visual} ({confidence:.0%})"
    sections = [
        ("Urgency", result.get("urgency_flag") or "Needs Review"),
        ("Condition", result.get("likely_condition") or "-"),
        ("Treatment", result.get("prescription") or "-"),
        ("Brand medicines", "\n".join(brand_lines)),
        ("Complaint", summary.get("complaint") or "Not provided"),
        ("Duration", summary.get("duration") or result.get("duration") or "Not provided"),
        ("Symptoms", symptom_text),
        ("Missing details", missing_text),
        ("Language", language_label),
        ("Report findings", result.get("report_findings") or "No report text extracted"),
        ("Image findings", visual),
        ("Patient message", result.get("patient_message") or "-"),
        ("Clinical note", result.get("summary_text") or "-"),
        ("Referral note", result.get("referral_note") or "-"),
    ]
    if result.get("created_at"):
        sections.insert(0, ("Case time", result.get("created_at") or "-"))
    lines = ["CLINICAL RESPONSE", "Generated from the multimodal triage case", ""]
    for title, body in sections:
        lines.append(title.upper())
        lines.extend(wrap_pdf(str(body)))
        lines.append("")
    page_size = 48
    pages = [lines[index : index + page_size] for index in range(0, len(lines), page_size)] or [["CLINICAL RESPONSE"]]
    objects: list[bytes] = []

    def add(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    bold_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    content_ids = []
    for page in pages:
        y = 800
        commands = ["BT"]
        for line in page:
            if not str(line).strip():
                y -= 10
                continue
            safe = pdf_text(line)
            font = bold_id if safe.isupper() and len(safe) < 40 else font_id
            size = 13 if font == bold_id and safe == "CLINICAL RESPONSE" else (11 if font == bold_id else 10)
            commands.append(f"/F{font} {size} Tf")
            commands.append(f"1 0 0 1 48 {y} Tm")
            commands.append(f"({safe}) Tj")
            y -= 15 if font == font_id else 18
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii", "replace")
        content_ids.append(
            add(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
        )
    pages_id = len(objects) + len(content_ids) + 1
    page_ids = []
    for content_id in content_ids:
        page_ids.append(
            add(
                b"<< /Type /Page /Parent "
                + str(pages_id).encode("ascii")
                + b" 0 R /MediaBox [0 0 595 842] /Contents "
                + str(content_id).encode("ascii")
                + b" 0 R /Resources << /Font << /F"
                + str(font_id).encode("ascii")
                + b" "
                + str(font_id).encode("ascii")
                + b" 0 R /F"
                + str(bold_id).encode("ascii")
                + b" "
                + str(bold_id).encode("ascii")
                + b" 0 R >> >> >>"
            )
        )
    kids = b" ".join(str(page_id).encode("ascii") + b" 0 R" for page_id in page_ids)
    actual_pages_id = add(
        b"<< /Type /Pages /Count " + str(len(page_ids)).encode("ascii") + b" /Kids [" + kids + b"] >>"
    )
    if actual_pages_id != pages_id:
        for page_id in page_ids:
            objects[page_id - 1] = objects[page_id - 1].replace(
                f"/Parent {pages_id} 0 R".encode("ascii"),
                f"/Parent {actual_pages_id} 0 R".encode("ascii"),
            )
    catalog_id = add(b"<< /Type /Catalog /Pages " + str(actual_pages_id).encode("ascii") + b" 0 R >>")
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(output)


def urgency_class(flag: str) -> str:
    if flag == "Urgent Professional Review":
        return "status-urgent"
    if flag == "Priority Review":
        return "status-priority"
    if flag == "Routine Review":
        return "status-routine"
    return "status-needs"


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{html.escape(label)}</div>
            <div class="value">{html.escape(value or "—")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_audio(result: dict) -> None:
    audio = result.get("audio_base64") or ""
    note = (result.get("audio_note") or "").strip()
    if not audio:
        if note:
            st.caption(note)
        return
    try:
        st.audio(base64.b64decode(audio), format=result.get("audio_mime") or "audio/mp3")
    except Exception:
        st.caption(note or "Audio could not be played. The written message is shown above.")
        return
    engine = result.get("tts_engine") or "gTTS"
    st.caption(f"Audio by {engine}. {note}".strip())


def render_brands(result: dict) -> None:
    brands = result.get("brand_medicines") or []
    if not brands:
        st.markdown('<div class="empty-panel">No brand medicines listed for this case.</div>', unsafe_allow_html=True)
        return
    for item in brands:
        names = ", ".join(item.get("brands") or [])
        use = item.get("use") or ""
        use_html = f'<div class="use">{html.escape(use)}</div>' if use else ""
        st.markdown(
            f"""
            <div class="brand-row">
                <div class="generic">{html.escape(item.get("generic", "") or "")}</div>
                <div class="brands">{html.escape(names)}</div>
                {use_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def section_diagnosis(result: dict) -> None:
    summary = result.get("patient_summary") or {}
    language = result.get("language") or "English"
    regional = (result.get("regional_language_name") or "").strip()
    language_label = f"{language} — {regional}" if regional else language
    if result.get("created_at"):
        st.caption(result["created_at"])
    top = st.columns(2, gap="medium")
    with top[0]:
        metric_card("Condition", (result.get("likely_condition") or "").strip())
    with top[1]:
        metric_card("Language", language_label)
    st.write("")
    bottom = st.columns(2, gap="medium")
    with bottom[0]:
        metric_card("Complaint", summary.get("complaint") or "Not provided")
    with bottom[1]:
        metric_card(
            "Duration",
            summary.get("duration") or result.get("duration") or "Not provided",
        )


def section_treatment(result: dict) -> None:
    metric_card("Suggested treatment", (result.get("prescription") or "").strip())
    st.write("")
    missing = result.get("missing_information") or []
    missing_text = ", ".join(str(item) for item in missing) if missing else "None"
    metric_card("Missing details", missing_text)


def section_findings(result: dict) -> None:
    summary = result.get("patient_summary") or {}
    symptoms = result.get("symptoms") or summary.get("additional_symptoms") or []
    symptom_text = ", ".join(str(item) for item in symptoms) if symptoms else "None reported"
    detail = result.get("visual_detail") or {}
    confidence = detail.get("confidence")
    visual = result.get("visual_findings") or "No image provided"
    if isinstance(confidence, (int, float)):
        visual = f"{visual} ({confidence:.0%})"
    row = st.columns(2, gap="medium")
    with row[0]:
        metric_card("Symptoms", symptom_text)
        st.write("")
        metric_card("Report", result.get("report_findings") or "No report text extracted")
    with row[1]:
        metric_card("Image", visual)
        st.write("")
        if result.get("english_text"):
            metric_card("English rendering", result.get("english_text") or "")


def section_message(result: dict) -> None:
    metric_card("Patient message", (result.get("patient_message") or "").strip())
    st.write("")
    st.markdown('<div class="section-title">Spoken response</div>', unsafe_allow_html=True)
    render_audio(result)


def pdf_download(result: dict, key_prefix: str, label: str = "Download clinical PDF") -> None:
    try:
        pdf_bytes = build_case_pdf(result)
    except Exception as exc:
        st.error(f"PDF could not be built. {exc.__class__.__name__}: {exc}")
        return
    if not pdf_bytes.startswith(b"%PDF"):
        st.error("PDF could not be built.")
        return
    case_id = result.get("id") or "case"
    st.download_button(
        label,
        data=pdf_bytes,
        file_name=f"clinical-response-{case_id}.pdf",
        mime="application/pdf",
        key=f"pdf-{key_prefix}-{case_id}-{label}",
        use_container_width=True,
    )


def section_report(result: dict, key_prefix: str) -> None:
    summary_text = (result.get("summary_text") or "").strip()
    referral = (result.get("referral_note") or "").strip()
    metric_card("Clinical note", summary_text or "-")
    st.write("")
    metric_card("Referral note", referral or "-")
    st.write("")
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    pdf_download(result, f"{key_prefix}-report", "Download clinical PDF")
    review = result.get("professional_review") or {}
    if review.get("status") == "reviewed":
        st.write("")
        metric_card(
            f"Decision by {review.get('role') or 'reviewer'}",
            result.get("final_decision") or "",
        )


def section_processing(result: dict) -> None:
    modules = result.get("modules") or {}
    problems = []
    for name in ("voice", "report", "image"):
        module = modules.get(name) or {}
        if module.get("status") == "error":
            problems.append(f"{name.title()}: {module.get('detail') or 'could not be read'}")
    if problems:
        st.warning("Some inputs could not be read.\n\n" + "\n".join(problems))
    metric_card("Orchestration", result.get("orchestration") or "not recorded")
    st.write("")
    metric_card("Steps", " → ".join(result.get("orchestration_steps") or []) or "—")
    st.write("")
    cols = st.columns(3, gap="medium")
    for index, name in enumerate(("voice", "report", "image")):
        module = modules.get(name) or {}
        detail = module.get("detail") or module.get("text") or "—"
        with cols[index]:
            metric_card(
                f"{name.title()} · {module.get('status', 'not run')}",
                str(detail),
            )


def render_result(result: dict, key_prefix: str = "main") -> None:
    if not isinstance(result, dict):
        st.error("The backend response could not be read.")
        return

    head, export = st.columns([3.4, 1.1], gap="medium")
    with head:
        st.markdown('<div class="section-title">Clinical response</div>', unsafe_allow_html=True)
        flag = result.get("urgency_flag") or "Needs Review"
        st.markdown(
            f'<div class="status-chip {urgency_class(flag)}">{html.escape(flag)}</div>',
            unsafe_allow_html=True,
        )
    with export:
        pdf_download(result, f"{key_prefix}-top", "Download PDF")

    overview, treatment, brands, findings, message, report, processing = st.tabs(
        [
            "Overview",
            "Treatment",
            "Brand medicines",
            "Findings",
            "Patient message",
            "Report & PDF",
            "Processing",
        ]
    )
    with overview:
        section_diagnosis(result)
    with treatment:
        section_treatment(result)
    with brands:
        render_brands(result)
    with findings:
        section_findings(result)
    with message:
        section_message(result)
    with report:
        section_report(result, key_prefix)
    with processing:
        section_processing(result)


def show_result(result, key_prefix: str = "main") -> None:
    if not result:
        return
    try:
        render_result(result, key_prefix)
    except Exception as exc:
        st.error(f"The case came back, but the page could not draw part of it. {exc}")
        safe = dict(result) if isinstance(result, dict) else {"result": str(result)}
        safe.pop("audio_base64", None)
        st.json(safe)


def submit_case(symptoms: str, language: str, regional_name: str, voice, report, image):
    data = {
        "symptoms": symptoms,
        "language": language,
        "regional_language_name": regional_name,
    }
    files = {}
    voice_file = file_payload(voice, "voice.wav")
    report_file = file_payload(report, report.name if report else "report")
    image_file = file_payload(image, image.name if image else "image")
    if voice_file:
        files["voice"] = voice_file
    if report_file:
        files["report"] = report_file
    if image_file:
        files["image"] = image_file
    if not symptoms.strip() and not files:
        st.warning("Add symptoms, a voice note, a report, or an image.")
        return None
    try:
        with st.spinner("Preparing the clinical response..."):
            response = api_post_triage(data, files or None)
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {api_base()}. {exc.__class__.__name__}.")
        return None
    except Exception as exc:
        st.error(f"The case could not be organized. {exc.__class__.__name__}: {exc}")
        return None
    try:
        payload = response.json()
    except ValueError:
        st.error("The backend returned an unreadable response.")
        return None
    if not isinstance(payload, dict):
        st.error("The backend returned an unreadable response.")
        return None
    if not response.ok:
        st.error(payload.get("error") or "The case could not be organized.")
        if payload.get("detail"):
            st.caption(str(payload["detail"]))
        return None
    return payload


st.markdown(
    """
    <div class="hero-bar">
        <div>
            <h1>Clinical Triage</h1>
            <p>Text, patient speech, PDF report, and image in one structured response.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System")
    st.text_input("Flask API", API_DEFAULT, key="api_url")
    try:
        health_response = api_get("/health", timeout=3)
        health = health_response.json() if health_response.ok else None
    except Exception:
        health = None
    if isinstance(health, dict) and health.get("status") == "ok":
        st.success("Backend connected")
        if use_local_backend():
            st.caption("Local backend in use")
        weight_state = "trained weights found" if health.get("cnn_weights") else "run cnn.ipynb to train"
        st.caption(f"CNN: {weight_state}")
        st.caption("LangChain: available" if health.get("langchain") else "LangChain package missing")
        st.caption("Groq: key loaded from .env" if health.get("groq") else "Groq: add GROQ_API_KEY to .env")
    else:
        st.error("Backend offline")
        st.caption("Restart with: python app.py")
    st.divider()
    st.caption("Patient tab collects inputs. Review tab records the clinical decision.")

patient_tab, review_tab = st.tabs(["Patient intake", "Professional review"])

with patient_tab:
    st.markdown('<div class="section-title">Case input</div>', unsafe_allow_html=True)
    with st.container(border=True):
        sample, language_col, symptom_col, voice_col, pdf_col, image_col, action_col = st.columns(
            [0.75, 1.15, 2.2, 1.35, 1.35, 1.35, 0.95],
            gap="small",
        )
        with sample:
            st.markdown("**Sample**")
            if st.button("Fill", use_container_width=True):
                st.session_state.symptoms = "Fever and weakness for 3 days"
                st.session_state.language = "Hindi"
                st.rerun()
        with language_col:
            st.selectbox("Language", LANGUAGES, key="language")
            if st.session_state.language == "Other regional language":
                st.text_input("Regional language", key="regional_language_name")
        with symptom_col:
            st.text_area(
                "Symptoms",
                height=118,
                placeholder="Fever and weakness for 3 days",
                key="symptoms",
            )
        with voice_col:
            st.markdown("**Patient speech**")
            if hasattr(st, "audio_input"):
                voice = st.audio_input("Record speech", key="patient_speech")
            else:
                voice = None
                st.caption("Restart with: python app.py")
        with pdf_col:
            st.markdown("**PDF report**")
            report = st.file_uploader(
                "PDF or photo",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                key="report_upload",
            )
        with image_col:
            st.markdown("**Image**")
            image = st.file_uploader(
                "Photo",
                type=["png", "jpg", "jpeg", "webp"],
                key="image_upload",
            )
        with action_col:
            st.markdown("**Run**")
            analyze = st.button("Analyze", type="primary", use_container_width=True)

    if analyze:
        payload = submit_case(
            st.session_state.symptoms,
            st.session_state.language,
            st.session_state.regional_language_name,
            voice,
            report,
            image,
        )
        if payload:
            st.session_state.last_result = payload
            st.rerun()

    st.markdown('<div class="section-title">Structured response</div>', unsafe_allow_html=True)
    with st.container(border=True):
        if st.session_state.last_result:
            show_result(st.session_state.last_result, "patient")
        else:
            st.markdown(
                '<div class="empty-panel">Run Analyze to open Overview, Treatment, Brand medicines, Findings, Patient message, Report & PDF, and Processing.</div>',
                unsafe_allow_html=True,
            )

with review_tab:
    st.markdown('<div class="section-title">Case review</div>', unsafe_allow_html=True)
    st.caption("Open an organized case, then record the final clinical decision.")
    try:
        history_response = api_get("/history")
        history = history_response.json() if history_response.ok else []
    except Exception:
        history = []
        st.error(f"Cases could not be loaded from {api_base()}.")

    if not isinstance(history, list) or not history:
        st.info("No organized case yet. Submit patient information on the first tab.")
    else:
        labels = [
            f"{item.get('created_at', '')} — {item.get('complaint', 'Case')} ({item.get('urgency_flag', '')})"
            for item in history
        ]
        selected = st.selectbox("Case", range(len(labels)), format_func=lambda index: labels[index])
        case_id = history[selected]["id"]
        try:
            case_response = api_get(f"/result/{case_id}")
            case = case_response.json() if case_response.ok else None
        except Exception:
            case = None
        if not isinstance(case, dict) or case.get("error"):
            st.error("That case could not be loaded.")
        else:
            with st.container(border=True):
                show_result(case, "review")
            st.divider()
            st.markdown("**Record the professional decision**")
            role = st.selectbox("Reviewer role", ["Doctor", "Nurse", "Medical Officer"], key="review_role")
            note = st.text_area(
                "Final clinical note",
                height=140,
                placeholder="Written by the qualified professional. The assistant does not fill this in.",
                key="review_note",
            )
            if st.button("Save professional review", type="primary"):
                try:
                    saved = api_post_json(
                        "/review",
                        {"id": case_id, "role": role, "note": note},
                    )
                    body = saved.json()
                except requests.RequestException:
                    st.error("The review could not be saved because the backend is offline.")
                except Exception as exc:
                    st.error(f"The review could not be saved. {exc.__class__.__name__}: {exc}")
                else:
                    if saved.ok and isinstance(body, dict):
                        st.session_state.last_result = body
                        st.rerun()
                    else:
                        message = body.get("error") if isinstance(body, dict) else None
                        st.error(message or "The review could not be saved.")
