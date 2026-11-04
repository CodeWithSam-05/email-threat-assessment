from pathlib import Path
import re
import unicodedata

from django.conf import settings
from django.shortcuts import render

import pandas as pd
import PyPDF2
from google import genai

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .forms import MessageForm


BASE_DIR = Path(__file__).resolve().parent.parent

POSSIBLE_DATASET_PATHS = [
    BASE_DIR / "emails.csv",
    BASE_DIR / "detector" / "emails.csv",
    BASE_DIR / "emails.xlsx",
    BASE_DIR / "detector" / "emails.xlsx",
]


def load_email_dataset():
    dataset = None
    last_error = None

    for path in POSSIBLE_DATASET_PATHS:
        if not path.exists():
            continue

        try:
            if path.suffix.lower() == ".csv":
                dataset = pd.read_csv(path, encoding="latin1")
            elif path.suffix.lower() in [".xlsx", ".xls"]:
                dataset = pd.read_excel(path)
            else:
                continue

            if dataset is not None and not dataset.empty:
                break
        except Exception as e:
            last_error = e
            dataset = None

    if dataset is None or dataset.empty:
        print("DATASET ERROR:", last_error, flush=True)
        return pd.DataFrame(columns=["text", "spam"])

    dataset.columns = dataset.columns.str.strip()
    lowered = {col.lower(): col for col in dataset.columns}

    if "text" in lowered and "spam" in lowered:
        dataset = dataset.rename(columns={
            lowered["text"]: "text",
            lowered["spam"]: "spam",
        })
    elif "v1" in lowered and "v2" in lowered:
        dataset = dataset.rename(columns={
            lowered["v1"]: "spam",
            lowered["v2"]: "text",
        })
    elif "label" in lowered and "text" in lowered:
        dataset = dataset.rename(columns={
            lowered["label"]: "spam",
            lowered["text"]: "text",
        })
    elif len(dataset.columns) >= 2:
        dataset = dataset.iloc[:, :2]
        dataset.columns = ["spam", "text"]

    if "text" not in dataset.columns or "spam" not in dataset.columns:
        print("DATASET ERROR: Required columns not found.", flush=True)
        return pd.DataFrame(columns=["text", "spam"])

    dataset = dataset[["text", "spam"]].dropna()
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset["spam"] = dataset["spam"].astype(str).str.strip().str.lower()
    dataset = dataset[dataset["text"] != ""]

    dataset["spam"] = dataset["spam"].replace({
        "1": "spam",
        "0": "ham",
        "ham ": "ham",
        "spam ": "spam",
    })

    dataset = dataset[dataset["spam"].isin(["spam", "ham"])]

    print("DATASET LOADED", flush=True)
    print("COLUMNS:", dataset.columns.tolist(), flush=True)
    print("LABELS:", dataset["spam"].unique(), flush=True)

    return dataset


dataset = load_email_dataset()

vectorizer = CountVectorizer(stop_words="english")
model = None

if not dataset.empty:
    X = vectorizer.fit_transform(dataset["text"])
    y = dataset["spam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"EMAIL MODEL ACCURACY: {accuracy:.4f}", flush=True)
else:
    print("MODEL NOT TRAINED: Dataset is empty.", flush=True)



def predict_message(message):
    if model is None:
        return "Ham"

    message = str(message).strip()
    if not message:
        return "Ham"

    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    return "Spam" if str(prediction).lower() == "spam" else "Ham"



def extract_pdf_text(file_obj):
    try:
        reader = PyPDF2.PdfReader(file_obj)
        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        final_text = "\n".join(text_parts).strip()
        print("EXTRACTED PDF TEXT:", final_text[:1000], flush=True)

        return final_text

    except Exception as e:
        print("PDF TEXT EXTRACTION ERROR:", e, flush=True)
        return ""


def normalize_pdf_text(pdf_text):
    """Normalize common whitespace, Unicode, and URL obfuscation variants."""
    text = unicodedata.normalize("NFKC", str(pdf_text))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"(?i)\bhxxps?\s*:\s*//", "https://", text)
    text = re.sub(r"(?i)\bwww\s*\[\s*dot\s*\]\s*", "www.", text)
    text = re.sub(
        r"(?i)\[\s*(?:dot|\.)\s*\]|\(\s*dot\s*\)|\s+dot\s+",
        ".",
        text,
    )
    text = re.sub(r"(?i)\s*\[\s*:\s*\]\s*", ":", text)
    return re.sub(r"\s+", " ", text).strip()


def _indicator(kind, value, severity, evidence=None):
    return {
        "type": kind,
        "value": value,
        "severity": severity,
        "evidence": evidence or value,
    }


def analyze_pdf_indicators(pdf_text):
    """Extract structured, explainable threat signals from PDF text."""
    text = normalize_pdf_text(pdf_text)
    lowered = text.lower()
    indicators = []

    url_pattern = r"(?i)\b(?:https?://|www\.)[^\s<>'\"]+"
    urls = []
    for match in re.findall(url_pattern, text):
        url = match.rstrip(".,;:!?)]}")
        if url not in urls:
            urls.append(url)
            indicators.append(_indicator("url", url, "medium"))

    domain_pattern = r"(?i)(?<![@\w])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}\b"
    domains = []
    for domain in re.findall(domain_pattern, text):
        domain = domain.lower().rstrip(".")
        if domain not in domains:
            domains.append(domain)
            severity = "high" if any(domain in url.lower() for url in urls) else "medium"
            indicators.append(_indicator("domain", domain, severity))

    email_pattern = r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,63}\b"
    for address in dict.fromkeys(re.findall(email_pattern, lowered)):
        indicators.append(_indicator("email_address", address, "low"))

    attachment_pattern = (
        r"(?i)\b[\w .-]+\.(?:exe|scr|js|vbs|ps1|bat|cmd|msi|dll|iso|img|"
        r"docm|xlsm|pptm|zip|rar|7z)\b"
    )
    for attachment in dict.fromkeys(match.strip() for match in re.findall(attachment_pattern, text)):
        indicators.append(_indicator("attachment", attachment, "high"))

    attachment_terms = (
        r"\b(?:open|enable macros?|run|execute|download)\b.{0,60}"
        r"\b(?:attachment|attached file|document)\b"
    )
    for match in dict.fromkeys(re.findall(attachment_terms, lowered)):
        indicators.append(_indicator("attachment_instruction", match, "high"))

    deobfuscated = re.sub(r"[\s._-]+", "", lowered)
    command_patterns = {
        "powershell": r"(?:powershell|p0wershell)",
        "command_execution": r"(?:cmd(?:/c|\.exe/c)|commandprompt|shellcode)",
        "encoded_command": r"(?:-enc(?:odedcommand)?|frombase64string|base64)",
        "payload_execution": r"(?:execute|run|download)(?:the)?payload",
        "security_bypass": r"(?:disable|bypass)(?:your)?(?:antivirus|security|defender)",
        "malware": r"(?:malware|ransomware|trojan|exploit)",
    }
    for kind, pattern in command_patterns.items():
        for match in dict.fromkeys(re.findall(pattern, deobfuscated)):
            indicators.append(_indicator("obfuscated_command" if kind == "powershell" else kind, match, "critical"))

    signal_patterns = {
        "urgency": r"\b(?:urgent|immediately|final warning|act now|expires today|within \d+ hours?)\b",
        "payment_request": r"\b(?:payment|pay now|invoice|refund|customs charge|bank details|upi pin)\b",
        "credential_request": r"\b(?:password|credentials?|otp|pin|verify your account|login)\b",
        "account_threat": r"\b(?:suspended|blocked|account closure|service suspension|unusual activity)\b",
    }
    for kind, pattern in signal_patterns.items():
        for match in dict.fromkeys(re.findall(pattern, lowered)):
            indicators.append(_indicator(kind, match, "medium"))

    malicious_count = sum(
        indicator["severity"] == "critical" for indicator in indicators
    )
    high_count = sum(indicator["severity"] == "high" for indicator in indicators)
    medium_count = sum(indicator["severity"] == "medium" for indicator in indicators)
    score = min(100, malicious_count * 35 + high_count * 20 + medium_count * 10)
    if malicious_count:
        verdict = "malicious"
    elif high_count or medium_count >= 2:
        verdict = "suspicious"
    else:
        verdict = "safe"

    return {
        "verdict": verdict,
        "risk_score": score,
        "indicators": indicators,
        "indicator_counts": {
            "total": len(indicators),
            "critical": malicious_count,
            "high": high_count,
            "medium": medium_count,
            "low": sum(i["severity"] == "low" for i in indicators),
        },
        "source": "local_structured_detection",
    }


def rule_based_pdf_check(pdf_text):
    """Return the legacy label while structured detection remains available."""
    return analyze_pdf_indicators(pdf_text)["verdict"]


def pdf_threat_detection(pdf_text):
    if not pdf_text or not str(pdf_text).strip():
        print("EMPTY PDF TEXT", flush=True)
        return {
            "verdict": "suspicious",
            "risk_score": 50,
            "indicators": [],
            "indicator_counts": {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0},
            "source": "empty_document",
        }

    report = analyze_pdf_indicators(pdf_text)
    rule_result = report["verdict"]
    print("RULE-BASED PDF RESULT:", rule_result, flush=True)

    # For project reliability, immediately trust clear rule hits
    if rule_result in ["malicious", "suspicious"]:
        return report

    api_key = getattr(settings, "GEMINI_API_KEY", None)

    if api_key is None:
        print("GEMINI API KEY NOT SET", flush=True)
        return report

    api_key = str(api_key).strip()

    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        print("INVALID GEMINI API KEY", flush=True)
        return report

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are a cybersecurity classifier.

Classify the following PDF text into exactly one category:
safe
suspicious
malicious

Rules:
- safe = normal notes, report, study material, invoice, documentation
- suspicious = phishing, urgent warning, account verification, fake payment request, suspicious links
- malicious = malware execution, payload delivery, disable antivirus, exploit code, run dangerous commands

PDF text:
{str(pdf_text)[:3000]}

Return exactly one lowercase word only:
safe
suspicious
malicious
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        output = ""
        if response is not None and getattr(response, "text", None):
            output = response.text.strip().lower()

        print("PDF MODEL RAW OUTPUT:", output, flush=True)

        if output == "malicious":
            report["verdict"] = "malicious"
        elif output == "suspicious":
            report["verdict"] = "suspicious"
        elif output == "safe":
            report["verdict"] = "safe"

        elif "malicious" in output:
            report["verdict"] = "malicious"
        elif "suspicious" in output:
            report["verdict"] = "suspicious"
        elif "safe" in output:
            report["verdict"] = "safe"

        else:
            print("UNEXPECTED OUTPUT, USING RULE RESULT", flush=True)
        report["source"] = "gemini"
        return report

    except Exception as e:
        print("PDF DETECTION ERROR:", e, flush=True)
        return report



def Home(request):
    result = None
    pdf_result = None
    pdf_message = None
    pdf_report = None
    form = MessageForm()

    print("REQUEST METHOD:", request.method, flush=True)

    if request.method == "POST":
        print("POST KEYS:", list(request.POST.keys()), flush=True)
        print("FILES KEYS:", list(request.FILES.keys()), flush=True)

        if "email_check" in request.POST:
            print("EMAIL FORM HIT", flush=True)
            form = MessageForm(request.POST)

            if form.is_valid():
                message = form.cleaned_data["text"]
                result = predict_message(message)

        elif "pdf_check" in request.POST:
            print("PDF FORM HIT", flush=True)

            if "pdf_file" not in request.FILES:
                pdf_message = "No PDF uploaded."
            else:
                pdf_file = request.FILES["pdf_file"]

                if not pdf_file.name.lower().endswith(".pdf"):
                    pdf_message = "Upload only PDF files."
                else:
                    pdf_text = extract_pdf_text(pdf_file)

                    if not pdf_text:
                        pdf_result = "suspicious"
                        pdf_message = "Could not extract text from this PDF."
                    else:
                        pdf_report = pdf_threat_detection(pdf_text)
                        pdf_result = pdf_report["verdict"]
                        indicator_total = pdf_report["indicator_counts"]["total"]
                        pdf_message = (
                            f"PDF Threat Result: {pdf_result.upper()} "
                            f"({indicator_total} indicators matched)"
                        )

    return render(request, "home.html", {
        "form": form,
        "result": result,
        "pdf_result": pdf_result,
        "pdf_message": pdf_message,
        "pdf_report": pdf_report,
    })