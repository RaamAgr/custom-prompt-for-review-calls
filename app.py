# app.py — COMPLETE QA AUDIT SYSTEM (FULL CODE)
# -----------------------------------------------------------------------------
# FEATURES:
# 1. Main Page Prompt Editor.
# 2. Aggressive JSON Parsing (Detects JSON inside Markdown/Text).
# 3. Visual QA Dashboard (Metrics, Progress Bars, Banners).
# 4. Robust Gemini Integration (Resumable Uploads, Polling, Retries).
# 5. Multi-Threaded Processing (Up to 128 workers).
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
import json
import os
import time
import logging
import mimetypes
import tempfile
import random
import math 
import html
import re
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
MODEL_NAME = "gemini-2.0-flash-exp" 

# Streaming download chunk size (8KB)
DOWNLOAD_CHUNK_SIZE = 8192

# Job-level retry configuration
MAX_WORKER_RETRIES = 3 
WORKER_BACKOFF_BASE = 2 

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("transcriber")

# --- UI STYLING (CSS) ---
BASE_CSS = """
<style>
/* Dashboard Metric Card Style */
.metric-box {
    background-color: var(--card-bg, #f8f9fa);
    border: 1px solid var(--border-color, #eee);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-title { 
    font-size: 0.85em; 
    font-weight: 700; 
    color: var(--meta-color, #666); 
    text-transform: uppercase; 
    letter-spacing: 0.5px; 
    margin-bottom: 5px;
}
.metric-value { 
    font-size: 2.0em; 
    font-weight: 800; 
    margin: 0; 
    line-height: 1.2;
}
.metric-sub { 
    font-size: 0.8em; 
    color: var(--meta-color, #888); 
    margin-top: 4px;
}

/* Global Theme Variables */
.dark-theme {
    --card-bg: #1e2126;
    --border-color: #333;
    --meta-color: #9aa0a6;
    color: #e6eef3;
}
.light-theme {
    --card-bg: #ffffff;
    --border-color: #e6e6e6;
    --meta-color: #666666;
    color: #111;
}

/* Expander Styling */
.streamlit-expanderHeader {
    font-weight: 600;
}
</style>
"""

st.set_page_config(page_title="QA Auditor", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)


# --- NETWORK UTILITIES ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    """Sleeps with jitter to prevent rate limit spikes."""
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 30)
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 0.5, **kwargs) -> requests.Response:
    """Robust HTTP request wrapper with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
            # Treat 429 (Rate Limit) and 5xx (Server Errors) as retryable
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning("Transient HTTP %s from %s. Retrying...", resp.status_code, url)
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning("RequestException: %s. Retrying...", str(e))
            _sleep_with_jitter(backoff_base, attempt)
            
    if last_exc: raise last_exc
    raise Exception("Retries exhausted")


# --- GOOGLE UPLOAD PIPELINE ---

def detect_extension_and_mime(url_path: str, header_content_type: Optional[str]) -> (str, str):
    """Detects audio format from URL or Content-Type header."""
    COMMON_AUDIO_MIME = {
        ".mp3": "audio/mpeg", ".wav": "audio/wave", ".m4a": "audio/mp4",
        ".aac": "audio/aac", ".ogg": "audio/ogg", ".webm": "audio/webm", ".flac": "audio/flac"
    }
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    
    # 1. Trust extension if known
    if ext and ext in COMMON_AUDIO_MIME: 
        return ext, COMMON_AUDIO_MIME[ext]
    
    # 2. Trust Header
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        for k, v in COMMON_AUDIO_MIME.items():
            if v == ctype: return k, ctype
        guessed = mimetypes.guess_extension(ctype)
        if guessed: return guessed.lower(), ctype
        
    # 3. Fallback
    return ".mp3", "audio/mpeg"

def initiate_upload(api_key: str, display_name: str, mime_type: str, file_size: int) -> str:
    """Starts Resumable Upload Session."""
    url = f"{UPLOAD_URL}?uploadType=resumable&key={api_key}"
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size), "X-Goog-Upload-Header-Content-Type": mime_type,
    }
    payload = json.dumps({"file": {"display_name": display_name}})
    resp = make_request_with_retry("POST", url, headers=headers, data=payload)
    if resp.status_code not in (200, 201): raise Exception(f"Init failed: {resp.text}")
    return resp.headers.get("X-Goog-Upload-URL")

def upload_bytes(upload_url: str, file_path: str, mime_type: str) -> Dict[str, Any]:
    """Uploads file bytes."""
    file_size = os.path.getsize(file_path)
    headers = {
        "Content-Type": mime_type or "application/octet-stream", "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"
    }
    # Try POST first
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=300)
    # Fallback to PUT if 400
    if resp.status_code == 400: 
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, headers=headers, data=f, timeout=300)
    if resp.status_code not in (200, 201): raise Exception(f"Upload failed: {resp.text}")
    return resp.json().get("file", resp.json())

def wait_for_active(api_key: str, file_name: str, timeout_seconds: int = 300) -> bool:
    """Polls until file is ACTIVE."""
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    start = time.time()
    while True:
        resp = make_request_with_retry("GET", url)
        if resp.status_code == 200:
            state = resp.json().get("state")
            if state == "ACTIVE": return True
            if state == "FAILED": raise Exception(f"Processing failed: {resp.json()}")
        time.sleep(2)
        if time.time() - start > timeout_seconds: raise Exception("Timed out waiting for file.")

def delete_file(api_key: str, file_name: str):
    """Deletes file from Gemini storage."""
    try: requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}", timeout=20)
    except: pass


# --- GEMINI GENERATION (JSON ENFORCED) ---

def generate_transcript(api_key: str, file_uri: str, mime_type: str, prompt: str) -> str:
    """Generates content using Gemini, enforcing JSON mode where possible."""
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    
    # Auto-append JSON instruction if missing to be safe
    if "JSON" not in prompt.upper():
        prompt += "\n\nCRITICAL: Output strictly valid JSON."

    payload = {
        "contents": [{"parts": [{"text": prompt}, {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}]}],
        "generationConfig": {
            "temperature": 0.2, 
            "maxOutputTokens": 8192,
            "response_mime_type": "application/json" # Hints the model to output JSON
        }
    }
    
    # Retry loop for empty responses
    for attempt in range(3):
        resp = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code != 200: return f"API ERROR {resp.status_code}: {resp.text}"
        
        try:
            body = resp.json()
            candidates = body.get("candidates") or []
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [])[0].get("text", "")
                if text.strip(): return text
        except: pass
        
        time.sleep(2 * (attempt + 1))
        
    return "{}" # Return empty JSON obj if all fails


# --- DATA PREPARATION ---

def prepare_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows for processing based on URL presence."""
    final_rows = []
    for index, row in df.iterrows():
        r = row.copy()
        url_val = r.get('recording_url')
        if pd.notna(url_val) and str(url_val).strip() != "":
            r['processing_action'] = 'TRANSCRIBE'
            r['status'] = 'Pending'
        else:
            r['processing_action'] = 'SKIP'
            r['transcript'] = "{}"
            r['status'] = 'Skipped'
        final_rows.append(r)
    return pd.DataFrame(final_rows).reset_index(drop=True)


# --- WORKER FUNCTION ---

def process_single_row(index: int, row: pd.Series, api_key: str, final_prompt: str, keep_remote: bool) -> Dict[str, Any]:
    """Processes a single row: Download -> Upload -> Generate -> Clean."""
    result = {
        "index": index, "mobile_number": str(row.get("mobile_number", "Unknown")),
        "recording_url": row.get("recording_url"), "transcript": row.get("transcript", ""),
        "status": row.get("status", "Pending"), "error": row.get("error", None)
    }
    if row.get("processing_action") == "SKIP": return result

    for attempt in range(MAX_WORKER_RETRIES):
        tmp_path, file_info = None, None
        try:
            # 1. Download
            r = make_request_with_retry("GET", result["recording_url"], stream=True)
            if r.status_code != 200: raise Exception(f"Download failed: {r.status_code}")
            ext, mime = detect_extension_and_mime(urlparse(result["recording_url"]).path, r.headers.get("content-type"))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                for chunk in r.iter_content(DOWNLOAD_CHUNK_SIZE): tmp.write(chunk)
                tmp_path = tmp.name
            
            # 2. Upload
            unique_name = f"rec_{index}_{int(time.time())}_{random.randint(100,999)}{ext}"
            up_url = initiate_upload(api_key, unique_name, mime, os.path.getsize(tmp_path))
            file_info = upload_bytes(up_url, tmp_path, mime)
            
            wait_for_active(api_key, file_info["name"])
            
            # 3. Generate
            transcript = generate_transcript(api_key, file_info["uri"], mime, final_prompt)
            result["transcript"] = transcript
            
            # 4. Check Success
            if "API ERROR" in transcript:
                if attempt < MAX_WORKER_RETRIES - 1: raise Exception("API Error")
                result["status"] = "❌ Error"
            else:
                result["status"] = "✅ Success"
                return result

        except Exception as e:
            if attempt == MAX_WORKER_RETRIES - 1:
                result["status"], result["error"] = "❌ Failed", str(e)
            else:
                _sleep_with_jitter(WORKER_BACKOFF_BASE, attempt)
        finally:
            if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
            if file_info and not keep_remote: delete_file(api_key, file_info["name"])
    return result


# --- DASHBOARD & PARSING UTILS ---

def aggressive_json_parse(text: str) -> Optional[Dict]:
    """
    Robustly extracts JSON from text. 
    Handles cases where Gemini adds ```json fences or preamble text.
    """
    if not text: return None
    try:
        # Strategy 1: Find outer braces
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1:
            json_candidate = text[start : end + 1]
            return json.loads(json_candidate)
        
        return None
    except json.JSONDecodeError:
        return None

def render_dashboard_card(data: Dict):
    """Renders the Visual QA Dashboard for a single row."""
    
    # 1. Overall Status Banner
    status = data.get("overall_status", "UNKNOWN")
    if status == "PASS":
        st.success(f"### Overall Status: {status} ✅")
    elif status == "FAIL":
        st.error(f"### Overall Status: {status} ❌")
    else:
        st.info(f"### Overall Status: {status}")

    # 2. Visual Metrics Grid
    scores = data.get("scores", {})
    if scores:
        st.markdown("#### 📊 Performance Metrics")
        
        # Split scores into chunks of 4 for grid layout
        items = list(scores.items())
        chunks = [items[i:i + 4] for i in range(0, len(items), 4)]

        for chunk in chunks:
            cols = st.columns(len(chunk))
            for i, (key, details) in enumerate(chunk):
                with cols[i]:
                    val = details.get("score", 0.0)
                    conf = details.get("confidence_score", 0.0)
                    
                    # Color Logic: High=Green, Mid=Yellow, Low=Red
                    color = "#28a745" if val >= 0.8 else "#ffc107" if val >= 0.5 else "#dc3545"
                    
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-title">{key.replace('_', ' ')}</div>
                            <div class="metric-value" style="color: {color}">{val}</div>
                            <div class="metric-sub">Conf: {int(conf*100)}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

    st.divider()

    # 3. Detailed Reasoning (Accordions)
    st.markdown("#### 📝 Detailed Reasoning")
    if scores:
        for key, details in scores.items():
            s_val = details.get("score", 0)
            # Icon logic
            icon = "🟢" if s_val >= 0.8 else "🔴" if s_val < 0.5 else "🟡"
            
            with st.expander(f"{icon} {key.replace('_', ' ').title()}"):
                st.write(f"**Reasoning:** {details.get('reasoning', 'No reasoning provided.')}")
                ts = details.get("timestamp_of_issue")
                if ts: st.warning(f"⚠️ Issue detected at timestamp: **{ts}**")


# --- DEFAULT PROMPT TEMPLATE ---
DEFAULT_AUDIT_PROMPT = """Analyze this call recording for Quality Assurance.

OUTPUT FORMAT (STRICT JSON):
{
  "scores": {
    "latency": {
      "score": <float 0.0-1.0>,
      "confidence_score": <float 0.0-1.0>,
      "timestamp_of_issue": "mm:ss",
      "reasoning": "..."
    },
    "repetition": { ... },
    "politeness": { ... },
    "hallucination": { ... },
    "language": { ... },
    "interruption_handling": { ... }
  },
  "overall_status": "PASS" or "FAIL"
}
"""

# --- MAIN APP ENTRY POINT ---

def main():
    if "processed_results" not in st.session_state: st.session_state.processed_results = []
    if "final_df" not in st.session_state: st.session_state.final_df = pd.DataFrame()

    # --- SIDEBAR: SETTINGS ONLY ---
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Gemini API Key", type="password")
        max_workers = st.slider("Concurrency (Threads)", 1, 128, 4)
        keep_remote = st.checkbox("Keep Google Files", False)
        
        st.divider()
        language_mode = st.selectbox("Language Context", ["English", "Hindi", "Hinglish"], index=2)
        
        st.divider()
        theme_choice = st.radio("UI Theme", ["Light", "Dark"], 0, horizontal=True)

    # Theme Injection
    theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
    st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)

    # --- MAIN PAGE ---
    st.title("🤖 QA Call Auditor")
    
    # 1. Prompt Editor (On Main Page)
    with st.expander("📝 Edit System Prompt & JSON Structure", expanded=True):
        prompt_input = st.text_area(
            "System Prompt", 
            value=DEFAULT_AUDIT_PROMPT, 
            height=250,
            help="Define the strict JSON structure here. The dashboard will adapt to whatever keys you put in 'scores'."
        )

    # 2. File Upload
    st.write("### 📂 Upload Excel Batch")
    uploaded_files = st.file_uploader("Select .xlsx files", type=["xlsx"], accept_multiple_files=True)

    # 3. Progress & Status
    progress_bar = st.empty()
    status_text = st.empty()
    
    # 4. Start Button & Logic
    if st.button("🚀 Start Audit Batch", type="primary"):
        if not api_key: st.error("Please enter API Key in sidebar."); st.stop()
        if not uploaded_files: st.error("Please upload at least one file."); st.stop()
        
        # Load Data
        all_dfs = [pd.read_excel(f) for f in uploaded_files]
        raw_df = pd.concat(all_dfs, ignore_index=True)
        if "recording_url" not in raw_df.columns: 
            st.error("Missing 'recording_url' column in Excel."); st.stop()

        # Prep Data
        df_ready = prepare_all_rows(raw_df)
        final_prompt = prompt_input + f"\n\nContext Language: {language_mode}"
        
        st.session_state.processed_results = []
        status_text.info(f"Auditing {len(df_ready)} calls...")
        
        # Execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_row, idx, row, api_key, final_prompt, keep_remote): idx for idx, row in df_ready.iterrows()}
            completed = 0
            for future in as_completed(futures):
                st.session_state.processed_results.append(future.result())
                completed += 1
                progress_bar.progress(completed / len(df_ready))
                status_text.markdown(f"**Processed {completed}/{len(df_ready)}**")

        st.session_state.final_df = pd.DataFrame(st.session_state.processed_results)
        status_text.success("Batch Processing Complete!")

    # --- DASHBOARD RESULTS VIEWER ---
    df = st.session_state.final_df
    if not df.empty:
        st.markdown("---")
        st.markdown("## 📊 Audit Results")
        
        # Search & Filters
        c1, c2, c3 = st.columns([3, 1, 1])
        q = c1.text_input("Search JSON/Phone/URL...")
        stat = c2.selectbox("Filter Status", ["All", "Success", "Failed"])
        
        view = df.copy()
        if stat == "Success": view = view[view["status"].str.contains("Success")]
        if stat == "Failed": view = view[view["status"].str.contains("Error|Failed")]
        if q: view = view[view.astype(str).apply(lambda x: x.str.contains(q, case=False)).any(axis=1)]

        # Export
        buf = BytesIO()
        view.to_excel(buf, index=False)
        st.download_button("📥 Download Excel Report", buf.getvalue(), "audit_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.write(f"Showing {len(view)} results:")

        # Render Individual Cards
        for i, row in view.iterrows():
            with st.expander(f"📞 {row.get('mobile_number')} | {row.get('status')}"):
                st.markdown(f"**URL:** [{row.get('recording_url')}]({row.get('recording_url')})")
                
                raw_txt = row.get("transcript", "")
                
                # Try Parsing
                json_data = aggressive_json_parse(raw_txt)
                
                if json_data:
                    render_dashboard_card(json_data)
                    with st.expander("View Raw JSON Output"):
                        st.code(raw_txt, language="json")
                else:
                    st.warning("⚠️ Could not parse JSON. Showing raw text below:")
                    st.code(raw_txt)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
