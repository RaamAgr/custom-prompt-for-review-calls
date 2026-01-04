# app.py — COMPLETE BATCH TRANSCRIBER
# FEATURES: All Rows Processed | Main Screen Prompt | Retry Logic | Gemini 3 Flash Preview

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
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"

# !!! ENSURING MODEL IS THE ONE YOU REQUESTED !!!
MODEL_NAME = "gemini-3-flash-preview"

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

# --- UI STYLING ---
BASE_CSS = """
<style>
.call-card { border: 1px solid var(--border-color, #e6e6e6); border-radius: 10px; padding: 12px; margin-bottom: 12px; background: var(--card-bg, #fff); box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.transcript-box { max-height: 320px; overflow: auto; padding: 8px; border-radius: 6px; background: var(--transcript-bg, #fafafa); border: 1px solid var(--border-color, #eee); font-family: monospace; white-space: pre-wrap; }
.speaker1 { color: #1f77b4; font-weight: 600; display: block; margin-bottom: 4px; }
.speaker2 { color: #d62728; font-weight: 600; display: block; margin-bottom: 4px; }
.other-speech { color: #333; display: block; margin-bottom: 4px; }
.meta-row { font-size: 13px; color: var(--meta-color, #666); margin-bottom: 8px; }
.dark-theme { --card-bg: #0f1115; --transcript-bg: #0b0c0f; --border-color: #222428; --meta-color: #9aa0a6; color: #e6eef3; }
.light-theme { --card-bg: #ffffff; --transcript-bg: #fafafa; --border-color: #e6e6e6; --meta-color: #666666; color: #111; }
</style>
"""

st.set_page_config(page_title="Batch Transcriber", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)

# --- NETWORK UTILITIES ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 30)
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 0.5, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning(f"Transient HTTP {resp.status_code} (attempt {attempt+1})")
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning(f"RequestException (attempt {attempt+1}): {e}")
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
    if last_exc: raise last_exc
    raise Exception("Retries exhausted")

# --- MIME TYPE ---

COMMON_AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wave", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".webm": "audio/webm", ".flac": "audio/flac"
}

def detect_extension_and_mime(url_path: str, header_content_type: Optional[str]) -> (str, str):
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    if ext and ext in COMMON_AUDIO_MIME: return ext, COMMON_AUDIO_MIME[ext]
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        for k, v in COMMON_AUDIO_MIME.items():
            if v == ctype: return k, ctype
        guessed = mimetypes.guess_extension(ctype)
        if guessed: return guessed.lower(), ctype
    return ".mp3", "audio/mpeg"

# --- GOOGLE UPLOAD PIPELINE ---

def initiate_upload(api_key: str, display_name: str, mime_type: str, file_size: int) -> str:
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
    file_size = os.path.getsize(file_path)
    headers = {
        "Content-Type": mime_type or "application/octet-stream", "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"
    }
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=300)
    if resp.status_code == 400:
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, headers=headers, data=f, timeout=300)
    if resp.status_code not in (200, 201): raise Exception(f"Upload failed: {resp.text}")
    j = resp.json()
    return j.get("file", j)

# --- GOOGLE FILE STATUS POLLING ---

def wait_for_active(api_key: str, file_name: str, timeout_seconds: int = 300) -> bool:
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    start = time.time()
    while True:
        resp = make_request_with_retry("GET", url)
        if resp.status_code == 200:
            j = resp.json()
            state = j.get("state")
            if state == "ACTIVE": return True
            if state == "FAILED": raise Exception(f"Processing failed: {j.get('processingError', j)}")
        time.sleep(2)
        if time.time() - start > timeout_seconds: raise Exception("Timed out waiting for ACTIVE.")

def delete_file(api_key: str, file_name: str):
    try: requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}", timeout=20)
    except: pass

# --- TRANSCRIPTION API ---

def generate_transcript(api_key: str, file_uri: str, mime_type: str, prompt: str) -> str:
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}]}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
    }
    
    for attempt in range(3):
        resp = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code != 200: return f"API ERROR {resp.status_code}: {resp.text}"
        try:
            body = resp.json()
            if body.get("promptFeedback", {}).get("blockReason"): return f"BLOCKED: {body['promptFeedback']['blockReason']}"
            candidates = body.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts"):
                text = candidates[0]["content"]["parts"][0].get("text", "")
                if text.strip(): return text
        except: pass
        time.sleep(2 * (attempt + 1))
    return "NO TRANSCRIPT (Empty Response)"

# --- WORKER LOGIC ---

def process_single_row(index, row, api_key, final_prompt, keep_remote):
    mobile = str(row.get("mobile_number", "Unknown"))
    result = {
        "index": index, "mobile_number": mobile,
        "recording_url": row.get("recording_url"), "transcript": row.get("transcript", ""),
        "status": row.get("status", "Pending"), "error": None
    }
    
    if row.get("processing_action") == "SKIP": return result
    
    audio_url = row.get("recording_url")
    if not audio_url or not isinstance(audio_url, str):
        result.update({"status": "❌ Failed", "error": "Invalid URL"})
        return result

    for attempt in range(MAX_WORKER_RETRIES):
        tmp_path = None
        file_info = None
        try:
            r = make_request_with_retry("GET", audio_url, stream=True)
            if r.status_code != 200: raise Exception(f"Download failed {r.status_code}")
            
            parsed = urlparse(audio_url)
            ext, mime = detect_extension_and_mime(parsed.path, r.headers.get("content-type"))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE): tmp.write(chunk)
                tmp_path = tmp.name
            
            clean_mob = "".join(ch for ch in mobile if ch.isalnum())
            u_name = f"rec_{clean_mob}_{int(time.time())}_{random.randint(100,999)}{ext}"
            
            up_url = initiate_upload(api_key, u_name, mime, os.path.getsize(tmp_path))
            file_info = upload_bytes(up_url, tmp_path, mime)
            wait_for_active(api_key, file_info["name"])
            
            transcript = generate_transcript(api_key, file_info["uri"], mime, final_prompt)
            result["transcript"] = transcript
            
            if any(x in transcript for x in ["API ERROR", "PARSE ERROR", "BLOCKED"]):
                if attempt < MAX_WORKER_RETRIES - 1: raise Exception(transcript)
                result["status"] = "❌ Error"
            elif "NO TRANSCRIPT" in transcript:
                if attempt < MAX_WORKER_RETRIES - 1: raise Exception("Empty")
                result["status"] = "❌ Empty"
            else:
                result["status"] = "✅ Success"
                return result

        except Exception as e:
            if attempt == MAX_WORKER_RETRIES - 1:
                result.update({"status": "❌ Failed", "transcript": f"SYSTEM ERROR: {e}", "error": str(e)})
            else:
                _sleep_with_jitter(WORKER_BACKOFF_BASE, attempt)
        finally:
            if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
            if file_info and not keep_remote: delete_file(api_key, file_info["name"])
    return result

# --- DISPLAY UTILS ---

def colorize_transcript_html(text):
    if not text or not text.strip(): return "<div class='other-speech'>No transcript</div>"
    html_out = ""
    for line in text.splitlines():
        clean = line.strip()
        if not clean: continue
        el = html.escape(clean)
        lc = clean.lower()
        if "speaker 1:" in lc: html_out += f"<div class='speaker1'>{el}</div>"
        elif "speaker 2:" in lc: html_out += f"<div class='speaker2'>{el}</div>"
        else: html_out += f"<div class='other-speech'>{el}</div>"
    return f"<div>{html_out}</div>"

# --- DEFAULT PROMPT ---
DEFAULT_PROMPT_TEMPLATE = """Transcribe this call in {language} exactly as spoken.

CRITICAL REQUIREMENTS:
1. Label every line as 'Speaker 1:' or 'Speaker 2:'.
2. Guess the speaker if unsure, never leave blank.
3. Timestamps [0ms-2500ms] at start of every line.
4. Hindi words in Hinglish (Latin script).
5. Exact transcription, no summarization.

Format: [timestamp] Speaker X: dialogue
Return ONLY the transcript.
"""

# --- MAIN APP ---

def main():
    if "processed_results" not in st.session_state: st.session_state.processed_results = []
    if "final_df" not in st.session_state: st.session_state.final_df = pd.DataFrame()

    # ---------------------------------------------------------
    # 1. SIDEBAR - SETTINGS ONLY (NO PROMPT HERE)
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Gemini API Key", type="password")
        st.divider()
        max_workers = st.slider("Concurrency", 1, 128, 4)
        keep_remote = st.checkbox("Keep Audio on Cloud", False)
        st.divider()
        theme_choice = st.radio("Theme", ["Light", "Dark"], horizontal=True)

    theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
    st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. MAIN SCREEN
    # ---------------------------------------------------------
    st.title("🎙️ Batch Transcriber")
    st.caption(f"Model: {MODEL_NAME}")

    st.write("### 📂 1. Upload Data")
    uploaded_files = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

    st.write("### 🧠 2. Transcription Logic")
    
    # --- PROMPT EDITOR MOVED HERE (MAIN SCREEN) ---
    col_settings, col_prompt = st.columns([1, 2])

    with col_settings:
        st.info("Language")
        language_mode = st.selectbox("Target Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
        lang_map = {"English (India)": "English", "Hindi": "Hindi", "Mixed (Hinglish)": "Hinglish"}

    with col_prompt:
        with st.expander("📝 Edit System Prompt", expanded=True):
            prompt_input = st.text_area("System Prompt", value=DEFAULT_PROMPT_TEMPLATE, height=250)

    st.markdown("---")
    
    progress_bar = st.empty()
    status_text = st.empty()
    result_placeholder = st.empty()
    
    start_button = st.button("🚀 Start Batch Processing", type="primary", use_container_width=True)

    # ---------------------------------------------------------
    # 3. PROCESSING
    # ---------------------------------------------------------
    if start_button:
        if not api_key: st.error("Missing API Key"); st.stop()
        if not uploaded_files: st.error("No files uploaded"); st.stop()

        all_dfs = []
        for f in uploaded_files:
            try: all_dfs.append(pd.read_excel(f))
            except: pass
        
        if not all_dfs: st.error("Could not read files"); st.stop()
        
        raw_df = pd.concat(all_dfs, ignore_index=True)
        if "recording_url" not in raw_df.columns: st.error("Missing 'recording_url' column"); st.stop()

        # Prep rows
        rows_to_process = []
        for i, row in raw_df.iterrows():
            r_copy = row.copy()
            url = r_copy.get("recording_url")
            if pd.notna(url) and str(url).strip():
                r_copy["processing_action"] = "TRANSCRIBE"
                r_copy["status"] = "Pending"
            else:
                r_copy["processing_action"] = "SKIP"
                r_copy["status"] = "Skipped"
            rows_to_process.append(r_copy)
        
        df_ready = pd.DataFrame(rows_to_process)
        final_prompt = prompt_input.replace("{language}", lang_map[language_mode])
        
        total = len(df_ready)
        processed_res = []
        st.session_state.processed_results = []
        
        status_text.info(f"Starting {total} tasks...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(process_single_row, idx, row, api_key, final_prompt, keep_remote): idx for idx, row in df_ready.iterrows()}
            done_count = 0
            for fut in as_completed(futures):
                res = fut.result()
                processed_res.append(res)
                done_count += 1
                progress_bar.progress(done_count / total)
                status_text.markdown(f"Processed **{done_count}/{total}**")
                
                # Live Preview
                prev_df = pd.DataFrame(processed_res[-5:])
                if not prev_df.empty:
                    result_placeholder.dataframe(prev_df[["mobile_number", "status", "transcript"]], hide_index=True)

        # Merge
        res_map = {r["index"]: r for r in processed_res}
        for i, row in df_ready.iterrows():
            if i in res_map:
                df_ready.at[i, "transcript"] = res_map[i]["transcript"]
                df_ready.at[i, "status"] = res_map[i]["status"]
                df_ready.at[i, "error"] = res_map[i]["error"]
        
        st.session_state.final_df = df_ready
        status_text.success("Done!")

    # ---------------------------------------------------------
    # 4. RESULTS
    # ---------------------------------------------------------
    final_df = st.session_state.final_df
    if not final_df.empty:
        st.markdown("## 🎛️ Transcript Browser")
        c1, c2, c3 = st.columns([3, 1, 1])
        q = c1.text_input("Search").lower()
        stat = c2.selectbox("Filter Status", ["All", "Success", "Failed"])
        
        v_df = final_df.copy()
        if stat == "Success": v_df = v_df[v_df["status"].str.contains("Success", case=False, na=False)]
        if stat == "Failed": v_df = v_df[v_df["status"].str.contains("Failed|Error|Empty", case=False, na=False)]
        if q:
            v_df = v_df[v_df["transcript"].fillna("").str.lower().str.contains(q) | v_df["recording_url"].astype(str).str.lower().str.contains(q)]
        
        st.write(f"Showing {len(v_df)} results")
        
        # Download
        buf = BytesIO()
        v_df.to_excel(buf, index=False)
        st.download_button("📥 Download Excel", buf.getvalue(), "transcripts.xlsx")

        # Cards
        # Paging for performance
        page_size = 10
        page = st.number_input("Page", 1, max(1, math.ceil(len(v_df)/page_size)), 1)
        start_idx = (page-1) * page_size
        
        for i, row in v_df.iloc[start_idx:start_idx+page_size].iterrows():
            lbl = f"{row.get('mobile_number')} — {row.get('status')}"
            with st.expander(lbl):
                st.markdown(f"<div class='meta-row'>URL: {row.get('recording_url')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='transcript-box'>{colorize_transcript_html(row.get('transcript'))}</div>", unsafe_allow_html=True)
                if row.get("error"): st.error(str(row.get("error")))
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
