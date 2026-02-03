# app.py — BATCH TRANSCRIBER (RESUME SUPPORT + FAIL FAST)
# -----------------------------------------------------------------------------
# FEATURES INCLUDED:
# 1. Multi-file Excel Upload (Merges multiple files).
# 2. Robust Gemini API Integration (Resumable Uploads).
# 3. FAIL FAST: No retries on error or empty response.
# 4. RESUME CAPABILITY: Detects processed rows and skips them.
# 5. REAL-TIME SAVE: Updates results instantly.
# 6. Persistent Results Viewer: Visible even after stopping.
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
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
MODEL_NAME = "gemini-3-flash-preview"

# Streaming download chunk size (8KB)
DOWNLOAD_CHUNK_SIZE = 8192

# Job-level retry configuration
MAX_WORKER_RETRIES = 1  
WORKER_BACKOFF_BASE = 0.5 

# Configure logging to console
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("transcriber")

# --- UI STYLING (CSS) ---
BASE_CSS = """
<style>
/* Card look for transcript entries */
.call-card {
    border: 1px solid var(--border-color, #e6e6e6);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
    background: var(--card-bg, #fff);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Transcript scroll area */
.transcript-box {
    max-height: 320px;
    overflow: auto;
    padding: 8px;
    border-radius: 6px;
    background: var(--transcript-bg, #fafafa);
    border: 1px solid var(--border-color, #eee);
    font-family: monospace;
    white-space: pre-wrap; 
}

.speaker1 { color: #1f77b4; font-weight: 600; display: block; margin-bottom: 4px; }
.speaker2 { color: #d62728; font-weight: 600; display: block; margin-bottom: 4px; }
.other-speech { color: #333; display: block; margin-bottom: 4px; }
.meta-row { font-size: 13px; color: var(--meta-color, #666); margin-bottom: 8px; }

.dark-theme {
    --card-bg: #0f1115;
    --transcript-bg: #0b0c0f;
    --border-color: #222428;
    --meta-color: #9aa0a6;
    color: #e6eef3;
}
.light-theme {
    --card-bg: #ffffff;
    --transcript-bg: #fafafa;
    --border-color: #e6e6e6;
    --meta-color: #666666;
    color: #111;
}
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
            resp = requests.request(method, url, timeout=120, **kwargs)
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning("Transient HTTP %s (attempt %d). Retrying...", resp.status_code, attempt + 1)
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning("RequestException on %s: %s (attempt %d)", method, str(e), attempt + 1)
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
    if last_exc: raise last_exc
    raise Exception("make_request_with_retry: retries exhausted")


# --- MIME TYPE & FILE EXTENSION HANDLING ---

COMMON_AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wave", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".webm": "audio/webm", ".flac": "audio/flac"
}

def detect_extension_and_mime(url_path: str, header_content_type: Optional[str]) -> (str, str):
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    if ext and ext in COMMON_AUDIO_MIME:
        return ext, COMMON_AUDIO_MIME[ext]
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        for k, v in COMMON_AUDIO_MIME.items():
            if v == ctype: return k, ctype
        guessed_ext = mimetypes.guess_extension(ctype)
        if guessed_ext: return guessed_ext.lower(), ctype
    return ".mp3", "audio/mpeg"


# --- GOOGLE UPLOAD PIPELINE ---

def initiate_upload(api_key: str, display_name: str, mime_type: str, file_size: int) -> str:
    url = f"{UPLOAD_URL}?uploadType=resumable&key={api_key}"
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
    }
    payload = json.dumps({"file": {"display_name": display_name}})
    resp = make_request_with_retry("POST", url, headers=headers, data=payload)
    if resp.status_code not in (200, 201):
        raise Exception(f"Init failed ({resp.status_code}): {resp.text}")
    return resp.headers.get("X-Goog-Upload-URL")

def upload_bytes(upload_url: str, file_path: str, mime_type: str) -> Dict[str, Any]:
    file_size = os.path.getsize(file_path)
    headers = {
        "Content-Type": mime_type or "application/octet-stream",
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=300)
    if resp.status_code == 400:
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, headers=headers, data=f, timeout=300)
    if resp.status_code not in (200, 201):
        raise Exception(f"UPLOAD FAILED {resp.status_code}: {resp.text}")
    return resp.json().get("file", resp.json())


# --- GOOGLE FILE STATUS POLLING ---

def wait_for_active(api_key: str, file_name: str, timeout_seconds: int = 60) -> bool:
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    start = time.time()
    while True:
        resp = make_request_with_retry("GET", url)
        if resp.status_code != 200:
            time.sleep(2)
        else:
            j = resp.json()
            state = j.get("state")
            if state == "ACTIVE": return True
            if state == "FAILED": raise Exception(f"File processing failed: {j.get('processingError', j)}")
            time.sleep(2)
        if time.time() - start > timeout_seconds:
            raise Exception("Timed out waiting for file to become ACTIVE.")

def delete_file(api_key: str, file_name: str):
    try:
        requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}", timeout=20)
    except Exception:
        pass


# --- TRANSCRIPTION API CALL ---

def generate_transcript(api_key: str, file_uri: str, mime_type: str, prompt: str) -> str:
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}]}],
        "safetySettings": safety_settings,
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
    }
    
    # NO RETRIES on content (Fail Fast)
    resp = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
    
    if resp.status_code != 200:
        return f"API ERROR {resp.status_code}: {resp.text}"

    try:
        body = resp.json()
    except ValueError:
        return "PARSE ERROR: Non-JSON response."

    prompt_feedback = body.get("promptFeedback", {})
    if prompt_feedback and prompt_feedback.get("blockReason"):
        return f"BLOCKED: {prompt_feedback.get('blockReason')}"

    candidates = body.get("candidates") or []
    if candidates:
        first = candidates[0]
        content = first.get("content", {})
        parts = content.get("parts", [])
        if parts:
            text = parts[0].get("text") or parts[0].get("content") or ""
            if text.strip():
                return text

    return "NO TRANSCRIPT (Empty Response)"


# --- DATA PREPARATION LOGIC ---

def prepare_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    final_rows = []
    for index, row in df.iterrows():
        row_data = row.copy()
        url_val = row_data.get('recording_url')
        if pd.notna(url_val) and str(url_val).strip() != "":
            row_data['processing_action'] = 'TRANSCRIBE'
            row_data['status'] = 'Pending'
        else:
            row_data['processing_action'] = 'SKIP'
            row_data['transcript'] = "⚠️ Skipped: No recording URL provided."
            row_data['status'] = '⚠️ Skipped'
            row_data['error'] = 'Missing recording_url'
        final_rows.append(row_data)
    
    return pd.DataFrame(final_rows).reset_index(drop=True)


# --- WORKER FUNCTION ---

def process_single_row(index: int, row: pd.Series, api_key: str, final_prompt: str, keep_remote: bool = False) -> Dict[str, Any]:
    mobile = str(row.get("mobile_number", "Unknown"))
    result = {
        "index": index, "mobile_number": mobile, "recording_url": row.get("recording_url"),
        "transcript": row.get("transcript", ""), "status": row.get("status", "Pending"), "error": row.get("error", None),
    }

    if row.get("processing_action") == "SKIP":
        return result

    audio_url = row.get("recording_url")
    if not audio_url or not isinstance(audio_url, str):
        result.update({"status": "❌ Failed", "error": "Invalid URL"})
        return result

    # SINGLE ATTEMPT (No retries loop)
    tmp_path = None
    file_info = None
    try:
        parsed = urlparse(audio_url)
        
        # Download
        logger.info("Downloading %s...", mobile)
        r = make_request_with_retry("GET", audio_url, stream=True)
        if r.status_code != 200: raise Exception(f"Download failed ({r.status_code})")
        header_ct = r.headers.get("content-type", "")
        ext, mime_type = detect_extension_and_mime(parsed.path, header_ct)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk: tmp.write(chunk)
            tmp_path = tmp.name

        # Upload
        file_size = os.path.getsize(tmp_path)
        logger.info("Uploading %s...", mobile)
        cleaned_mobile = "".join(ch for ch in mobile if ch.isalnum())
        unique_name = f"rec_{cleaned_mobile}_{int(time.time())}_{random.randint(100,999)}{ext}"
        upload_url = initiate_upload(api_key, unique_name, mime_type, file_size)
        file_info = upload_bytes(upload_url, tmp_path, mime_type)

        # Wait
        logger.info("Waiting for %s...", mobile)
        wait_for_active(api_key, file_info["name"])

        # Transcribe
        logger.info("Transcribing %s...", mobile)
        transcript = generate_transcript(api_key, file_info["uri"], mime_type, final_prompt)
        result["transcript"] = transcript
        
        if "API ERROR" in transcript or "BLOCKED" in transcript:
            result["status"] = "❌ Error"
        elif "NO TRANSCRIPT" in transcript:
            result["status"] = "❌ Empty"
        else:
            result["status"] = "✅ Success"

    except Exception as e:
        logger.exception("Failed %s: %s", mobile, str(e))
        result["transcript"] = f"SYSTEM ERROR: {str(e)}"
        result["status"] = "❌ Failed"
        result["error"] = str(e)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        if file_info and isinstance(file_info, dict) and file_info.get("name") and not keep_remote:
            delete_file(api_key, file_info["name"])

    return result


# --- UI UTILS ---

def colorize_transcript_html(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "<div class='other-speech'>No transcript</div>"
    lines = text.splitlines()
    html_output = ""
    for line in lines:
        clean = line.strip()
        if not clean: continue
        escaped_line = html.escape(clean)
        lc = clean.lower()
        if "speaker 1:" in lc: html_output += f"<div class='speaker1'>{escaped_line}</div>"
        elif "speaker 2:" in lc: html_output += f"<div class='speaker2'>{escaped_line}</div>"
        else: html_output += f"<div class='other-speech'>{escaped_line}</div>"
    return f"<div>{html_output}</div>"

DEFAULT_PROMPT_TEMPLATE = """Transcribe this call in {language} exactly as spoken.

CRITICAL REQUIREMENTS — FOLLOW STRICTLY:
1. EVERY line MUST start with exactly one of these labels:
   - Speaker 1:
   - Speaker 2:
2. NEVER merge dialogue from two speakers in one line.
3. If you are unsure who is speaking, GUESS — but DO NOT leave the speaker label blank.
4. If the call sounds like a single-person monologue, STILL label every line as:
   Speaker 1: <text>
5. Do NOT summarize or improve the language. Write EXACTLY what was said.

TIMESTAMP RULES:
- Add timestamps at the start of EVERY line.
- Format MUST be: [0ms-2500ms]
- Use raw milliseconds only.

LANGUAGE RULES:
- ALL Hindi words must be written in Hinglish (Latin script).
- NO Devanagari characters anywhere.

Return ONLY the transcript. No explanation.
"""

# --- MAIN APP ---

def main():
    # 1. Initialize Session State
    if "final_df" not in st.session_state:
        st.session_state.final_df = pd.DataFrame()

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Gemini API Key", type="password")
        max_workers = st.slider("Concurrency", 1, 128, 4)
        keep_remote = st.checkbox("Keep audio on Google", False)
        st.divider()
        language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
        lang_map = { "English (India)": "English (Indian accent)", "Hindi": "Hindi (Devanagari)", "Mixed (Hinglish)": "Mixed English and Hindi" }
        with st.expander("⚙️ Advanced: Edit System Prompt"):
            prompt_input = st.text_area("System Prompt", value=DEFAULT_PROMPT_TEMPLATE, height=300)
        st.divider()
        theme_choice = st.radio("Theme", ["Light", "Dark"], 0, horizontal=True)
    
    theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
    st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)

    st.write("### 📂 Upload Call Data")
    uploaded_files = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

    # Placeholders for progress
    progress_bar = st.empty()
    status_text = st.empty()
    result_placeholder = st.empty()

    # --- LOGIC TO HANDLE "RESUME" vs "START" ---
    
    # 1. Combine uploads first (needed for Resume check)
    raw_df = pd.DataFrame()
    if uploaded_files:
        dfs = []
        for file in uploaded_files:
            try: dfs.append(pd.read_excel(file))
            except: pass
        if dfs:
            raw_df = pd.concat(dfs, ignore_index=True)

    # 2. Check Session State for previous work
    previous_df = st.session_state.final_df
    rows_to_process = pd.DataFrame()
    
    col1, col2, col3 = st.columns([1,1,2])
    
    start_triggered = False
    resume_triggered = False
    clear_triggered = False

    if not raw_df.empty:
        # Check intersection based on recording_url
        if not previous_df.empty and 'recording_url' in previous_df.columns:
            processed_urls = set(previous_df['recording_url'].astype(str))
            # Calculate pending
            pending_df = raw_df[~raw_df['recording_url'].astype(str).isin(processed_urls)]
            
            completed_count = len(raw_df) - len(pending_df)
            
            if completed_count > 0:
                st.info(f"Detected **{completed_count}** previously processed rows in memory.")
                with col1:
                    resume_triggered = st.button(f"▶️ Resume ({len(pending_df)} remaining)", type="primary")
                with col2:
                    clear_triggered = st.button("🗑️ Clear & Start Fresh")
            else:
                with col1:
                    start_triggered = st.button("🚀 Start Processing", type="primary")
        else:
            with col1:
                start_triggered = st.button("🚀 Start Processing", type="primary")

    if clear_triggered:
        st.session_state.final_df = pd.DataFrame()
        st.rerun()

    # --- PROCESSING BLOCK ---
    if start_triggered or resume_triggered:
        if not api_key:
            st.error("Please enter API Key.")
            st.stop()

        # Determine dataset
        if resume_triggered:
            processed_urls = set(previous_df['recording_url'].astype(str))
            df_input = raw_df[~raw_df['recording_url'].astype(str).isin(processed_urls)]
            st.warning("Resuming... new results will be appended to existing data.")
        else:
            df_input = raw_df
            st.session_state.final_df = pd.DataFrame() # Clear old if fresh start

        # Prepare
        df_ready = prepare_all_rows(df_input)
        
        final_prompt_to_use = prompt_input.replace("{language}", lang_map[language_mode])
        total_rows = len(df_ready)
        
        if total_rows == 0:
            st.success("All rows are already processed!")
        else:
            status_text.info(f"Processing {total_rows} items with {max_workers} threads...")
            progress_bar.progress(0.0)

            results_buffer = []
            
            # Use ThreadPool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_single_row, idx, row, api_key, final_prompt_to_use, keep_remote): idx
                    for idx, row in df_ready.iterrows()
                }

                completed = 0
                for future in as_completed(futures):
                    res = future.result()
                    results_buffer.append(res)
                    completed += 1
                    
                    # Update Progress
                    progress_bar.progress(completed / total_rows)
                    status_text.markdown(f"Processed **{completed}/{total_rows}** items.")
                    
                    # Live Preview
                    preview_df = pd.DataFrame(results_buffer[-5:])
                    if not preview_df.empty:
                        result_placeholder.dataframe(
                            preview_df[["mobile_number", "status", "transcript"]], 
                            width=800, hide_index=True
                        )

                    # REAL-TIME SAVE TO GLOBAL STATE
                    # We merge the NEW results buffer into a DataFrame
                    new_results_df = pd.DataFrame(results_buffer)
                    # We merge this with the original 'df_ready' to get the full columns (url, etc)
                    # Note: We can't easily merge back to 'previous_df' inside the loop without getting slow.
                    # Strategy: Store the *accumulated* list in session state temporarily?
                    # Better: Just Append.
                    
            # FINAL MERGE after loop (or sub-merge)
            # Create the DataFrame for just this run
            run_results_df = pd.DataFrame(sorted(results_buffer, key=lambda r: r["index"]))
            if not run_results_df.empty:
                cols_to_update = ["transcript", "status", "error"]
                df_base = df_ready.drop(columns=[c for c in cols_to_update if c in df_ready.columns])
                merged_run_df = df_base.merge(run_results_df[["index"] + cols_to_update], left_index=True, right_on="index", how="left")
                if "index" in merged_run_df.columns: merged_run_df = merged_run_df.drop(columns=["index"])
                
                # Append to Session State
                st.session_state.final_df = pd.concat([st.session_state.final_df, merged_run_df], ignore_index=True)

            status_text.success("Batch Complete!")
            time.sleep(1)
            st.rerun() # Refresh to show full table

    # --- RESULTS VIEWER (ALWAYS VISIBLE IF DATA EXISTS) ---
    final_df = st.session_state.final_df

    if not final_df.empty:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"## 🎛️ Transcript Browser ({len(final_df)} items)")

        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
        with col_a: search_q = st.text_input("Search", placeholder="Search text...")
        with col_b: status_sel = st.selectbox("Status", ["All", "Success", "Failed", "Skipped"])
        with col_c: speaker_sel = st.selectbox("Speaker", ["All", "Speaker 1", "Speaker 2"])
        with col_d: per_page = st.selectbox("Per page", [5, 10, 20, 50], index=1)

        view_df = final_df.copy()
        if status_sel != "All":
            if status_sel == "Success": view_df = view_df[view_df["status"].str.contains("Success", case=False, na=False)]
            elif status_sel == "Failed": view_df = view_df[view_df["status"].str.contains("Failed|Error|Empty", case=False, na=False)]
            elif status_sel == "Skipped": view_df = view_df[view_df["status"].str.contains("Skipped", case=False, na=False)]
        
        if search_q.strip():
            q = search_q.lower()
            mask = (
                view_df["transcript"].fillna("").str.lower().str.contains(q) |
                view_df["mobile_number"].astype(str).str.lower().str.contains(q)
            )
            view_df = view_df[mask]

        if speaker_sel != "All":
            key = "speaker 1" if speaker_sel == "Speaker 1" else "speaker 2"
            mask = view_df["transcript"].fillna("").str.lower().str.contains(key)
            view_df = view_df[mask]

        # Pagination & Download
        total_items = len(view_df)
        st.caption(f"Showing {total_items} result(s)")
        
        out_buf = BytesIO()
        view_df.to_excel(out_buf, index=False)
        st.download_button("📥 Download Excel", data=out_buf.getvalue(), file_name="transcripts.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        pages = max(1, math.ceil(total_items / per_page))
        page_idx = st.number_input("Page", 1, pages, 1)
        start = (page_idx - 1) * per_page
        page_df = view_df.iloc[start:start+per_page]

        for idx, row in page_df.iterrows():
            with st.expander(f"{row.get('mobile_number')} — {row.get('status')}", expanded=False):
                st.markdown(f"<div class='meta-row'><b>URL:</b> {row.get('recording_url')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='transcript-box'>{colorize_transcript_html(row.get('transcript'))}</div>", unsafe_allow_html=True)
                if row.get("error"): st.error(row.get("error"))

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
