# app.py — COMPLETE BATCH TRANSCRIBER (STRICT FAIL FAST MODE)
# -----------------------------------------------------------------------------
# FEATURES INCLUDED:
# 1. Multi-file Excel Upload (Merges multiple files).
# 2. Robust Gemini API Integration (Resumable Uploads).
# 3. FAIL FAST: No retries. If it errors, it fails immediately.
# 4. RESUME CAPABILITY: Automatically detects processed rows and lets you continue.
# 5. LIVE TABLE VIEW: Shows the full results table updating in real-time.
# 6. SAFE STOP: Use the browser "Stop" button; data is saved instantly.
# 7. LONG CONTEXT FIX: Supports 65k tokens and stitches multi-part responses.
# 8. NO DATA LOSS: Text file download added for transcripts > 32k chars.
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
import io
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
MODEL_NAME = "gemini-1.5-flash" # Updated to current stable flash model, change if needed

# Streaming download chunk size (8KB)
DOWNLOAD_CHUNK_SIZE = 8192

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
.call-card {
    border: 1px solid var(--border-color, #e6e6e6);
    border-radius: 10px; padding: 12px; margin-bottom: 12px;
    background: var(--card-bg, #fff); box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.transcript-box {
    max-height: 320px; overflow: auto; padding: 8px; border-radius: 6px;
    background: var(--transcript-bg, #fafafa); border: 1px solid var(--border-color, #eee);
    font-family: monospace; white-space: pre-wrap; 
}
.speaker1 { color: #1f77b4; font-weight: 600; display: block; margin-bottom: 4px; }
.speaker2 { color: #d62728; font-weight: 600; display: block; margin-bottom: 4px; }
.other-speech { color: #333; display: block; margin-bottom: 4px; }
.meta-row { font-size: 13px; color: var(--meta-color, #666); margin-bottom: 8px; }

.dark-theme {
    --card-bg: #0f1115; --transcript-bg: #0b0c0f; --border-color: #222428; --meta-color: #9aa0a6; color: #e6eef3;
}
.light-theme {
    --card-bg: #ffffff; --transcript-bg: #fafafa; --border-color: #e6e6e6; --meta-color: #666666; color: #111;
}
</style>
"""

st.set_page_config(page_title="Batch Transcriber", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)


# --- NETWORK UTILITIES (STRICT FAIL FAST) ---

def make_request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """
    STRICT FAIL FAST: Tries exactly once. 
    If it hits a 429, 500, or connection error, it raises an exception immediately.
    """
    try:
        # 120s timeout so we don't hang on dead connections
        resp = requests.request(method, url, timeout=120, **kwargs)
        
        # Raise error for 4xx or 5xx status codes immediately
        if resp.status_code >= 400:
            # You can log specific codes here if needed
            if resp.status_code == 429:
                logger.warning(f"Rate Limit Hit (429) for {url}")
            resp.raise_for_status()
            
        return resp
        
    except requests.exceptions.RequestException as e:
        # Log it and re-raise immediately so the worker marks it as "Failed"
        # logger.warning(f"Fail Fast triggered for {url}: {str(e)}") # Optional logging
        raise e


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
        "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size), "X-Goog-Upload-Header-Content-Type": mime_type,
    }
    payload = json.dumps({"file": {"display_name": display_name}})
    
    # FAIL FAST call
    resp = make_request_with_retry("POST", url, headers=headers, data=payload)
    
    if resp.status_code not in (200, 201):
        raise Exception(f"Init failed ({resp.status_code}): {resp.text}")
    return resp.headers.get("X-Goog-Upload-URL")

def upload_bytes(upload_url: str, file_path: str, mime_type: str) -> Dict[str, Any]:
    file_size = os.path.getsize(file_path)
    headers = {
        "Content-Type": mime_type or "application/octet-stream", "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"
    }
    
    # Using standard requests here for the binary upload, wrapped in try/except for Fail Fast
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(upload_url, headers=headers, data=f, timeout=300)
            
        if resp.status_code == 400:
            # Fallback for some resume scenarios, but typically we want to fail fast
            with open(file_path, "rb") as f:
                resp = requests.put(upload_url, headers=headers, data=f, timeout=300)
                
        if resp.status_code not in (200, 201):
            raise Exception(f"UPLOAD FAILED {resp.status_code}: {resp.text}")
            
        return resp.json().get("file", resp.json())
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Upload connection failed: {str(e)}")


# --- GOOGLE FILE STATUS POLLING ---

def wait_for_active(api_key: str, file_name: str, timeout_seconds: int = 60) -> bool:
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    start = time.time()
    while True:
        resp = make_request_with_retry("GET", url)
        
        j = resp.json()
        state = j.get("state")
        if state == "ACTIVE": return True
        if state == "FAILED": raise Exception(f"File processing failed: {j.get('processingError', j)}")
        
        # Check timeout
        if time.time() - start > timeout_seconds:
            raise Exception("Timed out waiting for file to become ACTIVE.")
            
        # Short sleep to prevent hammering, but kept minimal
        time.sleep(2)


def delete_file(api_key: str, file_name: str):
    try: requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}", timeout=10)
    except Exception: pass


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
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 65536}
    }

    # FAIL FAST call
    resp = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
    
    # We shouldn't reach here if status >= 400 because make_request_with_retry raises
    # But just in case of weird 200 responses with errors:
    try:
        body = resp.json()
    except ValueError:
        return "PARSE ERROR"

    prompt_feedback = body.get("promptFeedback", {})
    if prompt_feedback and prompt_feedback.get("blockReason"):
        return f"BLOCKED: {prompt_feedback.get('blockReason')}"

    candidates = body.get("candidates") or []
    if candidates:
        first = candidates[0]
        content = first.get("content", {})
        parts = content.get("parts", [])
        
        full_transcript_list = []
        if parts:
            for part in parts:
                txt = part.get("text") or part.get("content") or ""
                if txt:
                    full_transcript_list.append(txt)
        
        full_text = "".join(full_transcript_list)

        finish_reason = first.get("finishReason")
        if finish_reason == "MAX_TOKENS":
            full_text += "\n\n[WARNING: TRANSCRIPT TRUNCATED BY TOKEN LIMIT]"

        if full_text.strip(): 
            return full_text

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

    # --- SINGLE ATTEMPT (FAIL FAST) ---
    tmp_path = None
    file_info = None

    try:
        parsed = urlparse(audio_url)

        # 1. Download
        # logger.info("Downloading %s...", mobile)
        r = make_request_with_retry("GET", audio_url, stream=True)
        header_ct = r.headers.get("content-type", "")
        ext, mime_type = detect_extension_and_mime(parsed.path, header_ct)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk: tmp.write(chunk)
            tmp_path = tmp.name

        file_size = os.path.getsize(tmp_path)

        # 2. Upload
        # logger.info("Uploading %s...", mobile)
        cleaned_mobile = "".join(ch for ch in mobile if ch.isalnum())
        unique_name = f"rec_{cleaned_mobile}_{int(time.time())}_{random.randint(100,999)}{ext}"
        upload_url = initiate_upload(api_key, unique_name, mime_type, file_size)
        file_info = upload_bytes(upload_url, tmp_path, mime_type)

        # 3. Wait
        # logger.info("Waiting for %s...", mobile)
        wait_for_active(api_key, file_info["name"])

        # 4. Transcribe
        # logger.info("Transcribing %s...", mobile)
        transcript = generate_transcript(api_key, file_info["uri"], mime_type, final_prompt)
        result["transcript"] = transcript
        
        if "API ERROR" in transcript or "BLOCKED" in transcript:
            result["status"] = "❌ Error"
        elif "NO TRANSCRIPT" in transcript:
            result["status"] = "❌ Empty"
        else:
            result["status"] = "✅ Success"

    except Exception as e:
        # logger.exception("Failed %s: %s", mobile, str(e))
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


# --- RESULT MERGING ---

def merge_results_with_original(df_consolidated: pd.DataFrame, processed_results: list) -> pd.DataFrame:
    """
    Merges the worker results back into the consolidated DataFrame.
    """
    results_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
    cols_to_update = ["transcript", "status", "error"]
    df_base = df_consolidated.drop(columns=[c for c in cols_to_update if c in df_consolidated.columns])
    merged = df_base.merge(results_df[["index"] + cols_to_update], left_index=True, right_on="index", how="left")
    if "index" in merged.columns: merged = merged.drop(columns=["index"])
    return merged

def colorize_transcript_html(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return "<div class='other-speech'>No transcript</div>"
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

# --- PROMPT ---
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

# --- MAIN ENTRY POINT ---

def main():
    if "processed_results" not in st.session_state: st.session_state.processed_results = []
    if "final_df" not in st.session_state: st.session_state.final_df = pd.DataFrame()

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

    # Status Containers
    progress_bar = st.empty()
    status_text = st.empty()
    
    # LIVE TABLE PLACEHOLDER (New)
    live_table_placeholder = st.empty()

    # --- RESUME LOGIC SETUP ---
    # 1. Combine Files
    raw_df = pd.DataFrame()
    if uploaded_files:
        dfs = []
        for f in uploaded_files:
            try: dfs.append(pd.read_excel(f))
            except: pass
        if dfs: raw_df = pd.concat(dfs, ignore_index=True)

    # 2. Check for Previous Work
    previous_df = st.session_state.final_df
    
    start_triggered = False
    resume_triggered = False
    clear_triggered = False

    if not raw_df.empty:
        # If we have a previous session with data, check what is pending
        if not previous_df.empty and 'recording_url' in previous_df.columns:
            processed_urls = set(previous_df['recording_url'].astype(str))
            # Calculate pending based on URL
            pending_df_check = raw_df[~raw_df['recording_url'].astype(str).isin(processed_urls)]
            
            completed_count = len(raw_df) - len(pending_df_check)
            
            if completed_count > 0:
                st.info(f"Detected **{completed_count}** previously processed rows in memory.")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if len(pending_df_check) > 0:
                        resume_triggered = st.button(f"▶️ Resume ({len(pending_df_check)} remaining)", type="primary")
                    else:
                        st.success("✅ All files processed!")
                with col2:
                    clear_triggered = st.button("🗑️ Clear & Start Fresh")
            else:
                start_triggered = st.button("🚀 Start Batch Processing", type="primary")
        else:
            start_triggered = st.button("🚀 Start Batch Processing", type="primary")

    if clear_triggered:
        st.session_state.final_df = pd.DataFrame()
        st.session_state.processed_results = []
        st.rerun()

    # --- PROCESSING ---
    if start_triggered or resume_triggered:
        if not api_key: st.error("Please enter API Key."); st.stop()
        
        # Decide which dataset to use
        if resume_triggered:
            # Filter for only pending rows
            processed_urls = set(previous_df['recording_url'].astype(str))
            df_input = raw_df[~raw_df['recording_url'].astype(str).isin(processed_urls)]
            # We want to keep the old processed results in the list
            # st.session_state.processed_results is already populated
        else:
            # Fresh start
            df_input = raw_df
            st.session_state.processed_results = []
            st.session_state.final_df = pd.DataFrame()

        # Prepare
        df_ready = prepare_all_rows(df_input)
        final_prompt_to_use = prompt_input.replace("{language}", lang_map[language_mode])
        total_rows = len(df_ready)

        status_text.info(f"Processing {total_rows} items with {max_workers} threads (FAIL FAST MODE)...")
        progress_bar.progress(0.0)
        
        # Display the *existing* data immediately if resuming
        if not st.session_state.final_df.empty:
             live_table_placeholder.dataframe(st.session_state.final_df[["mobile_number", "status", "transcript"]], height=300, use_container_width=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_row, idx, row, api_key, final_prompt_to_use, keep_remote): idx
                for idx, row in df_ready.iterrows()
            }

            completed = 0
            for future in as_completed(futures):
                res = future.result()
                
                # Append to the GLOBAL results list
                st.session_state.processed_results.append(res)
                completed += 1
                
                # Update UI Progress
                progress_bar.progress(completed / total_rows)
                status_text.markdown(f"Processed **{completed}/{total_rows}** items.")
                
                # --- LIVE TABLE UPDATE ---
                # Create a temporary view DF from the results list
                live_view_df = pd.DataFrame(st.session_state.processed_results)
                # Sort it to keep it stable
                if not live_view_df.empty:
                    live_view_df = live_view_df.sort_values(by="index")
                    
                    # Update the placeholder
                    live_table_placeholder.dataframe(
                        live_view_df[["mobile_number", "status", "transcript", "error"]], 
                        height=400,
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Update the Session State Final DF (Background Save)
                if not raw_df.empty and not live_view_df.empty:
                       st.session_state.final_df = merge_results_with_original(prepare_all_rows(raw_df), st.session_state.processed_results)

        status_text.success("Batch Processing Complete!")
        st.rerun()

    # --- RESULTS VIEWER (ALWAYS VISIBLE) ---
    final_df = st.session_state.final_df

    if not final_df.empty:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("## 🎛️ Transcript Browser")

        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
        with col_a: search_q = st.text_input("Search", placeholder="Search transcript, phone, or URL...")
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
            mask = (view_df["transcript"].fillna("").str.lower().str.contains(q) | 
                    view_df["mobile_number"].astype(str).str.lower().str.contains(q))
            view_df = view_df[mask]

        if speaker_sel != "All":
            key = "speaker 1" if speaker_sel == "Speaker 1" else "speaker 2"
            mask = view_df["transcript"].fillna("").str.lower().str.contains(key)
            view_df = view_df[mask]

        total_items = len(view_df)
        st.markdown(f"**Showing {total_items} result(s)**")

        # --- DOWNLOADS ---
        col_down1, col_down2 = st.columns(2)
        
        # 1. Excel Download
        out_buf = BytesIO()
        view_df.to_excel(out_buf, index=False)
        with col_down1:
            st.download_button(
                "📥 Download Excel (Limit 32k chars/cell)", 
                data=out_buf.getvalue(), 
                file_name=f"transcripts_export_{int(time.time())}.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # 2. Text Download (New Fix)
        txt_buf = io.StringIO()
        for idx, row in view_df.iterrows():
            txt_buf.write(f"==================================================\n")
            txt_buf.write(f"MOBILE: {row.get('mobile_number', 'N/A')}\n")
            txt_buf.write(f"URL: {row.get('recording_url', 'N/A')}\n")
            txt_buf.write(f"STATUS: {row.get('status', 'N/A')}\n")
            txt_buf.write(f"==================================================\n\n")
            txt_buf.write(f"{row.get('transcript', '')}\n\n")
        
        with col_down2:
            st.download_button(
                "📄 Download Text File (Full Data - No Limit)", 
                data=txt_buf.getvalue(), 
                file_name=f"transcripts_full_{int(time.time())}.txt", 
                mime="text/plain",
                use_container_width=True
            )

        # Pagination & Rendering
        pages = max(1, math.ceil(total_items / per_page))
        page_idx = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
        start = (page_idx - 1) * per_page
        page_df = view_df.iloc[start:start+per_page]

        for idx, row in page_df.iterrows():
            mobile_display = row.get('mobile_number', 'Unknown')
            status_display = row.get('status', '')
            header = f"{mobile_display} — {status_display}"
            with st.expander(header, expanded=False):
                url_val = html.escape(str(row.get('recording_url', 'None')))
                st.markdown(f"<div class='meta-row'><b>URL:</b> {url_val}</div>", unsafe_allow_html=True)
                transcript_text = row.get("transcript", "")
                st.markdown(f"<div class='transcript-box'>{colorize_transcript_html(transcript_text)}</div>", unsafe_allow_html=True)
                if row.get("error"): st.error(f"Error: {row.get('error')}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
