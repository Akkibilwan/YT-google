# --- START OF FILE app.py ---

import os
import time
import json
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
# Ensure this import is present if using find_peaks
from scipy.signal import find_peaks
import requests
import sqlite3
import hashlib
import re
from datetime import datetime, timedelta, timezone
# Handle potential OpenAI import errors
try:
    import openai
    from openai import APIError as OpenAI_APIError # Use alias for clarity
except ImportError:
    st.error("OpenAI library not installed. Please add 'openai' to requirements.txt")
    openai = None
    OpenAI_APIError = None # Define as None if library missing

import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict
import atexit
import subprocess
import tempfile
import shutil
import pandas as pd
import isodate

# Selenium & related imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    # Import specific exceptions
    from selenium.common.exceptions import WebDriverException, NoSuchElementException, TimeoutException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    st.warning("Selenium libraries not installed. Retention Analysis feature will be disabled. Add 'selenium' and 'webdriver-manager' to requirements.txt")
    SELENIUM_AVAILABLE = False

import imageio_ffmpeg
# YouTube specific libraries
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
    # Correct import for yt-dlp classes
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError as YTDLP_DownloadError
    YOUTUBE_LIBS_AVAILABLE = True
except ImportError:
    st.error("Required YouTube libraries ('youtube-transcript-api', 'yt-dlp') not found. Please add them to requirements.txt")
    YOUTUBE_LIBS_AVAILABLE = False
    # Define dummy classes if import failed to avoid NameErrors later if code tries to use them conditionally
    class YoutubeDL: pass
    class YTDLP_DownloadError(Exception): pass


# =============================================================================
# 1. Logging Setup
# =============================================================================
def setup_logger():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        try: os.makedirs(log_dir)
        except OSError as e: print(f"Error creating log directory {log_dir}: {e}"); return logging.getLogger() # Return basic logger on error
    log_file = os.path.join(log_dir, "youtube_analyzer.log")
    # Reduced log size, more backups
    file_handler = RotatingFileHandler(log_file, maxBytes=256*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s")) # Concise console output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set to INFO for better debugging during development
    # logger.setLevel(logging.WARNING) # Switch back to WARNING for production
    # Prevent adding handlers multiple times
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    # Suppress noisy logs from libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("webdriver_manager").setLevel(logging.WARNING)
    return logger

logger = setup_logger()
logger.info("Logger initialized.")

# =============================================================================
# 2. API Keys & Config
# =============================================================================
# YouTube API Key
YOUTUBE_API_KEY = None
try:
    YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", {}).get("key")
    if not YOUTUBE_API_KEY: # Try fallback structure
         YOUTUBE_API_KEY = st.secrets.get("YT_API_KEY")
    if YOUTUBE_API_KEY:
        logger.info("YouTube API key loaded.")
    else:
        # Avoid showing error repeatedly in UI, log instead
        logger.critical("YouTube API key not found in Streamlit secrets (Checked YOUTUBE_API_KEY.key and YT_API_KEY). App might not function.")
        # Optionally show error once on startup?
        # if 'api_key_error_shown' not in st.session_state:
        #      st.error("❗ YouTube API key not found in Streamlit secrets.")
        #      st.session_state.api_key_error_shown = True
except Exception as e:
    st.error(f"🚨 Error accessing YouTube API key from secrets: {e}")
    logger.error(f"Error loading YouTube API key: {e}", exc_info=True)

# OpenAI API Key
OPENAI_API_KEY = None
if openai: # Only proceed if library was imported
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", {}).get("key")
        if OPENAI_API_KEY:
            # Test key validity early? Optional.
            # try: openai.models.list() # Simple API call to test auth
            logger.info("OpenAI API key loaded.")
            # except openai.AuthenticationError: logger.error("OpenAI API key is invalid."); OPENAI_API_KEY = None; st.error("Invalid OpenAI Key.")
        else:
            logger.warning("OpenAI API key not found in Streamlit secrets. AI features disabled.")
            # Optionally show warning once?
            # if 'openai_key_warning_shown' not in st.session_state:
            #      st.warning("⚠️ OpenAI API key missing. AI features disabled.")
            #      st.session_state.openai_key_warning_shown = True
    except Exception as e:
        st.error(f"🚨 Error accessing OpenAI API key from secrets: {e}")
        logger.error(f"Error loading OpenAI API key: {e}", exc_info=True)
else:
    logger.warning("OpenAI library not installed. AI features disabled.")

def get_youtube_api_key():
    if not YOUTUBE_API_KEY:
        logger.error("get_youtube_api_key called but key is missing.")
        raise ValueError("YouTube API key is not configured.")
    return YOUTUBE_API_KEY

# Other Config
DB_PATH = "cache.db"
FOLDERS_FILE = "channel_folders.json"
COOKIE_FILE = "youtube_cookies.json" # For Selenium retention
CHROMIUM_PATH_STANDARD = "/usr/bin/chromium" # Common path in Linux containers

# =============================================================================
# 3. SQLite DB Setup (Caching)
# =============================================================================
@st.cache_resource # Cache DB initialization result (the path or None)
def init_db(db_path=DB_PATH):
    """Initializes the SQLite database and returns the path if successful."""
    try:
        with sqlite3.connect(db_path, timeout=15, check_same_thread=False) as conn:
            conn.execute("PRAGMA journal_mode=WAL;") # Write-Ahead Logging for better concurrency
            conn.execute("PRAGMA busy_timeout = 5000;") # Wait up to 5s if locked
            conn.execute("""
            CREATE TABLE IF NOT EXISTS youtube_cache (
                cache_key TEXT PRIMARY KEY,
                json_data TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON youtube_cache(cache_key);")
        logger.info(f"Database initialized/verified at {db_path}")
        return db_path # Return path for functions to use
    except sqlite3.Error as e:
        logger.critical(f"FATAL: Failed to initialize database {db_path}: {e}", exc_info=True)
        st.error(f"Database initialization failed: {e}. Caching will be disabled.")
        return None # Indicate failure

DB_CONN_PATH = init_db() # Initialize DB on app start

def get_cached_result(cache_key, ttl=600, db_path=DB_CONN_PATH):
    if not db_path: return None
    now = time.time()
    try:
        with sqlite3.connect(db_path, timeout=10) as conn: # Use timeout for connection
            conn.execute("PRAGMA busy_timeout = 3000;") # Wait up to 3s if locked during query
            cursor = conn.execute("SELECT json_data, timestamp FROM youtube_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
        if row:
            json_data, cached_time = row
            if (now - cached_time) < ttl:
                logger.debug(f"Cache hit: {cache_key[-8:]}")
                return json.loads(json_data)
            else:
                logger.debug(f"Cache expired: {cache_key[-8:]}")
                delete_cache_key(cache_key, db_path) # Explicitly delete expired
                return None
    except sqlite3.OperationalError as e:
         if "locked" in str(e).lower(): logger.warning(f"DB locked getting cache: {cache_key[-8:]}")
         else: logger.error(f"DB error get cache: {cache_key[-8:]}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in cache: {cache_key[-8:]}: {e}. Deleting.")
        delete_cache_key(cache_key, db_path)
    except Exception as e: logger.error(f"Get cache error: {cache_key[-8:]}: {e}", exc_info=True)
    return None

def set_cached_result(cache_key, data_obj, db_path=DB_CONN_PATH):
    if not db_path: return
    now = time.time()
    try: json_str = json.dumps(data_obj, default=str)
    except TypeError as e: logger.error(f"Cache data not JSON serializable: {cache_key[-8:]}: {e}"); return

    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute("PRAGMA busy_timeout = 3000;")
            conn.execute("INSERT OR REPLACE INTO youtube_cache (cache_key, json_data, timestamp) VALUES (?, ?, ?)",
                         (cache_key, json_str, now))
        logger.debug(f"Cache set: {cache_key[-8:]}")
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower(): logger.warning(f"DB locked setting cache: {cache_key[-8:]}")
        else: logger.error(f"DB error set cache: {cache_key[-8:]}: {e}")
    except Exception as e: logger.error(f"Set cache error: {cache_key[-8:]}: {e}", exc_info=True)

def delete_cache_key(cache_key, db_path=DB_CONN_PATH):
    if not db_path: return
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute("PRAGMA busy_timeout = 3000;")
            conn.execute("DELETE FROM youtube_cache WHERE cache_key = ?", (cache_key,))
        logger.debug(f"Cache deleted: {cache_key[-8:]}")
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower(): logger.warning(f"DB locked deleting cache: {cache_key[-8:]}")
        else: logger.error(f"DB error delete cache: {cache_key[-8:]}: {e}")
    except Exception as e: logger.error(f"Delete cache error: {cache_key[-8:]}: {e}", exc_info=True)

def clear_all_cache(db_path=DB_CONN_PATH):
    if not db_path: st.error("Database error, cannot clear cache."); return False
    try:
        with sqlite3.connect(db_path, timeout=15) as conn:
            conn.execute("DELETE FROM youtube_cache"); conn.execute("VACUUM")
        logger.info("Cleared all cache table entries.")
        keys_to_clear = [k for k in st.session_state if k.startswith(('search_', '_search_', 'transcript_', 'comments_', 'analysis_', 'retention_', 'summary_'))]
        for key in keys_to_clear: del st.session_state[key]
        logger.info(f"Cleared {len(keys_to_clear)} session state entries.")
        # Clear function caches if needed
        load_channel_folders.clear()
        # get_channel_id.clear() # If caching was enabled
        return True
    except sqlite3.Error as e: logger.error(f"Failed clear cache table: {e}"); st.error(f"Cache clear failed: {e}"); return False

# =============================================================================
# 4. Utility Helpers
# =============================================================================
def format_date(date_string):
    if not date_string or not isinstance(date_string, str): return "Unknown"
    try:
        # Improved parsing robustness
        ts = pd.to_datetime(date_string, errors='coerce', utc=True)
        return ts.strftime("%d-%m-%y") if pd.notna(ts) else "Invalid Date"
    except Exception: return "Parse Error"

def format_number(num):
    try:
        n = int(num)
        if abs(n) >= 1_000_000: return f"{n/1_000_000:.1f}M"
        elif abs(n) >= 1_000: return f"{n/1_000:.1f}K"
        return str(n)
    except (ValueError, TypeError): return str(num)
    except Exception: return "Err"

def build_cache_key(*args):
    try:
        key_parts = []
        for a in args:
            if isinstance(a, (list, tuple)):
                 try: key_parts.append(repr(sorted(a))) # Sort lists/tuples
                 except TypeError: key_parts.append(repr(a))
            else: key_parts.append(repr(a)) # Use repr for stability
        raw_str = "||".join(key_parts) # Use a less common separator
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    except Exception as e:
         logger.error(f"Cache key build failed: {args} - {e}", exc_info=True)
         raise ValueError("Cache key build failed") from e

def parse_iso8601_duration(duration_str):
    if not duration_str or duration_str == 'P0D': return 0
    try: return int(isodate.parse_duration(duration_str).total_seconds())
    except Exception: logger.debug(f"Duration parse failed: {duration_str}"); return 0

# =============================================================================
# 5. Channel Folders
# =============================================================================
@st.cache_data # Cache folder data loading
def load_channel_folders(file_path=FOLDERS_FILE):
    """Loads channel folders from JSON file, creates default if not found/invalid."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                folders = json.load(f)
                if isinstance(folders, dict):
                     logger.info(f"Loaded {len(folders)} folders from {file_path}")
                     return folders
                else: logger.warning(f"{file_path} invalid content, creating default.")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading {file_path}: {e}. Creating default.")

    logger.info("Creating default channel folders.")
    default_folders = {
        "Example Finance Channels": [
             {"channel_name": "Yahoo Finance", "channel_id": "UCEAZeUIeJs0IjQiqTCdVSIg"},
             {"channel_name": "Bloomberg Television", "channel_id": "UCIALMKvObZNtJ6AmdCLP7Lg"}
        ],
        "My Favorite Channels": []
    }
    save_channel_folders(default_folders, file_path) # Save defaults
    return default_folders

def save_channel_folders(folders, file_path=FOLDERS_FILE):
    """Saves the channel folders dictionary to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(folders, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(folders)} folders to {file_path}")
        load_channel_folders.clear() # Clear cache after saving
        return True
    except Exception as e:
        logger.error(f"Error saving folders to {file_path}: {e}", exc_info=True)
        st.error(f"Error saving folder data: {e}")
        return False

# @st.cache_data(ttl=3600) # Optionally cache channel ID lookups
def get_channel_id(channel_name_or_url):
    """Resolves various YouTube channel inputs to a channel ID using Search API."""
    identifier = channel_name_or_url.strip()
    if not identifier: return None
    if identifier.startswith("UC") and len(identifier) == 24 and re.match(r'^UC[A-Za-z0-9_\-]{22}$', identifier): return identifier
    match_id = re.search(r'youtube\.com/channel/(UC[A-Za-z0-9_\-]{22})', identifier, re.IGNORECASE)
    if match_id: return match_id.group(1)
    search_query = identifier # Search API handles '@' handles well now

    try:
        key = get_youtube_api_key()
        params = {'key': key, 'part': 'snippet', 'type': 'channel', 'q': search_query, 'maxResults': 1}
        api_url = "https://www.googleapis.com/youtube/v3/search"
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("items"):
            item = data["items"][0]; channel_id = item.get('id', {}).get('channelId')
            found_title = item.get('snippet', {}).get('title', '')
            if channel_id: logger.info(f"Resolved '{identifier}' to '{found_title}' ({channel_id})"); return channel_id
            else: logger.warning(f"Search ok but item lacked channelId: {identifier}")
        else: logger.warning(f"API search returned no items for: {identifier}")
    except ValueError as e: st.error(f"API key error: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error resolving '{identifier}': {e}", exc_info=True)
        status = getattr(e.response, 'status_code', None)
        msg = f"API Error ({status})" if status else f"Network Error ({e})"
        if status == 403: msg += " Check Key/Quota."
        st.error(msg)
    except Exception as e: logger.error(f"Unexpected error resolving '{identifier}': {e}", exc_info=True); st.error(f"Error: {e}")
    return None

# --- Folder Management UI ---
def show_channel_folder_manager():
    # Simplified UI structure
    st.write("#### Manage Channel Folders")
    folders = load_channel_folders(); folder_keys = list(folders.keys())
    action = st.radio("Action:", ["Add", "Remove", "Create", "Delete"], key="fldr_action_radio", horizontal=True)

    if action == "Create":
        with st.popover("Create New Folder"):
             with st.form("create_form"):
                  new_name = st.text_input("Folder Name:").strip()
                  if st.form_submit_button("Create"):
                      if not new_name: st.error("Name empty.")
                      elif new_name in folders: st.error("Name exists.")
                      else: folders[new_name] = []; save_channel_folders(folders); st.success("Created!"); st.rerun()
    elif action == "Delete":
         if not folder_keys: st.info("No folders to delete."); return
         f_del = st.selectbox("Delete Folder:", folder_keys, key="fldr_del_sel")
         if st.button(f"Delete '{f_del}'", type="primary"):
              if f_del in folders: del folders[f_del]; save_channel_folders(folders); st.success("Deleted."); st.rerun()
              else: st.error("Not found.")
    elif action == "Add":
         if not folder_keys: st.info("Create folder first."); return
         f_add = st.selectbox("Add to Folder:", folder_keys, key="fldr_add_sel")
         with st.popover("Add Channels"):
              with st.form("add_form"):
                   ch_text = st.text_area("Channel names/URLs/IDs (one per line):", height=100)
                   if st.form_submit_button("Resolve & Add"):
                       lines = ch_text.strip().split("\n"); added=0; current_ids={c['channel_id'] for c in folders.get(f_add,[])}
                       with st.spinner("Resolving..."):
                           for line in filter(None, map(str.strip, lines)):
                               cid = get_channel_id(line)
                               if cid and cid not in current_ids:
                                   cname = line;
                                   try: d=requests.get(f"https://...channels?part=snippet&id={cid}&key={get_youtube_api_key()}", timeout=5).json(); cname=d['items'][0]['snippet']['title'] if d.get('items') else line
                                   except Exception: pass
                                   folders[f_add].append({"channel_name": cname, "channel_id": cid}); current_ids.add(cid); added+=1
                               elif cid: st.warning(f"'{line}' already in folder.")
                               else: st.warning(f"Cannot resolve '{line}'.")
                       if added > 0: save_channel_folders(folders); st.success(f"Added {added}."); st.rerun()
                       else: st.info("No new channels added.")
    elif action == "Remove":
         if not folder_keys: st.info("No folders."); return
         f_rem = st.selectbox("Remove from Folder:", folder_keys, key="fldr_rem_sel")
         if f_rem and folders.get(f_rem):
              chans = folders[f_rem];
              if not chans: st.info("Folder empty."); return
              opts = {f"{c['channel_name']} ({c['channel_id']})": c['channel_id'] for c in chans}
              choice = st.selectbox("Remove Channel:", list(opts.keys()), key="fldr_rem_ch_sel")
              if st.button("Remove Selected", type="secondary"):
                   cid = opts.get(choice)
                   if cid: folders[f_rem]=[c for c in chans if c['channel_id']!=cid]; save_channel_folders(folders); st.success("Removed."); st.rerun()
         elif f_rem: st.info("Folder empty.")

# =============================================================================
# 6. Transcript & Fallback (Corrected `finally` block)
# =============================================================================
def get_transcript(video_id):
    """Fetches YouTube transcript if library available."""
    if not YOUTUBE_LIBS_AVAILABLE: return None
    try:
        t_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try: transcript = t_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            gen_langs = [t.language_code for t in t_list if t.is_generated]
            if gen_langs: transcript = t_list.find_generated_transcript(gen_langs)
            else: raise NoTranscriptFound("No generated transcripts")
        logger.info(f"Found transcript ({transcript.language}) for {video_id}")
        return transcript.fetch()
    except (TranscriptsDisabled, NoTranscriptFound) as e: logger.warning(f"Transcript {video_id}: {e}"); return None
    except Exception as e: logger.error(f"Transcript error {video_id}: {e}", exc_info=True); return None

def download_audio(video_id):
    """Downloads audio using yt-dlp if available."""
    if not YOUTUBE_LIBS_AVAILABLE: return None, None
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe(); ytdlp_ok = check_ytdlp_installed()
    if not ffmpeg_path or not os.path.exists(ffmpeg_path) or not ytdlp_ok:
        msg = ("ffmpeg not found. " if not ffmpeg_path else "") + ("yt-dlp not found." if not ytdlp_ok else "")
        st.error(f"{msg} Cannot download audio."); logger.error(msg); return None, None

    try: temp_dir = tempfile.mkdtemp(prefix=f"audio_{video_id}_")
    except Exception as e: logger.error(f"Create temp dir failed: {e}"); return None, None

    safe_id = re.sub(r'[^\w-]', '', video_id); out_tmpl = os.path.join(temp_dir, f"{safe_id}.%(ext)s")
    expected_mp3 = os.path.join(temp_dir, f"{safe_id}.mp3")
    ydl_opts = {'format': 'bestaudio[ext=m4a]/bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
                'outtmpl': out_tmpl, 'quiet': True, 'no_warnings': True, 'noprogress': True, 'ffmpeg_location': ffmpeg_path, 'socket_timeout': 60, 'ignoreconfig': True}

    try:
        logger.info(f"Downloading audio {video_id}...");
        with YoutubeDL(ydl_opts) as ydl: ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        if os.path.exists(expected_mp3): logger.info(f"Audio OK: {expected_mp3}"); return expected_mp3, temp_dir
        else: # Check alternatives if mp3 failed
            found = [f for f in os.listdir(temp_dir) if f.startswith(safe_id) and f.split('.')[-1] in ['m4a', 'opus', 'ogg', 'wav']]
            if found: alt = os.path.join(temp_dir, found[0]); logger.warning(f"Using alt audio: {alt}"); return alt, temp_dir
            else: raise FileNotFoundError("No audio file found after download")
    except (YTDLP_DownloadError, FileNotFoundError) as e: logger.error(f"Audio DL Error {video_id}: {e}"); st.error(f"Audio download failed: {e}")
    except Exception as e: logger.error(f"Audio DL Unexpected {video_id}: {e}", exc_info=True); st.error(f"Audio download error: {e}")
    # Cleanup on error
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    return None, None

def generate_transcript_with_openai(audio_file):
    """Generates transcript using Whisper if available."""
    if not openai or not OPENAI_API_KEY: st.warning("OpenAI unavailable."); return None, None
    max_size = 25*1024*1024; snippet_file = None; cleanup = False; snippet_dur = 0
    try: size = os.path.getsize(audio_file)
    except OSError as e: logger.error(f"Cannot access {audio_file}: {e}"); return None, None

    if size > max_size:
        st.info(f"Audio large ({size/(1<<20):.1f}MB), snippeting..."); logger.warning(f"Audio {audio_file} > 25MB, snippeting.")
        base, ext = os.path.splitext(audio_file); snippet_file = f"{base}_snippet{ext}"
        max_dur = int((max_size * 0.9) / (16 * 1024)); snippet_dur = min(max_dur, 20*60)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", audio_file, "-t", str(snippet_dur), "-c:a", "libmp3lame", "-b:a", "128k", snippet_file]
        try: subprocess.run(cmd, check=True, capture_output=True, timeout=180); file_to_use = snippet_file; cleanup = True
        except Exception as e: logger.error(f"Snippet failed: {e}"); st.error("Snippet creation failed."); return None, None
        if os.path.getsize(file_to_use) > max_size: logger.error("Snippet still too large"); return None, None
    else: file_to_use = audio_file

    try:
        logger.info(f"Transcribing {os.path.basename(file_to_use)} with Whisper...")
        with open(file_to_use, "rb") as f:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.audio.transcriptions.create(model="whisper-1", file=f)
            text = response.text
        dur_est = snippet_dur if snippet_dur > 0 else (size/(16*1024))
        segments = [{"start": 0.0, "duration": dur_est, "text": text or ""}]
        return segments, "openai_whisper"
    except OpenAI_APIError as e: logger.error(f"Whisper API error: {e}"); st.error(f"Whisper API Error: {e}"); return None, None
    except Exception as e: logger.error(f"Whisper unexpected error: {e}", exc_info=True); st.error(f"Whisper error: {e}"); return None, None
    finally:
        if cleanup and snippet_file and os.path.exists(snippet_file):
            try: os.remove(snippet_file); logger.info(f"Removed snippet: {snippet_file}")
            except OSError as e: logger.warning(f"Failed remove snippet {snippet_file}: {e}")

# --- CORRECTED get_transcript_with_fallback ---
def get_transcript_with_fallback(video_id):
    """Gets YouTube transcript, falls back to Whisper if needed and possible."""
    cache_key_data = f"transcript_data_{video_id}"
    cache_key_source = f"transcript_source_{video_id}"
    if cache_key_data in st.session_state:
        return st.session_state[cache_key_data], st.session_state[cache_key_source]

    # 1. Try YouTube
    yt_transcript = get_transcript(video_id)
    if yt_transcript:
        st.session_state[cache_key_data] = yt_transcript
        st.session_state[cache_key_source] = "youtube"
        return yt_transcript, "youtube"

    # 2. Try Whisper fallback
    logger.warning(f"YT transcript failed {video_id}, trying Whisper.")
    if not openai or not OPENAI_API_KEY or not YOUTUBE_LIBS_AVAILABLE:
        st.info("YT transcript unavailable. Whisper fallback not possible (check config/libs).")
        st.session_state[cache_key_data] = None; st.session_state[cache_key_source] = None
        return None, None

    st_placeholder = st.info("Downloading audio for Whisper fallback...")
    audio_path = None
    temp_dir = None
    whisper_transcript = None
    whisper_source = None

    try:
        audio_path, temp_dir = download_audio(video_id) # Assign result correctly

        if audio_path and temp_dir:
            st_placeholder.info("Audio downloaded. Transcribing with Whisper (this may take time)...")
            logger.info(f"Audio downloaded for {video_id} to {audio_path}. Proceeding with Whisper.")
            whisper_transcript, whisper_source = generate_transcript_with_openai(audio_path)

            if whisper_transcript:
                st_placeholder.success("Whisper transcription successful.")
                logger.info(f"Whisper transcription successful for {video_id}.")
            else:
                st_placeholder.error("Whisper transcription failed.")
                logger.error(f"Whisper transcription failed for {video_id}.")
        else:
            st_placeholder.error("Audio download failed. Cannot perform Whisper transcription.")
            logger.error(f"Audio download failed for {video_id}, cannot use Whisper fallback.")

    finally:
         # Always clean up the audio temp directory if it was created
         # Use the correct variable name 'temp_dir'
         if temp_dir and os.path.exists(temp_dir): # Check if dir exists
              try: # Start try on a NEW LINE, indented
                  logger.info(f"Cleaning up audio temp directory: {temp_dir}")
                  shutil.rmtree(temp_dir)
              except Exception as e: # Indent except accordingly
                  logger.error(f"Failed to clean up audio temp directory {temp_dir}: {e}")

    # Store results (or lack thereof) in session state
    st.session_state[cache_key_data] = whisper_transcript
    st.session_state[cache_key_source] = whisper_source
    # Check if placeholder exists before using (might fail during init)
    if 'st_placeholder' in locals() and isinstance(st_placeholder, st.delta_generator.DeltaGenerator):
        try: st_placeholder.empty()
        except Exception: pass # Ignore errors clearing placeholder if already gone
    return whisper_transcript, whisper_source
# --- END CORRECTED FUNCTION ---

# =============================================================================
# 7. AI Summarization (Functions kept from prev step)
# =============================================================================
# get_intro_outro_transcript, summarize_intro_outro, summarize_script

# =============================================================================
# 8. Searching & Calculating Outliers (Functions kept from prev step)
# =============================================================================
# chunk_list, calculate_metrics, fetch_all_snippets, search_youtube

# =============================================================================
# 9. Comments & Analysis (Functions kept from prev step)
# =============================================================================
# analyze_comments, get_video_comments

# =============================================================================
# 10. Retention Analysis (Functions kept from prev step)
# =============================================================================
# check_selenium_setup, load_cookies, capture_player_screenshot_with_hover,
# detect_retention_peaks, capture_frame_at_time, plot_brightness_profile,
# filter_transcript, check_ytdlp_installed, download_video_snippet

# =============================================================================
# 14. UI Pages (Functions kept from prev step)
# =============================================================================
# show_search_page, show_details_page

# =============================================================================
# Main App Logic
# =============================================================================
def main():
    """Main Streamlit execution function."""
    if 'db_initialized' not in st.session_state:
         if not DB_CONN_PATH: # Check if DB init failed
              st.error("Database failed to initialize. Application cannot continue.")
              st.stop() # Stop execution if DB is essential
         st.session_state.db_initialized = True
    if "page" not in st.session_state: st.session_state.page = "search"
    page = st.session_state.get("page")
    if page == "search": show_search_page()
    elif page == "details": show_details_page()
    else: st.session_state.page = "search"; show_search_page() # Default

def app_shutdown(): logger.info("======== App Shutdown ========")

if __name__ == "__main__":
    # Set page config as the absolute first command
    st.set_page_config(page_title="YouTube Niche Analysis", layout="wide", page_icon="📊")
    # Check essential libs early after config
    if not YOUTUBE_LIBS_AVAILABLE:
        st.error("Core YouTube libraries missing. Please install them. App halted.")
        st.stop() # Stop if core libs missing
    try: main()
    except Exception as e:
        logger.critical(f"FATAL: Unhandled exception: {e}", exc_info=True)
        st.error(f"A critical application error occurred: {e}. Please check logs or restart.")
    finally: atexit.register(app_shutdown)

# --- END OF FILE ---
