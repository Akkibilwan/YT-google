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
    from yt_dlp import YoutubeDL, DownloadError as YTDLP_DownloadError
    YOUTUBE_LIBS_AVAILABLE = True
except ImportError:
    st.error("Required YouTube libraries ('youtube-transcript-api', 'yt-dlp') not found. Please add them to requirements.txt")
    YOUTUBE_LIBS_AVAILABLE = False

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
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
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
        st.error("❗ YouTube API key not found in Streamlit secrets (Checked YOUTUBE_API_KEY.key and YT_API_KEY).")
        logger.critical("YouTube API key missing.")
except Exception as e:
    st.error(f"🚨 Error accessing YouTube API key from secrets: {e}")
    logger.error(f"Error loading YouTube API key: {e}", exc_info=True)

# OpenAI API Key
OPENAI_API_KEY = None
if openai: # Only proceed if library was imported
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", {}).get("key")
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            logger.info("OpenAI API key loaded.")
        else:
            st.warning("⚠️ OpenAI API key not found in Streamlit secrets (Checked OPENAI_API_KEY.key). AI features disabled.")
            logger.warning("OpenAI API key missing.")
    except Exception as e:
        st.error(f"🚨 Error accessing OpenAI API key from secrets: {e}")
        logger.error(f"Error loading OpenAI API key: {e}", exc_info=True)
else:
    st.warning("⚠️ OpenAI library not installed. AI features disabled.")

def get_youtube_api_key():
    if not YOUTUBE_API_KEY:
        raise ValueError("YouTube API key is not configured.")
    return YOUTUBE_API_KEY

# Other Config
DB_PATH = "cache.db"
CHANNELS_FILE = "channels.json" # Unused?
FOLDERS_FILE = "channel_folders.json"
COOKIE_FILE = "youtube_cookies.json" # For Selenium retention
CHROMIUM_PATH_STANDARD = "/usr/bin/chromium" # Common path in Linux containers

# =============================================================================
# 3. SQLite DB Setup (Caching)
# =============================================================================
@st.cache_resource # Cache DB connection/initialization
def init_db(db_path=DB_PATH):
    """Initializes the SQLite database and returns a connection pool (or just path)."""
    try:
        # For simplicity in Streamlit, often just using 'with sqlite3.connect' is sufficient
        # unless facing heavy concurrent writes (unlikely here).
        with sqlite3.connect(db_path, timeout=15, check_same_thread=False) as conn:
            conn.execute("PRAGMA journal_mode=WAL;") # Improve concurrency
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
        st.error(f"Database initialization failed: {e}. Caching will not work.")
        return None # Indicate failure

DB_CONN_PATH = init_db() # Initialize DB on app start

def get_cached_result(cache_key, ttl=600, db_path=DB_CONN_PATH):
    if not db_path: return None # Don't try if DB init failed
    now = time.time()
    try:
        # Use 'with' for automatic connection closing/commit/rollback
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.execute("SELECT json_data, timestamp FROM youtube_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
        if row:
            json_data, cached_time = row
            if (now - cached_time) < ttl:
                logger.debug(f"Cache hit: {cache_key[-8:]}")
                return json.loads(json_data)
            else:
                logger.debug(f"Cache expired: {cache_key[-8:]}")
                delete_cache_key(cache_key, db_path)
                return None # Explicitly return None for expired
    except sqlite3.OperationalError as e:
         if "database is locked" in str(e).lower(): logger.warning(f"DB locked getting cache: {cache_key[-8:]}")
         else: logger.error(f"DB error get cache: {cache_key[-8:]}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in cache: {cache_key[-8:]}: {e}. Deleting entry.")
        delete_cache_key(cache_key, db_path)
    except Exception as e:
        logger.error(f"Get cache error: {cache_key[-8:]}: {e}", exc_info=True)
    return None

def set_cached_result(cache_key, data_obj, db_path=DB_CONN_PATH):
    if not db_path: return # Don't try if DB init failed
    now = time.time()
    try:
        json_str = json.dumps(data_obj, default=str) # Ensure serializable
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute("INSERT OR REPLACE INTO youtube_cache (cache_key, json_data, timestamp) VALUES (?, ?, ?)",
                         (cache_key, json_str, now))
        logger.debug(f"Cache set: {cache_key[-8:]}")
    except TypeError as e: logger.error(f"Cache data not JSON serializable: {cache_key[-8:]}: {e}")
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower(): logger.warning(f"DB locked setting cache: {cache_key[-8:]}")
        else: logger.error(f"DB error set cache: {cache_key[-8:]}: {e}")
    except Exception as e: logger.error(f"Set cache error: {cache_key[-8:]}: {e}", exc_info=True)

def delete_cache_key(cache_key, db_path=DB_CONN_PATH):
    if not db_path: return
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute("DELETE FROM youtube_cache WHERE cache_key = ?", (cache_key,))
        logger.debug(f"Cache deleted: {cache_key[-8:]}")
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower(): logger.warning(f"DB locked deleting cache: {cache_key[-8:]}")
        else: logger.error(f"DB error delete cache: {cache_key[-8:]}: {e}")
    except Exception as e: logger.error(f"Delete cache error: {cache_key[-8:]}: {e}", exc_info=True)

def clear_all_cache(db_path=DB_CONN_PATH):
    """Clears the entire cache database table."""
    if not db_path: st.error("Database not initialized, cannot clear cache."); return False
    try:
        with sqlite3.connect(db_path, timeout=15) as conn:
            conn.execute("DELETE FROM youtube_cache")
            conn.execute("VACUUM") # Optional: Clean up space
        logger.info("Cleared all entries from youtube_cache table.")
        # Also clear relevant session state caches
        keys_to_clear = [k for k in st.session_state if k.startswith(('search_', '_search_', 'transcript_', 'comments_', 'analysis_', 'retention_', 'summary_'))]
        for key in keys_to_clear:
            del st.session_state[key]
        logger.info(f"Cleared {len(keys_to_clear)} related session state entries.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to clear cache table: {e}", exc_info=True)
        st.error(f"Failed to clear cache database: {e}")
        return False

# =============================================================================
# 4. Utility Helpers
# =============================================================================
def format_date(date_string):
    if not date_string or not isinstance(date_string, str): return "Unknown"
    try:
        date_string_cleaned = date_string.split('.')[0] + 'Z' if '.' in date_string else date_string
        date_obj = datetime.strptime(date_string_cleaned, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return date_obj.strftime("%d-%m-%y")
    except Exception: return "Invalid Date" # More informative than Unknown

def format_number(num):
    try:
        n = int(num)
        if abs(n) >= 1_000_000: return f"{n/1_000_000:.1f}M"
        elif abs(n) >= 1_000: return f"{n/1_000:.1f}K"
        return str(n)
    except (ValueError, TypeError): return str(num) # Return original if not convertible
    except Exception: return "Err"

def build_cache_key(*args):
    try:
        # Use repr for more stability with different types, sort if list/tuple
        key_parts = []
        for a in args:
            if isinstance(a, (list, tuple)):
                 try: key_parts.append(repr(sorted(a)))
                 except TypeError: key_parts.append(repr(a)) # Cannot sort mixed types
            else: key_parts.append(repr(a))
        raw_str = "-".join(key_parts)
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    except Exception as e:
         logger.error(f"Failed to build cache key from args: {args} - Error: {e}", exc_info=True)
         raise ValueError("Failed to build cache key") from e

def parse_iso8601_duration(duration_str):
    if not duration_str or duration_str == 'P0D': return 0
    try: return int(isodate.parse_duration(duration_str).total_seconds())
    except Exception: logger.debug(f"Could not parse duration: {duration_str}"); return 0

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
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(folders, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(folders)} folders to {file_path}")
        # Clear the cache after saving to force reload on next access
        load_channel_folders.clear()
        return True
    except Exception as e:
        logger.error(f"Error saving folders to {file_path}: {e}", exc_info=True)
        st.error(f"Error saving folder data: {e}")
        return False

# @st.cache_data(ttl=3600) # Cache channel ID lookups for 1 hour
def get_channel_id(channel_name_or_url):
    """Resolves various YouTube channel inputs to a channel ID using Search API."""
    # Caching added via st.cache_data if uncommented above
    identifier = channel_name_or_url.strip()
    if not identifier: return None

    # Quick checks for direct ID or common URL patterns
    if identifier.startswith("UC") and len(identifier) == 24 and re.match(r'^UC[A-Za-z0-9_\-]{22}$', identifier): return identifier
    match_id = re.search(r'youtube\.com/channel/(UC[A-Za-z0-9_\-]{22})', identifier, re.IGNORECASE)
    if match_id: return match_id.group(1)

    # Prepare search query (handle handles specifically)
    search_query = identifier
    if identifier.startswith('@'):
        search_query = identifier # Search API handles '@' handles well

    try:
        key = get_youtube_api_key()
        params = {'key': key, 'part': 'snippet', 'type': 'channel', 'q': search_query, 'maxResults': 1}
        api_url = "https://www.googleapis.com/youtube/v3/search"
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status() # Check for HTTP errors
        data = response.json()

        if data.get("items"):
            item = data["items"][0]
            channel_id = item.get('id', {}).get('channelId')
            found_title = item.get('snippet', {}).get('title', '')
            if channel_id:
                logger.info(f"Resolved '{identifier}' via search to '{found_title}' ({channel_id})")
                return channel_id
            else: logger.warning(f"Search for '{identifier}' succeeded but item lacked channelId.")
        else: logger.warning(f"API search for '{identifier}' returned no items.")

    except ValueError as e: st.error(f"API key error: {e}") # Missing key
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error resolving '{identifier}': {e}", exc_info=True)
        # Simplified error for user based on status code if available
        status_code = getattr(e.response, 'status_code', None)
        if status_code == 403: st.error("API Key/Quota Error resolving channel.")
        elif status_code == 400: st.error("API Bad Request resolving channel.")
        elif status_code: st.error(f"API HTTP Error {status_code} resolving channel.")
        else: st.error(f"Network error resolving channel: {e}")
    except Exception as e:
        logger.error(f"Unexpected error resolving '{identifier}': {e}", exc_info=True)
        st.error(f"Unexpected error resolving channel: {e}")

    return None # Fallback if resolution failed

# --- Folder Management UI ---
def show_channel_folder_manager():
    st.write("#### Manage Channel Folders")
    folders = load_channel_folders()
    folder_keys = list(folders.keys())

    action = st.selectbox("Action", ["Add Channels", "Remove Channel", "Create Folder", "Delete Folder"], key="folder_action_select")

    if action == "Create Folder":
        # Use form for better UX
        with st.form("create_folder_form"):
            new_folder_name = st.text_input("New Folder Name:", key="new_folder_name_input").strip()
            submitted = st.form_submit_button("Create Empty Folder")
            if submitted:
                if not new_folder_name: st.error("Folder name cannot be empty.")
                elif new_folder_name in folders: st.error(f"Folder '{new_folder_name}' already exists.")
                else:
                    folders[new_folder_name] = []
                    if save_channel_folders(folders):
                        st.success(f"Folder '{new_folder_name}' created.")
                        st.rerun() # Update folder lists immediately

    elif action == "Add Channels":
        if not folder_keys: st.info("Create a folder first."); return
        target_folder = st.selectbox("Add to Folder:", folder_keys, key="add_to_folder_select")
        with st.form("add_channels_form"):
            st.write("Enter Channel names, URLs, or IDs (one per line):")
            channels_text = st.text_area("Channels to Add:", height=100, key="add_channels_input")
            submitted = st.form_submit_button("Resolve and Add")
            if submitted and target_folder:
                lines = channels_text.strip().split("\n")
                added_count = 0
                current_ids = {ch['channel_id'] for ch in folders.get(target_folder, [])}
                with st.spinner("Resolving channels..."):
                    for line in lines:
                        line = line.strip();
                        if not line: continue
                        ch_id = get_channel_id(line)
                        if ch_id and ch_id not in current_ids:
                            ch_name = line # Default name
                            try: # Fetch real name
                                details = requests.get(f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={ch_id}&key={get_youtube_api_key()}", timeout=5).json()
                                if details.get('items'): ch_name = details['items'][0]['snippet']['title']
                            except Exception: pass # Ignore errors fetching name
                            folders[target_folder].append({"channel_name": ch_name, "channel_id": ch_id})
                            current_ids.add(ch_id)
                            added_count += 1
                        elif ch_id: st.warning(f"'{line}' ({ch_id}) already in folder.")
                        else: st.warning(f"Could not resolve '{line}'.")
                if added_count > 0:
                    if save_channel_folders(folders):
                        st.success(f"Added {added_count} channel(s) to '{target_folder}'.")
                        # Optional: Trigger pre-caching here?
                        st.rerun()
                else: st.info("No new channels added.")

    elif action == "Remove Channel":
        if not folder_keys: st.info("No folders to modify."); return
        target_folder = st.selectbox("Remove from Folder:", folder_keys, key="remove_from_folder_select")
        if target_folder and folders.get(target_folder):
            current_channels = folders[target_folder]
            if not current_channels: st.info("This folder is empty."); return
            options = {f"{ch['channel_name']} ({ch['channel_id']})": ch['channel_id'] for ch in current_channels}
            choice_display = st.selectbox("Select channel to remove:", list(options.keys()), key="remove_channel_select")
            if st.button("Remove Selected Channel", type="secondary"):
                choice_id = options.get(choice_display)
                if choice_id:
                    folders[target_folder] = [ch for ch in current_channels if ch['channel_id'] != choice_id]
                    if save_channel_folders(folders):
                        st.success(f"Removed '{choice_display}' from '{target_folder}'.")
                        st.rerun()
        elif target_folder: st.info("Folder is empty.")

    elif action == "Delete Folder":
        if not folder_keys: st.info("No folders to delete."); return
        folder_to_delete = st.selectbox("Select Folder to Delete:", folder_keys, key="delete_folder_select")
        st.warning(f"⚠️ Delete '{folder_to_delete}' permanently?")
        if st.button(f"Confirm Delete '{folder_to_delete}'", type="primary"):
            if folder_to_delete in folders:
                del folders[folder_to_delete]
                if save_channel_folders(folders):
                    st.success(f"Folder '{folder_to_delete}' deleted.")
                    st.rerun()
            else: st.error("Folder not found.")

# =============================================================================
# 6. Transcript & Fallback
# =============================================================================
# Ensure YOUTUBE_LIBS_AVAILABLE check
def get_transcript(video_id):
    """Fetches YouTube transcript if library available."""
    if not YOUTUBE_LIBS_AVAILABLE: return None
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try: transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            generated_languages = [t.language_code for t in transcript_list if t.is_generated]
            if generated_languages: transcript = transcript_list.find_generated_transcript(generated_languages)
            else: raise NoTranscriptFound("No generated transcripts found") # Raise if none generated
        logger.info(f"Found transcript ({transcript.language}) for {video_id}")
        return transcript.fetch()
    except TranscriptsDisabled: logger.warning(f"Transcripts disabled for {video_id}"); return None
    except NoTranscriptFound: logger.warning(f"No suitable transcript found for {video_id}"); return None
    except Exception as e: logger.error(f"Transcript fetch error {video_id}: {e}", exc_info=True); return None

def download_audio(video_id):
    """Downloads audio using yt-dlp if available."""
    if not YOUTUBE_LIBS_AVAILABLE: return None, None
    # Dependency checks
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        st.error("ffmpeg not found/path invalid. Cannot download audio.")
        logger.error("ffmpeg missing for audio download.")
        return None, None
    if not check_ytdlp_installed():
        st.error("yt-dlp not found. Cannot download audio.")
        logger.error("yt-dlp missing for audio download.")
        return None, None

    try: temp_dir = tempfile.mkdtemp(prefix=f"audio_{video_id}_")
    except Exception as e: logger.error(f"Failed create temp dir: {e}"); return None, None

    safe_id = re.sub(r'[^\w-]', '', video_id)
    out_tmpl = os.path.join(temp_dir, f"{safe_id}.%(ext)s")
    expected_mp3 = os.path.join(temp_dir, f"{safe_id}.mp3")

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
        'outtmpl': out_tmpl, 'quiet': True, 'no_warnings': True, 'noprogress': True,
        'ffmpeg_location': ffmpeg_path, 'socket_timeout': 60, 'ignoreconfig': True,
    }

    try:
        logger.info(f"Downloading audio for {video_id}...")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        if os.path.exists(expected_mp3):
            logger.info(f"Audio downloaded: {expected_mp3}")
            return expected_mp3, temp_dir
        else: # Check for other formats if mp3 conversion failed
            found = [f for f in os.listdir(temp_dir) if f.startswith(safe_id) and f.split('.')[-1] in ['m4a', 'opus', 'ogg', 'wav']]
            if found:
                alt_path = os.path.join(temp_dir, found[0])
                logger.warning(f"MP3 missing, using other format: {alt_path}")
                return alt_path, temp_dir
            else: logger.error(f"No audio file found after download: {video_id}"); shutil.rmtree(temp_dir); return None, None
    except YTDLP_DownloadError as e: logger.error(f"yt-dlp DownloadError {video_id}: {e}"); st.error(f"Audio download failed: {e}"); shutil.rmtree(temp_dir); return None, None
    except Exception as e: logger.error(f"Audio download unexpected error {video_id}: {e}", exc_info=True); st.error(f"Audio download error: {e}"); shutil.rmtree(temp_dir); return None, None

def generate_transcript_with_openai(audio_file):
    """Generates transcript using Whisper if available."""
    if not openai or not OPENAI_API_KEY:
        st.warning("OpenAI not available/configured. Cannot use Whisper.")
        return None, None

    max_size_bytes = 25 * 1024 * 1024
    temp_snippet_file = None
    cleanup_snippet = False
    snippet_duration_sec = 0

    try: actual_size = os.path.getsize(audio_file)
    except OSError as e: logger.error(f"Cannot access audio file {audio_file}: {e}"); return None, None

    if actual_size > max_size_bytes:
        st.info(f"Audio large ({actual_size/(1024*1024):.1f}MB), snippeting...")
        logger.warning(f"Audio {audio_file} > 25MB, creating snippet.")
        base, ext = os.path.splitext(audio_file)
        temp_snippet_file = f"{base}_snippet{ext}"
        estimated_max_duration = int((max_size_bytes * 0.9) / (16 * 1024)) # ~16KBps for 128k mp3
        snippet_duration_sec = min(estimated_max_duration, 20 * 60) # Max 20 min snippet
        ffmpeg_cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", audio_file, "-t", str(snippet_duration_sec), "-c:a", "libmp3lame", "-b:a", "128k", temp_snippet_file]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=180)
            logger.info(f"Created snippet: {temp_snippet_file}")
            file_to_transcribe = temp_snippet_file; cleanup_snippet = True
            # Verify size
            if os.path.getsize(file_to_transcribe) > max_size_bytes: raise ValueError("Snippet still too large.")
        except Exception as e:
            logger.error(f"Snippet creation failed: {e}")
            st.error("Failed to create audio snippet for transcription.")
            return None, None
    else:
        file_to_transcribe = audio_file

    try:
        logger.info(f"Transcribing {os.path.basename(file_to_transcribe)} with Whisper...")
        with open(file_to_transcribe, "rb") as f_handle:
            # Use v1.0+ client syntax
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.audio.transcriptions.create(model="whisper-1", file=f_handle)
            text = response.text
        logger.info("Whisper transcription successful.")
        duration_est = snippet_duration_sec if snippet_duration_sec > 0 else (actual_size / (16*1024))
        segments = [{"start": 0.0, "duration": duration_est, "text": text or ""}]
        return segments, "openai_whisper"
    except OpenAI_APIError as e: logger.error(f"Whisper API error: {e}"); st.error(f"Whisper API Error: {e}"); return None, None
    except Exception as e: logger.error(f"Whisper unexpected error: {e}", exc_info=True); st.error(f"Whisper error: {e}"); return None, None
    finally:
        if cleanup_snippet and temp_snippet_file and os.path.exists(temp_snippet_file):
            try: os.remove(temp_snippet_file); logger.info(f"Removed snippet: {temp_snippet_file}")
            except OSError as e: logger.warning(f"Failed remove snippet {temp_snippet_file}: {e}")

def get_transcript_with_fallback(video_id):
    """Gets YouTube transcript, falls back to Whisper if needed and possible."""
    cache_key_data = f"transcript_data_{video_id}"
    cache_key_source = f"transcript_source_{video_id}"
    if cache_key_data in st.session_state: return st.session_state[cache_key_data], st.session_state[cache_key_source]

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
    audio_path, temp_dir = download_audio(video_id)
    whisper_transcript, whisper_source = None, None
    if audio_path and temp_dir:
        try:
            st_placeholder.info("Transcribing audio with Whisper...")
            whisper_transcript, whisper_source = generate_transcript_with_openai(audio_path)
            if whisper_transcript: st_placeholder.success("Whisper transcription complete.")
            else: st_placeholder.error("Whisper transcription failed.")
        finally:
            if temp_dir and os.path.exists(temp_dir): try: shutil.rmtree(temp_dir) # Cleanup audio dir
            except Exception as e: logger.error(f"Failed cleanup audio dir {temp_dir}: {e}")
    else: st_placeholder.error("Audio download failed for Whisper.")

    st.session_state[cache_key_data] = whisper_transcript
    st.session_state[cache_key_source] = whisper_source
    st_placeholder.empty() # Clear status message
    return whisper_transcript, whisper_source

# =============================================================================
# 7. AI Summarization (Intro/Outro, Full Script) - Functions kept from prev step
# =============================================================================
# get_intro_outro_transcript, summarize_intro_outro, summarize_script functions
# remain the same as in the previous corrected version.
def get_intro_outro_transcript(video_id, total_duration):
    """Extracts text snippets for the intro and outro from a transcript."""
    transcript, source = get_transcript_with_fallback(video_id)
    if not transcript:
        logger.warning(f"No transcript available for {video_id} to extract intro/outro.")
        return (None, None)
    end_intro_sec = min(60, total_duration * 0.2 if total_duration > 0 else 60)
    start_outro_sec = -1
    if total_duration > 120: start_outro_sec = max(end_intro_sec, total_duration - 60)
    logger.debug(f"Extracting intro/outro {video_id}: IntroEnd={end_intro_sec:.1f}s, OutroStart={start_outro_sec:.1f}s")
    intro_texts, outro_texts = [], []
    if source == "youtube" and isinstance(transcript, list) and transcript and 'start' in transcript[0]:
        for item in transcript:
            try:
                start = float(item["start"]); duration = float(item.get("duration", 2.0)); end = start + duration
                text = item.get("text", "").strip();
                if not text: continue
                if max(0, start) < end_intro_sec: intro_texts.append(text)
                if start_outro_sec >= 0 and max(start_outro_sec, start) < min(total_duration, end): outro_texts.append(text)
            except Exception: continue # Skip bad segments
    elif source == "openai_whisper" and isinstance(transcript, list) and transcript and 'text' in transcript[0]:
        full_text = " ".join(seg.get("text", "") for seg in transcript).strip()
        if total_duration > 0 and len(full_text) > 20:
            words = full_text.split(); num_words = len(words); wps = num_words / total_duration
            intro_limit = int(end_intro_sec * wps); intro_approx = " ".join(words[:intro_limit])
            outro_approx = None
            if start_outro_sec >= 0:
                outro_start_idx = int(start_outro_sec * wps)
                if outro_start_idx < num_words: outro_approx = " ".join(words[outro_start_idx:])
            intro_full = intro_approx if intro_approx and len(intro_approx) > 10 else None
            outro_full = outro_approx if outro_approx and len(outro_approx) > 10 else None
            return (intro_full, outro_full)
        else: return (None, None) # Cannot split
    else: return (None, None) # Unknown format
    intro_full = " ".join(intro_texts) if intro_texts else None
    outro_full = " ".join(outro_texts) if outro_texts else None
    return (intro_full, outro_full)

def summarize_intro_outro(intro_text, outro_text):
    """Summarizes intro/outro text using OpenAI, with session caching."""
    if not intro_text and not outro_text: return (None, None)
    if not openai or not OPENAI_API_KEY: return ("*AI Summary unavailable (check config)*",)*2

    combined = (intro_text or "")[:2000] + "||" + (outro_text or "")[:2000] # Limit text for hash/prompt
    cache_key = f"summary_io_{hashlib.sha256(combined.encode()).hexdigest()}"
    if cache_key in st.session_state: return (st.session_state[cache_key],)*2

    prompt = "Summarize the key points/hooks from the Intro and the takeaways/CTAs from the Outro in brief bullet points (max 3-4 each).\n"
    if intro_text: prompt += f"\n**Intro Snippet:**\n'''{intro_text[:2000]}'''\n"
    else: prompt += "\n**Intro Snippet:** Not available.\n"
    if outro_text: prompt += f"\n**Outro Snippet:**\n'''{outro_text[:2000]}'''\n"
    else: prompt += "\n**Outro Snippet:** Not available.\n"
    prompt += "\n**Summary Output:**\n**Intro Summary:**\n- ...\n**Outro Summary:**\n- ..."

    try:
        logger.info("Summarizing intro/outro with OpenAI...")
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.5)
        result_txt = response.choices[0].message.content.strip() or "*AI returned empty summary.*"
        st.session_state[cache_key] = result_txt
        return (result_txt, result_txt)
    except OpenAI_APIError as e: logger.error(f"OpenAI API error IO summary: {e}"); return ("*API Error*",)*2
    except Exception as e: logger.error(f"Unexpected error IO summary: {e}", exc_info=True); return ("*Error*",)*2

def summarize_script(script_text):
    """Summarizes full script text using OpenAI, with session caching."""
    if not script_text or not script_text.strip(): return "No script text."
    if not openai or not OPENAI_API_KEY: return "*AI Summary unavailable (check config)*"

    hash_key = hashlib.sha256(script_text[:1000].encode() + script_text[-1000:].encode()).hexdigest()
    cache_key = f"summary_full_{hash_key}"
    if cache_key in st.session_state: return st.session_state[cache_key]

    max_chars = 12000 # Limit input size
    truncated = script_text[:max_chars] + (" [...truncated]" if len(script_text) > max_chars else "")
    prompt = f"Provide a concise, neutral summary (~150 words) of the main topics in this script:\n'''{truncated}'''"

    try:
        logger.info("Summarizing full script with OpenAI...")
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}],
            max_tokens=250, temperature=0.5)
        summary = response.choices[0].message.content.strip() or "*AI returned empty summary.*"
        st.session_state[cache_key] = summary
        return summary
    except OpenAI_APIError as e: logger.error(f"OpenAI API error full summary: {e}"); return "*API Error*"
    except Exception as e: logger.error(f"Unexpected error full summary: {e}", exc_info=True); return "*Error*"

# =============================================================================
# 8. Searching & Calculating Outliers (Functions kept from prev step)
# =============================================================================
# chunk_list, calculate_metrics, fetch_all_snippets, search_youtube functions
# remain the same as in the previous corrected version.

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n): yield lst[i:i + n]

def calculate_metrics(df):
    """Calculates performance metrics including the outlier score."""
    if df.empty: return df, None
    now_utc = datetime.now(timezone.utc)
    if 'published_at' not in df.columns: logger.error("Missing 'published_at'"); return pd.DataFrame(), None
    df['published_at_dt'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
    initial_rows = len(df); df.dropna(subset=['published_at_dt'], inplace=True)
    if len(df) < initial_rows: logger.warning(f"Dropped {initial_rows - len(df)} rows: invalid dates.")
    if df.empty: return pd.DataFrame(), None
    df['hours_since_published'] = ((now_utc - df['published_at_dt']).dt.total_seconds() / 3600).apply(lambda x: max(x, 1/60))
    df['days_since_published'] = df['hours_since_published'] / 24
    stat_cols = ['views', 'like_count', 'comment_count']
    for col in stat_cols:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['raw_vph'] = df['views'] / df['hours_since_published']
    df['peak_hours'] = df['hours_since_published'].apply(lambda x: min(x, 720.0)).apply(lambda x: max(x, 1/60))
    df['peak_vph'] = df['views'] / df['peak_hours']
    df['effective_vph'] = df.apply(lambda r: r['raw_vph'] if r['days_since_published'] < 90 else r['peak_vph'], axis=1)
    df['engagement_metric'] = df['like_count'] + 5 * df['comment_count']
    df['engagement_rate'] = df['engagement_metric'] / df['views'].apply(lambda x: max(x, 1))
    df['cvr_float'] = df['comment_count'] / df['views'].apply(lambda x: max(x, 1))
    df['clr_float'] = df['like_count'] / df['views'].apply(lambda x: max(x, 1))
    results_list = []
    if 'channelId' not in df.columns: logger.error("Missing 'channelId'"); return pd.DataFrame(), None
    for channel_id, group in df.groupby('channelId'):
        group = group.copy()
        recent = group[group['days_since_published'] <= 30].copy(); period = 30
        if len(recent) < 5: recent = group[group['days_since_published'] <= 90].copy(); period = 90
        if len(recent) < 5: recent = group.copy(); period = int(group['days_since_published'].max()) if not group.empty else 0; logger.warning(f"{channel_id}: Using all {len(group)} videos for avg")
        num_avg = len(recent)
        if num_avg > 0:
            avg_vph = recent['effective_vph'].mean(); avg_eng = recent['engagement_rate'].mean()
            avg_cvr = recent['cvr_float'].mean(); avg_clr = recent['clr_float'].mean()
        else: avg_vph, avg_eng, avg_cvr, avg_clr = 0.1, 0.0001, 0.0001, 0.0001; logger.warning(f"{channel_id}: No videos for averaging.")
        group['channel_avg_vph'] = avg_vph; group['channel_avg_engagement'] = avg_eng
        group['channel_avg_cvr'] = avg_cvr; group['channel_avg_clr'] = avg_clr
        group['vph_ratio'] = group['effective_vph'] / max(avg_vph, 0.1)
        group['engagement_ratio'] = group['engagement_rate'] / max(avg_eng, 0.0001)
        group['outlier_cvr'] = group['cvr_float'] / max(avg_cvr, 0.0001)
        group['outlier_clr'] = group['clr_float'] / max(avg_clr, 0.0001)
        vph_w, eng_w = 0.85, 0.15 # Weights for outlier score
        group['combined_performance'] = (vph_w * group['vph_ratio']) + (eng_w * group['engagement_ratio'])
        group['outlier_score'] = group['combined_performance']
        group['breakout_score'] = group['outlier_score'] # Alias
        group['formatted_views'] = group['views'].apply(format_number)
        group['comment_to_view_ratio'] = group['cvr_float'].apply(lambda x: f"{x*100:.2f}%")
        group['like_to_view_ratio'] = group['clr_float'].apply(lambda x: f"{x*100:.2f}%")
        group['vph_display'] = group['effective_vph'].apply(lambda x: f"{int(round(x,0))} VPH" if x>0 else "0 VPH")
        results_list.append(group)
    if not results_list: return pd.DataFrame(), None
    final_df = pd.concat(results_list).reset_index(drop=True)
    logger.info(f"Calculated metrics for {len(final_df)} videos.")
    return final_df, None

def fetch_all_snippets(channel_id, order_param, timeframe, query, published_after):
    """Fetches basic video snippets via Search API."""
    all_videos = []; page_token = None; fetched_count = 0; max_results = 200; max_pages = (max_results+49)//50
    try: key = get_youtube_api_key()
    except ValueError as e: st.error(f"API key error: {e}"); return []
    base_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={channel_id}&maxResults=50&type=video&order={order_param}&key={key}"
    if published_after: base_url += f"&publishedAfter={published_after}"
    if query: from urllib.parse import quote_plus; base_url += f"&q={quote_plus(query)}"
    logger.info(f"Fetching snippets {channel_id} (Order:{order_param}, Q:'{query}', After:{published_after}, Max:{max_results})")
    for page_num in range(max_pages):
        url = base_url + (f"&pageToken={page_token}" if page_token else "")
        try:
            resp = requests.get(url, timeout=20); resp.raise_for_status()
            data = resp.json(); items = data.get("items", [])
            if not items and fetched_count == 0: break # No results
            for it in items:
                vid_id = it.get("id", {}).get("videoId"); snippet = it.get("snippet")
                published_at = snippet.get("publishedAt") if snippet else None
                if not vid_id or not snippet or not published_at: continue # Skip invalid
                all_videos.append({
                    "video_id": vid_id, "title": snippet.get("title", "N/A"),
                    "channel_name": snippet.get("channelTitle", "N/A"),
                    "channelId": snippet.get("channelId", channel_id),
                    "publish_date": format_date(published_at), "published_at": published_at,
                    "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", "")})
                fetched_count += 1
                if fetched_count >= max_results: break
            if fetched_count >= max_results: break
            page_token = data.get("nextPageToken");
            if not page_token: break
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            logger.error(f"HTTP Error {status} fetching snippets {channel_id}: {e.response.text}")
            if status == 403: st.error("API Key/Quota Error fetching snippets.")
            elif status == 400: st.error("API Bad Request fetching snippets.")
            else: st.error(f"HTTP Error {status} fetching snippets.")
            break # Stop on error
        except requests.exceptions.RequestException as e: logger.error(f"Network Error snippets {channel_id}: {e}"); st.error("Network error fetching snippets."); break
    logger.info(f"Fetched {len(all_videos)} snippets for {channel_id}.")
    return all_videos

def search_youtube(query, channel_ids, timeframe, content_filter, ttl=600):
    """Searches YT, fetches details, calculates metrics, caches, returns list of dicts."""
    query = query.strip(); channel_ids_tuple = tuple(sorted(channel_ids))
    is_broad = not query and timeframe == "3 months" and content_filter == "Both"
    effective_ttl = 7776000 if is_broad else ttl
    cache_key = build_cache_key(query, channel_ids_tuple, timeframe, content_filter)
    cached_list = get_cached_result(cache_key, ttl=effective_ttl)
    if cached_list is not None and isinstance(cached_list, list): # Check cache validity
        is_valid_cache = not cached_list or ('video_id' in cached_list[0] and 'outlier_score' in cached_list[0])
        if is_valid_cache:
             logger.info(f"Cache hit: {cache_key[-8:]}");
             # Apply content filter post-cache
             if content_filter == "Shorts": return [r for r in cached_list if r.get("content_category") == "Short"]
             if content_filter == "Videos": return [r for r in cached_list if r.get("content_category") == "Video"]
             return cached_list
        else: logger.warning(f"Invalid cache format: {cache_key[-8:]}"); delete_cache_key(cache_key)

    logger.info(f"Cache miss/invalid: {cache_key[-8:]}. Fetching fresh data.")
    st_placeholder = st.info("Fetching data from YouTube API...")
    order = "relevance" if query else "date"; pub_after = None
    if timeframe != "Lifetime":
        deltas = {"Last 24 hours": 1, "Last 48 hours": 2, "Last 4 days": 4, "Last 7 days": 7, "Last 15 days": 15, "Last 28 days": 28, "3 months": 90}
        if timeframe in deltas: pub_after = (datetime.now(timezone.utc) - timedelta(days=deltas[timeframe])).strftime('%Y-%m-%dT%H:%M:%SZ')
    all_snippets = []
    with st.spinner(f"Fetching video list for {len(channel_ids)} channels..."):
        for cid in channel_ids: all_snippets.extend(fetch_all_snippets(cid, order, timeframe, query, pub_after))
    if not all_snippets: st_placeholder.warning("No videos found."); set_cached_result(cache_key, []); return []
    unique_snippets = list({s['video_id']: s for s in all_snippets}.values())
    vid_ids = [s["video_id"] for s in unique_snippets]; logger.info(f"Found {len(vid_ids)} unique IDs.")
    all_details = {}; total_fetched = 0
    try:
        key = get_youtube_api_key()
        with st.spinner(f"Fetching details for {len(vid_ids)} videos...") as det_spinner:
            for i, chunk in enumerate(chunk_list(vid_ids, 50)):
                ids_str = ','.join(chunk)
                url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={ids_str}&key={key}"
                try:
                    resp = requests.get(url, timeout=25); resp.raise_for_status(); data = resp.json()
                    items = data.get("items", []); total_fetched += len(items)
                    for item in items:
                        vid = item["id"]; stats = item.get("statistics", {}); content = item.get("contentDetails", {}); snip = item.get("snippet", {})
                        duration = parse_iso8601_duration(content.get("duration", "PT0S"))
                        is_short = snip.get("liveBroadcastContent", "none") == "none" and 0 < duration <= 60
                        all_details[vid] = {
                            "views": int(stats.get("viewCount", 0)), "like_count": int(stats.get("likeCount", 0)),
                            "comment_count": int(stats.get("commentCount", 0)), "duration_seconds": duration,
                            "content_category": "Short" if is_short else "Video", "published_at": snip.get("publishedAt")}
                except requests.exceptions.RequestException as e: logger.error(f"Details fetch error chunk {i+1}: {e}"); st.warning(f"API error fetching details chunk {i+1}.")
    except ValueError as e: st_placeholder.error(f"API key error: {e}"); set_cached_result(cache_key, []); return []
    except Exception as e: st_placeholder.error(f"Unexpected error fetching details: {e}"); logger.error(f"Details fetch setup error: {e}", exc_info=True); set_cached_result(cache_key, []); return []

    combined = []
    missing_count = 0
    for snip_data in unique_snippets:
        vid = snip_data["video_id"]
        if vid in all_details:
            details = all_details[vid]
            if details.get("published_at"): # Use details API publish time
                 snip_data["published_at"] = details["published_at"]
                 snip_data["publish_date"] = format_date(details["published_at"])
            combined.append({**snip_data, **details})
        else: missing_count += 1; logger.warning(f"Missing details for {vid}")
    if missing_count > 0: st.warning(f"Could not fetch details for {missing_count} videos.")
    if not combined: st_placeholder.warning("Could not retrieve details for any videos."); set_cached_result(cache_key, []); return []

    st_placeholder.info("Calculating metrics...")
    try:
        df_combined = pd.DataFrame(combined)
        df_metrics, _ = calculate_metrics(df_combined)
        if df_metrics is None or df_metrics.empty: final_list = []
        else: final_list = df_metrics.to_dict("records")
        try: set_cached_result(cache_key, final_list); logger.info(f"Cached {len(final_list)} results: {cache_key[-8:]}")
        except Exception as e: logger.error(f"Failed to cache results: {e}", exc_info=True)
    except Exception as e: logger.error(f"Metrics calc error: {e}", exc_info=True); st_placeholder.error(f"Error calculating metrics: {e}"); final_list = []

    st_placeholder.empty()
    # Apply content filter post-cache
    if content_filter == "Shorts": filtered_list = [r for r in final_list if r.get("content_category") == "Short"]
    elif content_filter == "Videos": filtered_list = [r for r in final_list if r.get("content_category") == "Video"]
    else: filtered_list = final_list
    logger.info(f"Returning {len(filtered_list)} videos (Filter: {content_filter})")
    return filtered_list

# =============================================================================
# 9. Comments & Analysis (Functions kept from prev step)
# =============================================================================
# analyze_comments, get_video_comments

# =============================================================================
# 10. Retention Analysis (Functions kept from prev step, ensure SELENIUM_AVAILABLE check)
# =============================================================================
# check_selenium_setup, load_cookies, capture_player_screenshot_with_hover,
# detect_retention_peaks, capture_frame_at_time, plot_brightness_profile,
# filter_transcript, check_ytdlp_installed, download_video_snippet

# =============================================================================
# 14. UI Pages (Functions kept from prev step, added SELENIUM_AVAILABLE checks)
# =============================================================================
def show_search_page():
    st.title("🚀 YouTube Niche Search")
    with st.sidebar: # --- Sidebar ---
        st.header("⚙️ Search Filters")
        folders = load_channel_folders(); available_folders = list(folders.keys()); folder_choice = None
        if not available_folders: st.warning("No folders found."); st.caption("Use 'Manage Folders' below.")
        else: folder_choice = st.selectbox("Channel Folder", available_folders, index=0 if available_folders else -1, key="folder_choice_sb")
        timeframe_opts = ["Last 24 hours", "Last 48 hours", "Last 4 days", "Last 7 days", "Last 15 days", "Last 28 days", "3 months", "Lifetime"]
        selected_timeframe = st.selectbox("Timeframe", timeframe_opts, index=timeframe_opts.index("3 months"), key="timeframe_sb")
        content_opts = ["Both", "Videos", "Shorts"]; content_filter = st.selectbox("Content Type", content_opts, index=0, key="content_filter_sb")
        min_outlier_score = st.number_input("Min Outlier Score", value=0.0, min_value=0.0, step=0.1, format="%.2f", key="min_outlier_sb", help="Performance vs channel avg (1.0=avg).")
        search_query = st.text_input("Keyword Search (opt)", key="query_sb", placeholder="e.g., AI tools")
        search_pressed = st.button("🔍 Search Videos", key="search_btn_sb", type="primary", use_container_width=True)
        st.divider()
        if st.button("🔄 Clear Cache", help="Force fresh data fetch."):
            if clear_all_cache(): st.success("Cache cleared. Search again.")
        st.divider()
        with st.expander("📂 Manage Folders", expanded=False): show_channel_folder_manager()

    # --- Main Page ---
    selected_ids = []
    if folder_choice and folder_choice in folders:
        st.subheader(f"Folder: {folder_choice}"); selected_ids = [ch["channel_id"] for ch in folders[folder_choice]]
        if not selected_ids: st.warning("Folder empty.")
        else:
            with st.expander("Channels", expanded=False):
                ch_names = [ch['channel_name'] for ch in folders[folder_choice]]
                st.caption(" | ".join(ch_names[:20]) + ("..." if len(ch_names)>20 else ""))
    elif not available_folders: st.info("Create a folder first.")
    else: st.info("Select a folder.")

    # --- Search Execution ---
    if search_pressed:
        if not folder_choice or not selected_ids: st.error("Select folder with channels.")
        else:
            st.session_state.search_params = {"query": search_query, "channel_ids": selected_ids, "timeframe": selected_timeframe, "content_filter": content_filter, "min_outlier_score": min_outlier_score, "folder_choice": folder_choice}
            if st.session_state.get('_search_params_for_results') != st.session_state.search_params:
                 if 'search_results' in st.session_state: del st.session_state['search_results']
            st.session_state.page = "search"; st.rerun()

    # --- Display Results ---
    if 'search_params' in st.session_state and st.session_state.get("page") == "search":
        params = st.session_state.search_params
        if 'search_results' not in st.session_state or st.session_state.get('_search_params_for_results') != params:
            try: results = search_youtube(params["query"], params["channel_ids"], params["timeframe"], params["content_filter"])
            except Exception as e: st.error(f"Search Error: {e}"); results = []
            st.session_state.search_results = results; st.session_state._search_params_for_results = params
        results_to_show = st.session_state.get('search_results', [])
        min_score = params["min_outlier_score"]
        filtered = [r for r in results_to_show if pd.to_numeric(r.get("outlier_score"), errors='coerce') is not None and pd.to_numeric(r.get("outlier_score"), errors='coerce') >= min_score] if min_score > 0 else results_to_show
        if min_score > 0 and len(results_to_show) > len(filtered): st.caption(f"Filtered {len(results_to_show) - len(filtered)} results below score {min_score:.2f}.")

        if not filtered:
             if search_pressed or '_search_params_for_results' in st.session_state: st.info("No videos found matching criteria.")
        else:
            sort_map = {"Outlier Score": "outlier_score", "Upload Date": "published_at", "Views": "views", "Effective VPH": "effective_vph", "VPH Ratio": "vph_ratio", "C/V Ratio": "cvr_float", "L/V Ratio": "clr_float", "Comments": "comment_count", "Likes": "like_count"}
            sort_label = st.selectbox("Sort by:", list(sort_map.keys()), index=0, key="sort_select")
            sort_key = sort_map[sort_label]
            try:
                rev = True if sort_key != "published_at" else False # Dates ascending = older first? No, descending=newer first
                if sort_key == "published_at": sorted_data = sorted(filtered, key=lambda x: x.get(sort_key, "1970"), reverse=True)
                else: sorted_data = sorted(filtered, key=lambda x: pd.to_numeric(x.get(sort_key), errors='coerce') or 0, reverse=True)
            except Exception as e: st.error(f"Sort error: {e}"); sorted_data = filtered
            st.subheader(f"📊 Found {len(sorted_data)} Videos", anchor=False); st.caption(f"Sorted by {sort_label}")

            cols_per_row = 3
            for i in range(0, len(sorted_data), cols_per_row):
                 cols = st.columns(cols_per_row)
                 row_items = sorted_data[i : i + cols_per_row]
                 for j, item in enumerate(row_items):
                      with cols[j]:
                           days = int(round(item.get("days_since_published", 0))); days_text = f"{days}d ago" if days > 1 else ("1d ago" if days == 1 else "today")
                           score = pd.to_numeric(item.get('outlier_score'), errors='coerce'); score_str = f"{score:.2f}x" if score is not None else "N/A"
                           score_col = "#9aa0a6" # Grey
                           if score is not None:
                               if score >= 1.75: score_col = "#1e8e3e"; # Dark Green
                               elif score >= 1.15: score_col = "#34a853"; # Green
                               elif score >= 0.85: score_col = "#4285f4"; # Blue
                               elif score >= 0.5: score_col = "#fbbc04"; # Orange
                               else: score_col = "#ea4335" # Red
                           badge = f'<span style="background-color:{score_col};color:white;padding:2px 7px;border-radius:10px;font-size:0.8em;font-weight:bold;">{score_str}</span>'
                           watch = f"https://www.youtube.com/watch?v={item['video_id']}"; thumb = item.get('thumbnail', '') or 'https://placehold.co/320x180?text=N/A'
                           card_style = "border:1px solid #dfe1e5; border-radius:8px; padding:12px; margin-bottom:16px; height:430px; display:flex; flex-direction:column; background:#fff; box-shadow:0 1px 2px 0 rgba(60,64,67,.3),0 1px 3px 1px rgba(60,64,67,.15); transition:box-shadow .2s ease-in-out;"
                           card_html = f'''<div style="{card_style}" onmouseover="this.style.boxShadow='0 1px 3px 0 rgba(60,64,67,.3), 0 4px 8px 3px rgba(60,64,67,.15)'" onmouseout="this.style.boxShadow='0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15)'">
                               <a href="{watch}" target="_blank" style="text-decoration:none;color:inherit;display:block;"><img src="{thumb}" alt="Thumb" style="width:100%;border-radius:4px;margin-bottom:10px;object-fit:cover;aspect-ratio:16/9;" /><div title="{item.get('title', '')}" style="font-weight:600;font-size:1rem;line-height:1.35;height:48px;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;margin-bottom:6px;color:#202124;">{item.get('title', 'N/A')}</div></a>
                               <div style="font-size:0.88rem;color:#5f6368;margin-bottom:10px;height:18px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{item.get('channel_name', 'N/A')}</div>
                               <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;"><span style="font-weight:500;color:#3c4043;font-size:0.9rem;">{item.get('formatted_views', 'N/A')} views</span>{badge}</div>
                               <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#5f6368;margin-bottom:12px;"><span>{item.get('vph_display', 'N/A')}</span><span>{days_text}</span></div>
                               <div style="margin-top:auto;">{{/* Button space */}}</div></div>'''
                           st.markdown(card_html, unsafe_allow_html=True)
                           btn_key = f"view_{item['video_id']}"
                           if st.button("View Details", key=btn_key, use_container_width=True):
                                st.session_state.selected_video_id = item["video_id"]
                                st.session_state.selected_video_title = item["title"]
                                st.session_state.selected_video_data = item
                                st.session_state.page = "details"; st.rerun()

def show_details_page():
    """Displays detailed analysis for the selected video."""
    vid = st.session_state.get("selected_video_id")
    vtitle = st.session_state.get("selected_video_title")
    vdata = st.session_state.get("selected_video_data")
    if st.button("⬅️ Back to Search", key="details_back_btn"): st.session_state.page = "search"; st.rerun()
    if not vid or not vtitle or not vdata: st.error("Video data missing."); return
    st.header(f"🔬 Analysis: {vtitle}"); vurl = f"https://www.youtube.com/watch?v={vid}"
    st.caption(f"ID: {vid} | [Watch on YouTube]({vurl})")
    col_t, col_m = st.columns([1, 2])
    with col_t: # Thumbnail & basic info
        thumb = vdata.get('thumbnail', '') or 'https://placehold.co/320x180?text=N/A'; st.image(thumb, use_column_width=True)
        st.caption(f"Channel: **{vdata.get('channel_name', 'N/A')}**")
        dur_s = vdata.get('duration_seconds', 0); dur_m, dur_s_rem = divmod(int(dur_s), 60)
        st.caption(f"Duration: **{dur_m}m {dur_s_rem}s** ({vdata.get('content_category', 'N/A')})")
        st.caption(f"Published: **{vdata.get('publish_date', 'N/A')}** ({int(round(vdata.get('days_since_published', 0)))}d ago)")
    with col_m: # Metrics
        st.subheader("Performance", anchor=False)
        m1,m2,m3 = st.columns(3)
        m1.metric("Views", vdata.get('formatted_views', 'N/A'))
        m2.metric("Likes 👍", format_number(vdata.get('like_count', 0)))
        m3.metric("Comments 💬", format_number(vdata.get('comment_count', 0)))
        score = pd.to_numeric(vdata.get('outlier_score'), errors='coerce'); score_d = f"{score:.2f}x" if score is not None else "N/A"
        delta = f"{(score - 1.0):+.1%}" if score is not None and score != 1.0 else None
        st.metric("Outlier Score", score_d, delta=delta, help="Perf. vs channel avg (1.0=avg)")
        st.markdown("**Ratios (vs Channel Avg):**")
        vph_r = pd.to_numeric(vdata.get('vph_ratio'), errors='coerce'); st.text(f" • VPH: {vph_r:.2f}x" if vph_r else " • VPH: N/A")
        eng_r = pd.to_numeric(vdata.get('engagement_ratio'), errors='coerce'); st.text(f" • Engage: {eng_r:.2f}x" if eng_r else " • Engage: N/A")
        cvr_r = pd.to_numeric(vdata.get('outlier_cvr'), errors='coerce'); st.text(f" • C/V: {cvr_r:.2f}x" if cvr_r else " • C/V: N/A")
        clr_r = pd.to_numeric(vdata.get('outlier_clr'), errors='coerce'); st.text(f" • L/V: {clr_r:.2f}x" if clr_r else " • L/V: N/A")
        st.caption(f"Raw C/V: {vdata.get('comment_to_view_ratio', 'N/A')}, L/V: {vdata.get('like_to_view_ratio', 'N/A')}, Eff. VPH: {vdata.get('vph_display', 'N/A')}")
    st.divider()
    tab_comm, tab_scr, tab_ret = st.tabs(["💬 Comments", "📜 Script", "📈 Retention"])
    with tab_comm: # --- Comments Tab ---
        st.subheader("Comments Analysis", anchor=False); ckey = f"comments_{vid}"; akey = f"analysis_{vid}"
        if ckey not in st.session_state:
            with st.spinner("Fetching comments..."): st.session_state[ckey] = get_video_comments(vid, 100)
        comments = st.session_state.get(ckey, [])
        if not comments: st.info("No comments found/disabled.")
        else:
            st.caption(f"Analysis based on ~{len(comments)} comments.");
            if akey not in st.session_state:
                with st.spinner("Analyzing comments (AI)..."): st.session_state[akey] = analyze_comments(comments)
            st.markdown("**AI Summary:**"); st.markdown(st.session_state.get(akey, "*Failed*"))
            with st.expander("Show Top 5 Comments"):
                 top5 = sorted([c for c in comments if c.get('likeCount',0)>0], key=lambda c: c.get('likeCount',0), reverse=True)[:5]
                 if not top5: st.write("_No comments with likes._")
                 else:
                      for i,c in enumerate(top5): st.markdown(f"**{i+1}. {c.get('likeCount',0)} 👍** *{c.get('author','Anon')}*"); st.caption(f'"{c.get("text","")[:300]}"...'); st.markdown("---") if i<4 else None
    with tab_scr: # --- Script Tab ---
        st.subheader("Script Analysis", anchor=False); dur = vdata.get('duration_seconds', 0); is_short = vdata.get('content_category') == "Short"
        tkey = f"transcript_data_{vid}"; skey = f"transcript_source_{vid}"
        if tkey not in st.session_state:
            with st.spinner("Fetching transcript..."): get_transcript_with_fallback(vid)
        transcript = st.session_state.get(tkey); source = st.session_state.get(skey)
        if not transcript: st.warning("Transcript unavailable.")
        else:
            st.caption(f"Source: {source or 'N/A'}"); full_script = " ".join([s.get("text","") for s in transcript])
            if is_short:
                st.markdown("**Short Summary (AI):**"); sum_key = f"summary_short_{vid}"
                if sum_key not in st.session_state:
                    with st.spinner("Summarizing..."): st.session_state[sum_key] = summarize_script(full_script)
                st.write(st.session_state.get(sum_key, "*Failed*"))
                with st.expander("Full Script"): st.text_area("", full_script, height=200, key="script_area_short")
            else: # Long form
                st.markdown("**Intro/Outro Summary (AI):**"); io_key = f"intro_outro_data_{vid}"
                if io_key not in st.session_state:
                    with st.spinner("Analyzing intro/outro..."):
                        intro, outro = get_intro_outro_transcript(vid, dur); summary, _ = summarize_intro_outro(intro, outro)
                        st.session_state[io_key] = {"intro": intro, "outro": outro, "summary": summary}
                io_data = st.session_state[io_key]; st.markdown(io_data.get("summary", "*Failed*"))
                with st.expander("Raw Intro/Outro"): st.markdown("**Intro:**"); st.caption(io_data.get("intro") or "*N/A*"); st.markdown("**Outro:**"); st.caption(io_data.get("outro") or "*N/A*")
                sum_key_long = f"summary_long_{vid}"
                if st.button("Summarize Full Script (AI)", key="sum_full_btn"):
                    with st.spinner("Summarizing..."): st.session_state[sum_key_long] = summarize_script(full_script); st.rerun()
                if sum_key_long in st.session_state: st.markdown("**Full Summary (AI):**"); st.write(st.session_state[sum_key_long])
                with st.expander("Full Script"): st.text_area("", full_script, height=300, key="script_area_long")
    with tab_ret: # --- Retention Tab ---
        st.subheader("Retention Analysis (Experimental)", anchor=False); prefix = f"retention_{vid}_"
        # Eligibility
        pub_iso = vdata.get("published_at"); can_run = False; msg = ""
        if not SELENIUM_AVAILABLE: msg = "⚠️ Selenium libraries not installed."
        elif is_short: msg = "ℹ️ Less meaningful for Shorts."
        elif not pub_iso: msg = "⚠️ Cannot check age."
        else:
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(pub_iso.replace('Z','+00:00'))
                if age < timedelta(days=2): msg = f"ℹ️ Video too recent ({age.days}d old)."
                elif not os.path.exists(COOKIE_FILE): msg = f"⚠️ `{COOKIE_FILE}` not found (login cookies needed)." # st.markdown("[Cookie Help](...)")
                elif not check_selenium_setup(): msg = "⚠️ Browser/Driver setup failed."
                else: can_run = True
            except Exception as e: msg = f"⚠️ Eligibility check error: {e}"
        if not can_run: st.info(msg)
        else: # Eligible
            running = st.session_state.get(prefix+'running', False); done = st.session_state.get(prefix+'done', False); error = st.session_state.get(prefix+'error')
            if not running and not done:
                if st.button("▶️ Run Retention Analysis", key="run_ret_btn"):
                    for k in list(st.session_state): # Clear previous state
                        if k.startswith(prefix): del st.session_state[k]
                    st.session_state[prefix+'running'] = True; st.rerun()
            if running: # --- Execution Block ---
                st.info("Retention analysis running..."); pb = st.progress(0, "Starting..."); success = False; err_msg = None
                try:
                    pb.progress(10, "Init browser..."); ss_key = prefix+'ss_path'; dur_key = prefix+'duration'; tdir_key = prefix+'temp_dir'
                    temp_dir = tempfile.mkdtemp(prefix=f"ret_{vid}_"); st.session_state[tdir_key] = temp_dir
                    ss_path = os.path.join(temp_dir, "retention.png")
                    pb.progress(25, "Capture graph..."); base_ts = dur if dur>0 else 120
                    vid_dur = capture_player_screenshot_with_hover(vurl, base_ts, ss_path, use_cookies=True)
                    if not os.path.exists(ss_path): raise RuntimeError("Screenshot failed.")
                    st.session_state[ss_key] = ss_path; st.session_state[dur_key] = vid_dur; eff_dur = vid_dur if vid_dur>0 else (dur if dur>0 else 1)
                    pb.progress(50, "Analyze peaks..."); peaks,_,_,roi_w,sums = detect_retention_peaks(ss_path)
                    st.session_state[prefix+'peaks'] = peaks; st.session_state[prefix+'roi_w'] = roi_w; st.session_state[prefix+'col_sums'] = sums
                    pb.progress(70, "Process peaks..."); peak_details = {}
                    if len(peaks)>0 and roi_w>0:
                         if transcript is None and tkey in st.session_state: transcript = st.session_state.get(tkey) # Get transcript if needed
                         for i, px in enumerate(peaks):
                             pt = (px / roi_w) * eff_dur; pid = f"peak_{i+1}"; peak_details[pid] = {'time': pt}
                             fr_path = os.path.join(temp_dir, f"{pid}_frame.png"); capture_frame_at_time(vurl, pt, fr_path, use_cookies=True)
                             if os.path.exists(fr_path): peak_details[pid]['frame'] = fr_path
                             if transcript: peak_details[pid]['transcript'] = filter_transcript(transcript, pt, 5)
                             peak_details[pid]['snippet_start'] = max(0, pt-4); peak_details[pid]['snippet_path'] = os.path.join(temp_dir, f"{pid}.mp4")
                    st.session_state[prefix+'peak_details'] = peak_details; success = True
                    pb.progress(100, "Complete!"); time.sleep(1)
                except Exception as e: err_msg = f"Failed: {e}"; logger.error(f"Retention fail {vid}: {e}",exc_info=True); st.error(err_msg)
                finally: st.session_state[prefix+'running']=False; st.session_state[prefix+'done']=success; st.session_state[prefix+'error']=err_msg; pb.empty(); st.rerun()
            elif done: # --- Display Block ---
                 st.success("Retention analysis done."); ss_path = st.session_state.get(prefix+'ss_path'); peaks = st.session_state.get(prefix+'peaks'); sums = st.session_state.get(prefix+'col_sums'); details = st.session_state.get(prefix+'peak_details', {})
                 if ss_path and os.path.exists(ss_path): st.image(ss_path, caption="Retention Graph")
                 if peaks is not None:
                      st.write(f"{len(peaks)} peak(s) detected."); plot_buf = plot_brightness_profile(sums, peaks)
                      if plot_buf: st.image(plot_buf, caption="Brightness Profile")
                      if details:
                           st.markdown("**Peak Details:**"); snip_dur = st.slider("Snippet Duration (s)", 4, 20, 8, 2, key="ret_snip_dur")
                           for pid, detail in details.items():
                                pt = detail.get('time',0); st.markdown(f"--- \n**{pid.replace('_',' ').title()} ~ {pt:.1f}s**")
                                col_fr, col_inf = st.columns([1,1])
                                with col_fr:
                                    fr = detail.get('frame');
                                    if fr and os.path.exists(fr): st.image(fr, caption=f"Frame {pt:.1f}s")
                                    else: st.caption("_Frame fail_")
                                with col_inf:
                                    st.markdown("**Transcript:**"); st.caption(detail.get('transcript') or "_N/A_")
                                    snip_path = detail.get('snippet_path'); snip_key = f"{prefix}{pid}_snip_dl_{snip_dur}s"
                                    if snip_path:
                                        if st.session_state.get(snip_key) and os.path.exists(snip_path):
                                            try: st.video(snip_path)
                                            except Exception as ve: st.error(f"Video display error: {ve}")
                                        else:
                                            if st.button(f"Load Snippet ({snip_dur}s)", key=f"load_{pid}"):
                                                if check_ytdlp_installed():
                                                    with st.spinner("Downloading..."):
                                                        try: start_t = max(0, pt-snip_dur/2); download_video_snippet(vurl, start_t, snip_dur, snip_path); st.session_state[snip_key]=True; st.rerun()
                                                        except Exception as se: st.error(f"Snippet fail: {se}")
                                                else: st.error("yt-dlp missing.")
                 if st.button("Clear Retention Results", key="clear_ret_btn"):
                     tdir = st.session_state.get(prefix+'temp_dir');
                     if tdir and os.path.isdir(tdir): try: shutil.rmtree(tdir)
                     except Exception as e: logger.error(f"Cleanup error {tdir}: {e}")
                     for k in list(st.session_state):
                          if k.startswith(prefix): del st.session_state[k]
                     st.rerun()
            elif error: # --- Error Display Block ---
                 st.error(f"Analysis failed: {error}")
                 if st.button("Retry Analysis?", key="retry_ret_btn"):
                     for k in list(st.session_state): # Clear state
                          if k.startswith(prefix): del st.session_state[k]
                     st.session_state[prefix+'running'] = True; st.rerun() # Trigger again

def main():
    """Main Streamlit execution function."""
    if 'db_initialized' not in st.session_state:
         # DB init called via @st.cache_resource now, check return value?
         if not DB_CONN_PATH: return # Stop if DB failed to init
         st.session_state.db_initialized = True
    if "page" not in st.session_state: st.session_state.page = "search"
    page = st.session_state.get("page")
    if page == "search": show_search_page()
    elif page == "details": show_details_page()
    else: st.session_state.page = "search"; show_search_page() # Default

def app_shutdown(): logger.info("======== App Shutdown ========")

if __name__ == "__main__":
    st.set_page_config(page_title="YouTube Niche Analysis", layout="wide", page_icon="📊")
    # Check essential libs early
    if not YOUTUBE_LIBS_AVAILABLE: st.stop()
    try: main()
    except Exception as e: logger.critical(f"FATAL: Unhandled exception: {e}", exc_info=True); st.error(f"Critical Error: {e}")
    finally: atexit.register(app_shutdown)

# --- END OF FILE ---
