
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
import openai
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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
# Import specific exceptions
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import imageio_ffmpeg
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# =============================================================================
# 1. Logging Setup
# =============================================================================
def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_file = "logs/youtube_finance_search.log"
    # Reduced log size for potentially faster rotation/less disk usage if needed
    file_handler = RotatingFileHandler(log_file, maxBytes=512*1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")) # More detailed format
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger = logging.getLogger()
    # Set level to INFO for more visibility during development/debugging if needed
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.WARNING) # Keep WARNING for production
    # Prevent adding handlers multiple times if logger is already configured
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

# =============================================================================
# 2. API Keys from Streamlit Secrets (single key, no proxies)
# =============================================================================
try:
    # Prioritize YOUTUBE_API_KEY as used in app(1).py
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]["key"]
    logger.info("Using YOUTUBE_API_KEY.")
except KeyError:
    try:
        # Fallback to YT_API_KEY if YOUTUBE_API_KEY isn't found (from app.py)
        YOUTUBE_API_KEY = st.secrets["YT_API_KEY"]
        logger.info("Using YT_API_KEY as fallback.")
    except KeyError:
        st.error("❗ No YouTube API key (YOUTUBE_API_KEY or YT_API_KEY) found in Streamlit secrets!")
        logger.critical("CRITICAL: YouTube API key missing in secrets.")
        YOUTUBE_API_KEY = None
except Exception as e:
    st.error(f"🚨 Error loading YouTube API key: {e}")
    logger.error(f"Error loading YouTube API key from secrets: {e}", exc_info=True)
    YOUTUBE_API_KEY = None


try:
    # Assuming OPENAI_API_KEY is structured like YOUTUBE_API_KEY
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]["key"]
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key loaded successfully.")
except KeyError:
     st.error("❗ OpenAI API Key (OPENAI_API_KEY) not found or structured incorrectly in Streamlit secrets!")
     logger.error("OpenAI API key missing or misconfigured in secrets.")
     OPENAI_API_KEY = None # Ensure it's None if not found
except Exception as e:
    logger.error(f"Failed to load OpenAI API key: {str(e)}", exc_info=True)
    st.error("Failed to load OpenAI API key. Please check your secrets configuration.")
    OPENAI_API_KEY = None

def get_youtube_api_key():
    if not YOUTUBE_API_KEY:
        logger.error("get_youtube_api_key called but key is missing.")
        raise ValueError("YouTube API key is not configured.") # Use ValueError for missing config
    return YOUTUBE_API_KEY

# =============================================================================
# 3. SQLite DB Setup (Caching)
# =============================================================================
DB_PATH = "cache.db"

def init_db(db_path=DB_PATH):
    try:
        with sqlite3.connect(db_path, timeout=10) as conn: # Added timeout
            conn.execute("""
            CREATE TABLE IF NOT EXISTS youtube_cache (
                cache_key TEXT PRIMARY KEY,
                json_data TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
            """)
            # Optional: Index for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON youtube_cache(cache_key);")
            logger.info(f"Database initialized successfully at {db_path}")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database at {db_path}: {e}", exc_info=True)
        st.error(f"Database initialization failed: {e}")


def get_cached_result(cache_key, ttl=600, db_path=DB_PATH):
    now = time.time()
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT json_data, timestamp FROM youtube_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
        if row:
            json_data, cached_time = row
            if (now - cached_time) < ttl:
                logger.debug(f"Cache hit for key {cache_key[-8:]}")
                return json.loads(json_data)
            else:
                logger.debug(f"Cache expired for key {cache_key[-8:]}")
                # Optionally delete expired key here or let set_cached_result overwrite
                delete_cache_key(cache_key, db_path) # Explicitly delete expired
    except sqlite3.OperationalError as e:
         # Handle potential DB locking issues more gracefully
         if "database is locked" in str(e).lower():
              logger.warning(f"Database locked while getting cache for {cache_key[-8:]}. Cache miss.")
         else:
              logger.error(f"get_cached_result DB error for key {cache_key[-8:]}: {e}", exc_info=True)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode cached JSON for key {cache_key[-8:]}: {e}. Deleting invalid cache entry.")
        delete_cache_key(cache_key, db_path) # Delete corrupted entry
    except Exception as e:
        logger.error(f"Unexpected error in get_cached_result for key {cache_key[-8:]}: {e}", exc_info=True)
    return None

def set_cached_result(cache_key, data_obj, db_path=DB_PATH):
    now = time.time()
    try:
        # Ensure data is JSON serializable before connecting to DB
        json_str = json.dumps(data_obj, default=str)
    except TypeError as e:
        logger.error(f"Data for cache key {cache_key[-8:]} is not JSON serializable: {e}")
        return # Do not attempt to cache non-serializable data

    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute("INSERT OR REPLACE INTO youtube_cache (cache_key, json_data, timestamp) VALUES (?, ?, ?)",
                         (cache_key, json_str, now))
            logger.debug(f"Cache set for key {cache_key[-8:]}")
    except sqlite3.OperationalError as e:
         if "database is locked" in str(e).lower():
              logger.warning(f"Database locked while setting cache for {cache_key[-8:]}. Cache not updated.")
         else:
              logger.error(f"set_cached_result DB error for key {cache_key[-8:]}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in set_cached_result for key {cache_key[-8:]}: {e}", exc_info=True)


def delete_cache_key(cache_key, db_path=DB_PATH):
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute("DELETE FROM youtube_cache WHERE cache_key = ?", (cache_key,))
            logger.debug(f"Cache deleted for key {cache_key[-8:]}")
    except sqlite3.OperationalError as e:
         if "database is locked" in str(e).lower():
              logger.warning(f"Database locked while deleting cache for {cache_key[-8:]}. Deletion might fail.")
         else:
              logger.error(f"delete_cache_key DB error for key {cache_key[-8:]}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in delete_cache_key for key {cache_key[-8:]}: {e}", exc_info=True)

# =============================================================================
# 4. Utility Helpers
# =============================================================================
def format_date(date_string):
    """Formats ISO date string to DD-MM-YY."""
    if not date_string or not isinstance(date_string, str):
        return "Unknown"
    try:
        # Handle potential milliseconds or 'Z'
        date_string_cleaned = date_string.split('.')[0] + 'Z' if '.' in date_string else date_string
        date_obj = datetime.strptime(date_string_cleaned, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return date_obj.strftime("%d-%m-%y")
    except ValueError:
        logger.warning(f"Could not parse date string: {date_string}", exc_info=False) # Log less severe errors quietly
        return "Unknown"
    except Exception as e:
        logger.error(f"Unexpected error formatting date '{date_string}': {e}", exc_info=True)
        return "Error"


def format_number(num):
    """Formats numbers into K/M format, handles potential errors."""
    try:
        n = int(num)
        if abs(n) >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        elif abs(n) >= 1_000:
            return f"{n/1_000:.1f}K"
        return str(n)
    except (ValueError, TypeError):
        # logger.debug(f"Could not format number: {num}") # Debug if needed
        return str(num) # Return original string if conversion fails
    except Exception as e:
         logger.error(f"Unexpected error formatting number {num}: {e}")
         return "Error"


def build_cache_key(*args):
    """Builds a SHA256 hash cache key from input arguments."""
    try:
        # Convert all args to string and handle None values
        raw_str = "-".join(str(a) if a is not None else 'None' for a in args)
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    except Exception as e:
         logger.error(f"Failed to build cache key from args: {args} - Error: {e}", exc_info=True)
         # Fallback to a simple hash of the string representation? Risky. Best to raise or return fixed key.
         raise ValueError("Failed to build cache key") from e


def parse_iso8601_duration(duration_str):
    """Parse ISO 8601 duration format (PTnHnMnS) to seconds."""
    if not duration_str or duration_str == 'P0D': # Handle zero duration or empty
        return 0
    try:
        duration = isodate.parse_duration(duration_str)
        return int(duration.total_seconds())
    except isodate.ISO8601Error:
        logger.warning(f"Could not parse ISO 8601 duration: {duration_str}")
        return 0 # Return 0 if parsing fails
    except Exception as e:
         logger.error(f"Unexpected error parsing duration {duration_str}: {e}", exc_info=True)
         return 0

# =============================================================================
# 5. Channel Folders
# =============================================================================
CHANNELS_FILE = "channels.json" # Might not be used if loading directly into folders
FOLDERS_FILE = "channel_folders.json"

# Removed load_channels_json as it seemed unused, simplifying folder init
# channels_data_json = load_channels_json()

def load_channel_folders():
    """Loads channel folders from JSON file, creates default if not found."""
    if os.path.exists(FOLDERS_FILE):
        try:
            with open(FOLDERS_FILE, "r", encoding="utf-8") as f:
                folders = json.load(f)
                # Basic validation: ensure it's a dictionary
                if isinstance(folders, dict):
                     logger.info(f"Loaded {len(folders)} channel folders from {FOLDERS_FILE}")
                     return folders
                else:
                     logger.warning(f"{FOLDERS_FILE} does not contain a valid dictionary. Re-creating.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {FOLDERS_FILE}: {e}. Re-creating file.")
        except Exception as e:
            logger.error(f"Error loading {FOLDERS_FILE}: {e}. Re-creating file.", exc_info=True)

    # Create default folders if file doesn't exist or is invalid
    logger.info(f"{FOLDERS_FILE} not found or invalid, creating default folders.")
    default_folders = {
        "USA Finance Niche (Example)": [
            # Add a couple of well-known example Channel IDs or Names if desired
             {"channel_name": "Example Finance Channel US", "channel_id": "UC_example_finance_us"}
        ],
        "India Finance Niche (Example)": [
             {"channel_name": "Example Finance Channel IN", "channel_id": "UC_example_finance_in"}
        ],
        "My Channels": [] # Add an empty folder for users to start with
    }
    # Attempt to save the defaults
    save_channel_folders(default_folders)
    return default_folders


def save_channel_folders(folders):
    """Saves the channel folders dictionary to a JSON file."""
    try:
        with open(FOLDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(folders, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(folders)} channel folders to {FOLDERS_FILE}")
    except TypeError as e:
        logger.error(f"Failed to serialize folders data to JSON: {e}", exc_info=True)
        st.error(f"Error saving folder data: Data couldn't be serialized. {e}")
    except IOError as e:
        logger.error(f"Failed to write channel folders to {FOLDERS_FILE}: {e}", exc_info=True)
        st.error(f"Error saving folder data: Could not write to file. {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving channel folders: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while saving folders: {e}")


def get_channel_id(channel_name_or_url):
    """Resolves various YouTube channel inputs (URL, name, handle, ID) to a channel ID."""
    if not channel_name_or_url or not isinstance(channel_name_or_url, str):
        return None
    identifier = channel_name_or_url.strip()
    if not identifier:
        return None

    # 1. Check if it's a direct Channel ID
    if identifier.startswith("UC") and len(identifier) == 24 and re.match(r'^UC[A-Za-z0-9_\-]{22}$', identifier):
        logger.debug(f"Input '{identifier}' appears to be a direct Channel ID.")
        return identifier

    # 2. Check common URL patterns
    patterns = {
        'channel_id': r'youtube\.com/channel/([^/\s?]+)',
        'handle': r'youtube\.com/@([^/\s?]+)',
        'custom_url': r'youtube\.com/c/([^/\s?]+)',
        'user_url': r'youtube\.com/user/([^/\s?]+)',
    }
    resolved_id = None
    identifier_to_resolve = identifier # Default to original input
    resolve_type = 'search' # Default resolution method

    for id_type, pattern in patterns.items():
        match = re.search(pattern, identifier, re.IGNORECASE)
        if match:
            extracted_part = match.group(1)
            logger.debug(f"Matched URL pattern '{id_type}' with value '{extracted_part}'.")
            if id_type == 'channel_id' and extracted_part.startswith('UC'):
                return extracted_part # Direct ID from URL
            else:
                # Need to resolve handle, custom URL, or user URL via API
                identifier_to_resolve = extracted_part
                resolve_type = id_type
                break # Stop after first match

    logger.debug(f"Attempting to resolve '{identifier_to_resolve}' using method '{resolve_type}'.")

    # 3. Resolve using YouTube API
    try:
        key = get_youtube_api_key()
    except ValueError as e:
        st.error(f"Cannot resolve channel identifier: {e}")
        return None

    api_url = None
    params = {'key': key, 'part': 'id'} # Always need ID part

    try:
        # Prioritize direct lookups if possible
        if resolve_type == 'user_url':
            # Try 'forUsername' endpoint (less reliable now)
             params['forUsername'] = identifier_to_resolve
             api_url = "https://www.googleapis.com/youtube/v3/channels"
             response = requests.get(api_url, params=params, timeout=10)
             response.raise_for_status()
             data = response.json()
             if data.get("items"):
                 resolved_id = data["items"][0]["id"]
                 logger.info(f"Resolved username '{identifier_to_resolve}' to ID: {resolved_id}")
                 return resolved_id
             else:
                 # If forUsername fails, fall back to search
                 logger.debug(f"'forUsername={identifier_to_resolve}' failed, falling back to search.")
                 resolve_type = 'search' # Ensure search happens next
                 identifier_to_resolve = identifier # Reset to original input for search? Or use extracted part? Use original.


        # Use Search API for handle, custom URL, or general search
        if resolve_type in ['handle', 'custom_url', 'search']:
             # Prepare search query (use original identifier for search unless it was handle)
             search_query = identifier_to_resolve if resolve_type != 'handle' else identifier # Use original if not handle
             if resolve_type == 'handle': search_query = '@' + identifier_to_resolve # Search with @ for handles? Or without? YT search handles both.
             params['part'] = 'snippet' # Need snippet for matching title/handle in search results
             params['type'] = 'channel'
             params['q'] = search_query
             params['maxResults'] = 5 # Get a few results to find the best match
             api_url = "https://www.googleapis.com/youtube/v3/search"
             response = requests.get(api_url, params=params, timeout=10)
             response.raise_for_status()
             data = response.json()

             if data.get("items"):
                 # Try to find the best match based on title, custom URL, or handle
                 target_lower = identifier_to_resolve.lower() # Use the part extracted or original for matching
                 best_match_id = None
                 for item in data["items"]:
                     snippet = item.get('snippet', {})
                     channel_id = item.get('id', {}).get('channelId')
                     title = snippet.get('title', '').lower()
                     custom_url = snippet.get('customUrl', '').lower() # May not exist
                     # Handle needs to be checked against target_lower carefully
                     # if resolve_type == 'handle' and handle_from_snippet == target_lower...

                     if channel_id:
                         # Prioritize exact match on customURL or title if types match expectation
                         if (resolve_type == 'custom_url' and custom_url == target_lower) or \
                            (resolve_type == 'handle' and title == identifier_to_resolve) or \
                            (resolve_type == 'search' and target_lower in title): # Simple containment for search
                             best_match_id = channel_id
                             logger.info(f"Found likely match for '{search_query}' via search: ID {best_match_id} (Match type: {resolve_type})")
                             break # Found a strong match
                         elif not best_match_id: # Keep first result as fallback
                              best_match_id = channel_id

                 if best_match_id:
                      logger.info(f"Resolved '{search_query}' via search to ID: {best_match_id}")
                      return best_match_id
                 else:
                      logger.warning(f"Search for '{search_query}' returned items but no suitable channel ID found.")
                      return None
             else:
                  logger.warning(f"API search for '{search_query}' returned no items.")
                  return None

    except requests.exceptions.HTTPError as e:
        # Handle specific API errors (quota, invalid key etc.)
        status_code = e.response.status_code
        error_text = e.response.text
        if status_code == 403:
            logger.error(f"API Key/Quota Error resolving '{identifier_to_resolve}': {error_text}")
            st.error(f"YouTube API Error (Forbidden/Quota?). Check key and usage.")
        elif status_code == 400:
            logger.error(f"API Bad Request resolving '{identifier_to_resolve}': {error_text}")
            st.error(f"YouTube API Error (Bad Request): {error_text}")
        else:
            logger.error(f"HTTP Error resolving '{identifier_to_resolve}': Status {status_code}, Response: {error_text}")
            st.error(f"HTTP error {status_code} resolving channel.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error resolving channel '{identifier_to_resolve}': {e}", exc_info=True)
        st.error(f"Network error: Could not connect to YouTube API.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error resolving channel '{identifier_to_resolve}': {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
        return None

    logger.warning(f"Could not resolve channel identifier after all attempts: {channel_name_or_url}")
    return None # Fallback if nothing resolved

# --- Rest of Folder Management UI ---
def show_channel_folder_manager():
    # Use columns for better layout if needed
    st.write("### Manage Channel Folders")
    folders = load_channel_folders()
    folder_keys = list(folders.keys())

    action = st.selectbox("Action", ["Create New Folder", "Modify Folder", "Delete Folder"], key="folder_action_select")

    if action == "Create New Folder":
        with st.form("create_folder_form"):
            folder_name = st.text_input("New Folder Name", key="new_folder_name_input").strip()
            st.write("Enter Channel names, URLs, or IDs (one per line):")
            channels_text = st.text_area("Channels", height=100, key="new_folder_channels_input")
            submitted = st.form_submit_button("Create Folder")

            if submitted:
                if not folder_name:
                    st.error("Folder name cannot be empty.")
                    return
                if folder_name in folders:
                    st.error(f"Folder '{folder_name}' already exists.")
                    return

                lines = channels_text.strip().split("\n")
                channel_list = []
                processed_ids = set() # Avoid duplicates

                with st.spinner("Resolving channel IDs..."):
                    for line in lines:
                        line = line.strip()
                        if not line: continue
                        ch_id = get_channel_id(line)
                        if ch_id and ch_id not in processed_ids:
                             # Fetch actual channel name for better display
                             ch_name = line # Default to input
                             try:
                                 ch_details_url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={ch_id}&key={get_youtube_api_key()}"
                                 ch_details_res = requests.get(ch_details_url, timeout=5).json()
                                 if ch_details_res.get('items'):
                                     ch_name = ch_details_res['items'][0]['snippet']['title']
                             except Exception as e:
                                 logger.warning(f"Could not fetch channel name for {ch_id}: {e}")

                             channel_list.append({"channel_name": ch_name, "channel_id": ch_id})
                             processed_ids.add(ch_id)
                             logger.debug(f"Resolved '{line}' to {ch_name} ({ch_id})")
                        elif ch_id and ch_id in processed_ids:
                             logger.debug(f"Skipping duplicate channel ID {ch_id} from input '{line}'")
                        elif not ch_id:
                             st.warning(f"Could not resolve '{line}' to a channel ID. Skipping.")

                if not channel_list:
                    st.error("No valid channels were resolved. Folder not created.")
                    return

                folders[folder_name] = channel_list
                save_channel_folders(folders)
                st.success(f"Folder '{folder_name}' created with {len(channel_list)} channel(s).")
                st.balloons() # Fun confirmation

                # Trigger pre-caching (optional, can take time)
                # Consider making this a separate button or background task?
                with st.spinner("Pre-caching recent data for new channels (this might take a moment)..."):
                     for ch in channel_list:
                          try:
                              # Use a short TTL for pre-cache to avoid stale data if user searches soon
                              search_youtube("", [ch["channel_id"]], "3 months", "Both", ttl=3600) # 1 hour TTL
                          except Exception as e:
                              st.warning(f"Could not pre-cache data for {ch['channel_name']}: {e}")
                st.rerun() # Rerun to update UI


    elif action == "Modify Folder":
        if not folder_keys:
            st.info("No folders available to modify.")
            return

        selected_folder = st.selectbox("Select Folder to Modify", folder_keys, key="modify_folder_select")

        if selected_folder and selected_folder in folders:
            st.write(f"**Modifying Folder:** {selected_folder}")
            current_channels = folders.get(selected_folder, [])
            current_channel_ids = {ch['channel_id'] for ch in current_channels}

            with st.expander("Current Channels in Folder", expanded=True):
                 if current_channels:
                      df_channels = pd.DataFrame(current_channels)
                      st.dataframe(df_channels[['channel_name', 'channel_id']], use_container_width=True, hide_index=True)
                 else:
                      st.write("(This folder is empty)")

            modify_action = st.radio("Modification Type", ["Add Channels", "Remove Channel"], key="modify_type_radio", horizontal=True)

            if modify_action == "Add Channels":
                 with st.form("add_channels_form"):
                      st.write("Enter new Channel names, URLs, or IDs (one per line):")
                      new_ch_text = st.text_area("Channels to Add", height=100, key="add_channels_input")
                      add_submitted = st.form_submit_button("Add to Folder")

                      if add_submitted:
                          lines = new_ch_text.strip().split("\n")
                          added_channels_data = []
                          with st.spinner("Resolving and adding channels..."):
                              for line in lines:
                                  line = line.strip()
                                  if not line: continue
                                  ch_id = get_channel_id(line)
                                  if ch_id and ch_id not in current_channel_ids:
                                       ch_name = line # Default
                                       try:
                                           ch_details_url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={ch_id}&key={get_youtube_api_key()}"
                                           ch_details_res = requests.get(ch_details_url, timeout=5).json()
                                           if ch_details_res.get('items'):
                                               ch_name = ch_details_res['items'][0]['snippet']['title']
                                       except Exception as e: logger.warning(f"Could not fetch channel name for {ch_id}: {e}")

                                       new_channel = {"channel_name": ch_name, "channel_id": ch_id}
                                       folders[selected_folder].append(new_channel)
                                       added_channels_data.append(new_channel)
                                       current_channel_ids.add(ch_id) # Update live set
                                       logger.debug(f"Resolved '{line}' to add: {ch_name} ({ch_id})")
                                  elif ch_id and ch_id in current_channel_ids:
                                       st.warning(f"'{line}' ({ch_id}) is already in the folder. Skipping.")
                                  elif not ch_id:
                                       st.warning(f"Could not resolve '{line}'. Skipping.")

                          if added_channels_data:
                              save_channel_folders(folders)
                              st.success(f"Added {len(added_channels_data)} channel(s) to '{selected_folder}'.")
                              # Optional: Pre-cache for added channels
                              with st.spinner("Pre-caching data for newly added channels..."):
                                  for ch_data in added_channels_data:
                                      try: search_youtube("", [ch_data["channel_id"]], "3 months", "Both", ttl=3600)
                                      except Exception as e: st.warning(f"Could not pre-cache for {ch_data['channel_name']}: {e}")
                              st.rerun()
                          else:
                              st.info("No new valid channels were added.")


            elif modify_action == "Remove Channel":
                 if current_channels:
                      # Create display options: "Channel Name (Channel ID)"
                      channel_options = {f"{ch['channel_name']} ({ch['channel_id']})": ch['channel_id'] for ch in current_channels}
                      remove_choice_display = st.selectbox("Select channel to remove", list(channel_options.keys()), key="remove_channel_select")

                      if st.button("Remove Selected Channel", key="remove_channel_button", type="secondary"):
                          remove_choice_id = channel_options.get(remove_choice_display)
                          if remove_choice_id:
                              original_count = len(folders[selected_folder])
                              folders[selected_folder] = [c for c in folders[selected_folder] if c["channel_id"] != remove_choice_id]
                              if len(folders[selected_folder]) < original_count:
                                  save_channel_folders(folders)
                                  st.success(f"Removed '{remove_choice_display}' from '{selected_folder}'.")
                                  st.rerun()
                              else:
                                  st.error("Internal error: Failed to remove the channel.") # Should not happen
                          else:
                              st.error("Invalid selection or channel not found.") # Should not happen with selectbox
                 else:
                      st.info("No channels in this folder to remove.")


    elif action == "Delete Folder":
        if not folder_keys:
            st.info("No folders available to delete.")
            return

        selected_folder_to_delete = st.selectbox("Select Folder to Delete", folder_keys, key="delete_folder_select")

        if selected_folder_to_delete and selected_folder_to_delete in folders:
             st.warning(f"⚠️ Are you sure you want to permanently delete the folder '{selected_folder_to_delete}' and all its channels? This cannot be undone.")
             if st.button(f"Confirm Delete '{selected_folder_to_delete}'", key="delete_folder_confirm_button", type="primary"):
                 try:
                      del folders[selected_folder_to_delete]
                      save_channel_folders(folders)
                      st.success(f"Folder '{selected_folder_to_delete}' deleted.")
                      logger.info(f"Deleted folder: {selected_folder_to_delete}")
                      st.rerun() # Update UI
                 except KeyError:
                      st.error("Folder not found (might have been deleted already).")
                 except Exception as e:
                      st.error(f"Error deleting folder: {e}")
                      logger.error(f"Error deleting folder {selected_folder_to_delete}: {e}", exc_info=True)


# =============================================================================
# 6. Transcript & Fallback
# =============================================================================
def get_transcript(video_id):
    """Fetches YouTube transcript, trying English then any available generated transcript."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try common English variants first
        try:
            transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
            logger.debug(f"Found English generated transcript for {video_id}")
            return transcript.fetch()
        except NoTranscriptFound:
            logger.debug(f"No English generated transcript for {video_id}. Checking other languages.")
            # If English fails, try finding *any* generated transcript
            try:
                # Get list of available generated languages
                generated_languages = [t.language_code for t in transcript_list if t.is_generated]
                if generated_languages:
                     transcript = transcript_list.find_generated_transcript(generated_languages)
                     logger.info(f"Found generated transcript in '{transcript.language}' for {video_id}")
                     return transcript.fetch()
                else:
                     logger.debug(f"No generated transcripts found for {video_id}. Checking manual.")
                     # If still no generated, try finding *any* manual transcript (less likely useful)
                     try:
                         manual_languages = [t.language_code for t in transcript_list if not t.is_generated]
                         if manual_languages:
                              transcript = transcript_list.find_manually_created_transcript(manual_languages)
                              logger.info(f"Found manual transcript in '{transcript.language}' for {video_id}")
                              return transcript.fetch()
                         else:
                              logger.warning(f"No manual transcripts found for {video_id} either.")
                              return None # No transcripts at all
                     except NoTranscriptFound: # Should not happen if manual_languages is populated, but just in case
                          logger.warning(f"Error finding manual transcript despite listing availability for {video_id}.")
                          return None
            except NoTranscriptFound:
                logger.warning(f"No generated transcript found for {video_id} in any language.")
                return None # No generated transcripts at all

    except TranscriptsDisabled:
        logger.warning(f"Transcripts are disabled for video {video_id}")
        return None
    except Exception as e:
        # Catch broader errors like network issues, API changes etc.
        logger.error(f"Error fetching transcript list or content for {video_id}: {e}", exc_info=True)
        return None


def download_audio(video_id):
    """Downloads audio using yt-dlp, returns path and temp directory used."""
    # Check dependencies
    if not shutil.which("ffmpeg"):
        st.error("ffmpeg not found. Please install ffmpeg and add it to your system's PATH.")
        logger.error("ffmpeg executable not found in PATH.")
        return None, None
    if not check_ytdlp_installed():
         st.error("yt-dlp not found. Please install yt-dlp (`pip install yt-dlp`).")
         logger.error("yt-dlp command not found.")
         return None, None

    # Create a unique temporary directory for this download
    try:
         temp_dir = tempfile.mkdtemp(prefix=f"audio_dl_{video_id}_")
         logger.info(f"Created temporary directory for audio download: {temp_dir}")
    except Exception as e:
         logger.error(f"Failed to create temporary directory: {e}")
         st.error("Failed to create temporary directory for download.")
         return None, None


    safe_video_id = re.sub(r'[^\w-]', '', video_id) # Sanitize ID for filename
    output_template = os.path.join(temp_dir, f"{safe_video_id}.%(ext)s")
    # Define expected output path AFTER potential conversion by yt-dlp
    audio_path_expected = os.path.join(temp_dir, f"{safe_video_id}.mp3")
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best', # Prefer m4a if available, else best audio
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128', # 128kbps good balance for speech
        }],
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(), # Ensure ffmpeg path is known
        'socket_timeout': 45, # Timeout for network operations (seconds)
        # Add option to ignore configuration files which might cause issues
        'ignoreconfig': True,
        # Limit download size? Risky, might cut off audio. Prefer handling large files in Whisper func.
        # 'max_filesize': '50M', # Example: limit to 50MB (might fail transcription)
    }

    try:
        import yt_dlp
        logger.info(f"Attempting audio download for {video_id} using yt-dlp...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Check if the explicitly requested MP3 file exists
        if os.path.exists(audio_path_expected):
            logger.info(f"Audio downloaded and converted successfully: {audio_path_expected}")
            return audio_path_expected, temp_dir
        else:
            # If mp3 doesn't exist, maybe conversion failed or download used different ext?
            # yt-dlp usually cleans up intermediates, so check if *any* audio file remains
            found_audio_files = [f for f in os.listdir(temp_dir) if f.startswith(safe_video_id) and f.split('.')[-1] in ['m4a', 'opus', 'ogg', 'wav', 'aac']]
            if found_audio_files:
                 # If an intermediate (like m4a) exists, it means conversion to mp3 might have failed
                 # We could try using this file directly with Whisper if supported, or log error
                 remaining_file = os.path.join(temp_dir, found_audio_files[0])
                 logger.warning(f"Expected mp3 not found, but found other audio file: {remaining_file}. Conversion might have failed. Attempting to use it.")
                 # Whisper generally handles m4a, opus etc. Return this file.
                 return remaining_file, temp_dir
            else:
                 # No mp3 and no other audio files found.
                 logger.error(f"yt-dlp download finished but no audio file found in {temp_dir} for {video_id}")
                 st.error("Audio download failed: No output file created.")
                 shutil.rmtree(temp_dir) # Clean up the failed attempt's directory
                 return None, None

    except yt_dlp.utils.DownloadError as e:
         # Provide more specific feedback if possible
         err_str = str(e).lower()
         if 'video unavailable' in err_str:
              st.error("Audio download failed: The video is unavailable.")
         elif 'private video' in err_str:
              st.error("Audio download failed: The video is private.")
         elif 'region-locked' in err_str:
              st.error("Audio download failed: The video is region-locked.")
         else:
              st.error(f"Audio download error: {e}")
         logger.error(f"yt-dlp DownloadError for {video_id}: {e}", exc_info=False) # Don't need full traceback usually
         shutil.rmtree(temp_dir) # Clean up
         return None, None
    except Exception as e:
        logger.error(f"Unexpected error during audio download process for {video_id}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during audio download: {e}")
        # Clean up the temp directory in case of any error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None, None


def generate_transcript_with_openai(audio_file):
    """Generates transcript using OpenAI Whisper, handling large files by snippeting."""
    if not OPENAI_API_KEY:
         st.error("OpenAI API Key not configured. Cannot use Whisper transcription fallback.")
         return None, None

    max_size_bytes = 25 * 1024 * 1024 # OpenAI API limit (26,214,400 bytes), use slightly less
    try:
        actual_size = os.path.getsize(audio_file)
        logger.info(f"Audio file size for Whisper: {actual_size / (1024*1024):.2f} MB")
    except OSError as e:
        st.error(f"Error accessing audio file for transcription: {e}")
        logger.error(f"Could not get size of audio file {audio_file}: {e}")
        return None, None

    file_to_transcribe = audio_file
    temp_snippet_file = None
    cleanup_snippet = False # Flag to ensure snippet is removed
    snippet_duration_sec = 0 # Keep track of snippet duration if created

    if actual_size > max_size_bytes:
        st.warning(f"Audio file is large ({actual_size / (1024*1024):.1f}MB). Transcribing only the beginning (approx first ~15-20 mins).")
        logger.warning(f"Audio file {audio_file} size {actual_size} exceeds Whisper limit {max_size_bytes}. Creating snippet.")

        # Create a temporary snippet file path in the same directory
        base, ext = os.path.splitext(audio_file)
        temp_snippet_file = f"{base}_snippet{ext}"

        # Estimate max duration based on size and assumed bitrate (e.g., 128kbps = 16KB/s)
        # Max duration ~ max_size_bytes / (16 * 1024)
        # Use a margin of safety (e.g., 90%)
        estimated_max_duration = int((max_size_bytes * 0.9) / (16 * 1024))
        # Also cap the snippet duration to something reasonable, e.g., 20 minutes max
        snippet_duration_sec = min(estimated_max_duration, 20 * 60)
        logger.info(f"Creating audio snippet of max duration: {snippet_duration_sec} seconds")

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        # Command to create snippet using stream copy if possible (fast), otherwise re-encode
        cmd = [
            ffmpeg_exe, "-y", # Overwrite if exists
            "-i", audio_file,
            "-t", str(snippet_duration_sec), # Duration limit
            "-c", "copy", # Attempt codec copy (fastest)
            temp_snippet_file
        ]
        try:
            logger.info(f"Running ffmpeg (copy): {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, timeout=60) # Add timeout
            # Check if codec copy worked (file exists and > 0)
            if not os.path.exists(temp_snippet_file) or os.path.getsize(temp_snippet_file) == 0:
                 raise subprocess.CalledProcessError(1, cmd, stderr=b"ffmpeg copy likely failed, file not created/empty")
            logger.info(f"Created snippet using codec copy: {temp_snippet_file}")
            file_to_transcribe = temp_snippet_file
            cleanup_snippet = True # Mark for cleanup

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"ffmpeg snippet creation with 'copy' failed: {e}. Retrying with re-encoding.")
            # Fallback: Re-encode snippet (slower, but more robust)
            cmd = [
                 ffmpeg_exe, "-y",
                 "-i", audio_file,
                 "-t", str(snippet_duration_sec),
                 # Specify audio codec explicitly (e.g., libmp3lame or aac)
                 "-c:a", "libmp3lame", "-b:a", "128k", # Re-encode to MP3 128kbps
                 temp_snippet_file
            ]
            try:
                 logger.info(f"Running ffmpeg (re-encode): {' '.join(cmd)}")
                 subprocess.run(cmd, check=True, capture_output=True, timeout=180) # Longer timeout for re-encode
                 logger.info(f"Created snippet using re-encoding: {temp_snippet_file}")
                 file_to_transcribe = temp_snippet_file
                 cleanup_snippet = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e_reencode:
                 logger.error(f"ffmpeg snippet re-encoding also failed: {getattr(e_reencode, 'stderr', b'Timeout or Error').decode(errors='ignore')}")
                 st.error("Failed to create a smaller audio snippet for transcription.")
                 return None, None # Abort transcription
            except Exception as e_reencode_unexp:
                 logger.error(f"Unexpected error during snippet re-encoding: {e_reencode_unexp}")
                 st.error(f"Unexpected error creating audio snippet: {e_reencode_unexp}")
                 return None, None


        # Final check on snippet size after creation attempt
        if file_to_transcribe == temp_snippet_file:
            try:
                 snippet_size = os.path.getsize(file_to_transcribe)
                 if snippet_size > max_size_bytes:
                      logger.error(f"Created snippet size {snippet_size} still exceeds limit {max_size_bytes}. Aborting.")
                      st.error("Failed to create a small enough audio snippet (<25MB).")
                      if cleanup_snippet and os.path.exists(temp_snippet_file): os.remove(temp_snippet_file)
                      return None, None
            except OSError as e:
                 logger.error(f"Could not get size of created snippet {file_to_transcribe}: {e}")
                 st.error("Error checking snippet file size.")
                 if cleanup_snippet and os.path.exists(temp_snippet_file): os.remove(temp_snippet_file)
                 return None, None


    transcription_success = False
    try:
        logger.info(f"Transcribing {os.path.basename(file_to_transcribe)} using OpenAI Whisper...")
        with open(file_to_transcribe, "rb") as file_handle:
            # Use the appropriate OpenAI client syntax
            try:
                 from openai import OpenAI, APIError
                 client = OpenAI(api_key=OPENAI_API_KEY)
                 transcript_response = client.audio.transcriptions.create(
                      model="whisper-1",
                      file=file_handle
                      # Add language parameter? Might improve accuracy if known.
                      # language="en" # Example
                 )
                 text = transcript_response.text
            except ImportError:
                 # Fallback to older openai<1.0 syntax
                 logger.debug("Using legacy OpenAI < 1.0 syntax for Whisper.")
                 # Need to explicitly import APIError from older structure if used in except block
                 from openai.error import APIError as OldAPIError
                 global APIError # Make it accessible in except block
                 APIError = OldAPIError # Assign for uniform handling below

                 transcript_response = openai.Audio.transcribe(
                      model="whisper-1",
                      file=file_handle,
                      api_key=OPENAI_API_KEY
                 )
                 text = transcript_response["text"] # type: ignore

        logger.info("Whisper transcription successful.")
        transcription_success = True

        # --- Process Transcript Text ---
        # Whisper model 'whisper-1' usually returns only text, not segments/timestamps.
        # Return a single segment containing the full text. Estimate duration if needed.
        # Use snippet duration if created, else estimate from original size
        duration_estimate = snippet_duration_sec if snippet_duration_sec > 0 else (actual_size / (16*1024) if actual_size > 0 else 0)

        segments = [{
             "start": 0.0,
             "duration": duration_estimate, # Approximate duration
             "text": text or "" # Ensure text is not None
        }]

        return segments, "openai_whisper"

    except (APIError, openai.APIError) as e: # Catch both new and old APIError types
         logger.error(f"OpenAI API error during Whisper transcription: {e}", exc_info=True)
         st.error(f"OpenAI API Error during transcription: {e}. Check API key/quota.")
         return None, None
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI Whisper transcription: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during transcription: {e}")
        return None, None
    finally:
         # --- Cleanup ---
         # Always remove the snippet file if it was created
         if cleanup_snippet and temp_snippet_file and os.path.exists(temp_snippet_file):
             try:
                 os.remove(temp_snippet_file)
                 logger.info(f"Removed temporary snippet file: {temp_snippet_file}")
             except OSError as e:
                 logger.warning(f"Failed to remove temporary snippet file {temp_snippet_file}: {e}")
         # Clean up the original audio's temp *directory* in the calling function (get_transcript_with_fallback)


def get_transcript_with_fallback(video_id):
    """Gets transcript from YouTube, falls back to Whisper via audio download."""
    # Use session state to cache per video_id during a single Streamlit run
    cache_key_data = f"transcript_data_{video_id}"
    cache_key_source = f"transcript_source_{video_id}"

    if cache_key_data in st.session_state:
        logger.info(f"Using cached transcript for {video_id} from session state (Source: {st.session_state.get(cache_key_source)}).")
        return st.session_state[cache_key_data], st.session_state.get(cache_key_source)

    # 1. Try YouTube Direct Transcript
    logger.info(f"Attempting to fetch YouTube transcript for {video_id}")
    direct_transcript = get_transcript(video_id)

    if direct_transcript:
        logger.info(f"Successfully fetched YouTube transcript for {video_id}")
        st.session_state[cache_key_data] = direct_transcript
        st.session_state[cache_key_source] = "youtube"
        return direct_transcript, "youtube"

    # 2. Fallback to Whisper if YouTube fails
    logger.warning(f"YouTube transcript failed or unavailable for {video_id}. Attempting Whisper fallback.")
    # Check if OpenAI key is available before attempting download
    if not OPENAI_API_KEY:
         logger.warning("OpenAI API key missing, skipping Whisper fallback.")
         st.info("YouTube transcript unavailable. OpenAI key not configured for Whisper fallback.")
         st.session_state[cache_key_data] = None
         st.session_state[cache_key_source] = None
         return None, None

    # Proceed with Whisper fallback
    st_whisper_placeholder = st.info("YouTube transcript unavailable. Trying Whisper fallback (downloading audio)...")
    audio_file_path = None
    audio_temp_dir = None
    whisper_transcript = None
    whisper_source = None

    try:
        audio_file_path, audio_temp_dir = download_audio(video_id)

        if audio_file_path and audio_temp_dir:
            st_whisper_placeholder.info("Audio downloaded. Transcribing with Whisper (this may take time)...")
            logger.info(f"Audio downloaded for {video_id} to {audio_file_path}. Proceeding with Whisper.")
            whisper_transcript, whisper_source = generate_transcript_with_openai(audio_file_path)

            if whisper_transcript:
                st_whisper_placeholder.success("Whisper transcription successful.")
                logger.info(f"Whisper transcription successful for {video_id}.")
            else:
                st_whisper_placeholder.error("Whisper transcription failed.")
                logger.error(f"Whisper transcription failed for {video_id}.")
        else:
            st_whisper_placeholder.error("Audio download failed. Cannot perform Whisper transcription.")
            logger.error(f"Audio download failed for {video_id}, cannot use Whisper fallback.")

    finally:
         # Always clean up the audio temp directory if it was created
         if audio_temp_dir and os.path.exists(audio_temp_dir):
              try:
                  logger.info(f"Cleaning up audio temp directory: {audio_temp_dir}")
                  shutil.rmtree(audio_temp_dir)
              except Exception as e:
                  logger.error(f"Failed to clean up audio temp directory {audio_temp_dir}: {e}")

    # Store results (or lack thereof) in session state
    st.session_state[cache_key_data] = whisper_transcript
    st.session_state[cache_key_source] = whisper_source
    return whisper_transcript, whisper_source


def get_intro_outro_transcript(video_id, total_duration):
    """Extracts text snippets for the intro and outro from a transcript."""
    transcript, source = get_transcript_with_fallback(video_id)
    if not transcript:
        logger.warning(f"No transcript available for {video_id} to extract intro/outro.")
        return (None, None)

    # Define intro/outro time boundaries
    # Intro: First 60 seconds or first 20% of video, whichever is shorter
    end_intro_sec = min(60, total_duration * 0.2 if total_duration > 0 else 60)
    # Outro: Last 60 seconds, but only if video is longer than 120s, starts at max(end_intro, duration-60)
    start_outro_sec = -1 # Default to no outro
    if total_duration > 120: # Only extract outro if video is longer than 2 minutes
         start_outro_sec = max(end_intro_sec, total_duration - 60)

    logger.debug(f"Extracting intro (0-{end_intro_sec:.1f}s) and outro ({start_outro_sec:.1f}s-{total_duration:.1f}s) for {video_id}")

    intro_texts = []
    outro_texts = []

    # Handle different transcript formats based on source
    if source == "youtube" and isinstance(transcript, list) and transcript and 'start' in transcript[0]:
        # YouTube format: list of dicts with 'start', 'duration', 'text'
        for item in transcript:
            try:
                start = float(item["start"])
                duration = float(item.get("duration", 2.0)) # Estimate duration if missing
                end = start + duration
                text = item.get("text", "").strip()
                if not text: continue

                # Check overlap with intro period [0, end_intro_sec)
                if max(0, start) < end_intro_sec:
                     intro_texts.append(text)

                # Check overlap with outro period [start_outro_sec, total_duration)
                if start_outro_sec >= 0 and max(start_outro_sec, start) < min(total_duration, end):
                     outro_texts.append(text)
            except (ValueError, TypeError):
                 logger.warning(f"Skipping transcript segment due to invalid time data: {item}")
                 continue

    elif source == "openai_whisper" and isinstance(transcript, list) and transcript and 'text' in transcript[0]:
        # Whisper format: often a single segment with full text, possibly no reliable timestamps
        full_text = " ".join(seg.get("text", "") for seg in transcript).strip()
        if not full_text: return (None, None)

        # Approximate splitting by word count if duration is known
        if total_duration > 0:
            words = full_text.split()
            num_words = len(words)
            if num_words < 10: return (None, None) # Too short to split reliably

            words_per_second = num_words / total_duration
            intro_word_limit = int(end_intro_sec * words_per_second)
            intro_text_approx = " ".join(words[:intro_word_limit]) if intro_word_limit > 0 else None

            outro_text_approx = None
            if start_outro_sec >= 0:
                 outro_start_word_index = int(start_outro_sec * words_per_second)
                 if outro_start_word_index < num_words:
                      outro_text_approx = " ".join(words[outro_start_word_index:])

            # Basic validation
            intro_text = intro_text_approx if intro_text_approx and len(intro_text_approx) > 10 else None
            outro_text = outro_text_approx if outro_text_approx and len(outro_text_approx) > 10 else None
            logger.debug("Used approximate word splitting for Whisper intro/outro.")
            return (intro_text, outro_text)
        else:
            # Cannot split reliably without duration
            logger.warning(f"Cannot split Whisper transcript for intro/outro without video duration for {video_id}.")
            return (None, None)

    else: # Unknown or unexpected format
         logger.warning(f"Unrecognized transcript format ('{source}') for {video_id}. Cannot extract intro/outro reliably.")
         return (None, None)

    # Join collected text snippets
    intro_full = " ".join(intro_texts) if intro_texts else None
    outro_full = " ".join(outro_texts) if outro_texts else None

    logger.debug(f"Intro extracted length: {len(intro_full or '')}, Outro extracted length: {len(outro_full or '')}")
    return (intro_full, outro_full)

# --- CORRECTED summarize_intro_outro function ---
def summarize_intro_outro(intro_text, outro_text):
    if not intro_text and not outro_text:
        return (None, None)

    if not OPENAI_API_KEY:
         st.error("OpenAI API key not configured. Cannot summarize.")
         return ("*Summary failed: OpenAI key missing.*", "*Summary failed: OpenAI key missing.*")

    # Generate a hash based on the combined text for caching
    combined_text = (intro_text or "") + "||" + (outro_text or "")
    # Truncate combined text before hashing if it's excessively long?
    max_hash_len = 5000
    cache_key = f"intro_outro_summary_{hashlib.sha256(combined_text[:max_hash_len].encode()).hexdigest()}"

    # Check session state cache
    if cache_key in st.session_state:
        logger.info(f"Using cached intro/outro summary for hash {cache_key[-8:]}")
        # Return tuple format
        return (st.session_state[cache_key], st.session_state[cache_key])

    prompt_parts = []
    # Limit input text length to control token usage/cost
    max_snippet_len = 2500 # Characters (~600 tokens)
    if intro_text:
        prompt_parts.append(f"Intro snippet:\n'''\n{intro_text[:max_snippet_len]}\n'''\n")
    if outro_text:
        prompt_parts.append(f"Outro snippet:\n'''\n{outro_text[:max_snippet_len]}\n'''\n")

    prompt_parts.append(
        "Based *only* on the text provided above, produce two concise bullet-point summaries (max 3-4 points each):\n"
        "1.  **Intro Summary:** Key points or hooks mentioned.\n"
        "2.  **Outro Summary:** Key takeaways or calls to action.\n"
        "If a snippet is missing, state 'Not available'. If a snippet is too short or unclear, state 'Not enough information'. "
        "Format the output clearly, using markdown for section titles (e.g., '**Intro Summary:**')."
    )
    prompt_str = "\n".join(prompt_parts)

    try:
        logger.info("Generating new intro/outro summary using OpenAI...")
        # Use updated client syntax if possible
        try:
             from openai import OpenAI, APIError
             client = OpenAI(api_key=OPENAI_API_KEY)
             response = client.chat.completions.create(
                 model="gpt-3.5-turbo", # Faster/cheaper model
                 messages=[{"role": "user", "content": prompt_str}],
                 max_tokens=350, # Adjusted token limit
                 temperature=0.5,
             )
             result_txt = response.choices[0].message.content

        except ImportError:
             # Fallback to older openai<1.0 syntax
             logger.debug("Using legacy OpenAI < 1.0 syntax for summarization.")
             # Import legacy error if needed for except block
             try:
                 from openai.error import APIError as OldAPIError
             except ImportError: # Handle case where even legacy openai isn't installed fully?
                 OldAPIError = Exception # Fallback to generic Exception if specific error class missing
             global APIError # Make accessible
             APIError = OldAPIError

             response = openai.ChatCompletion.create(
                 model="gpt-3.5-turbo",
                 messages=[{"role": "user", "content": prompt_str}],
                 max_tokens=350,
                 temperature=0.5,
                 api_key=OPENAI_API_KEY
             )
             result_txt = response.choices[0].message['content'] # type: ignore

        # --- Code after successful API call ---
        result_txt = result_txt.strip() if result_txt else "*Summary generation failed or returned empty.*"
        # Store the raw result in session cache
        st.session_state[cache_key] = result_txt
        logger.info(f"Stored new intro/outro summary in cache for hash {cache_key[-8:]}")

        # Return in the expected tuple format
        return (result_txt, result_txt)

    # --- Error handling for the API call (Correctly placed) ---
    except (APIError, openai.APIError) as e: # Catch both new/old errors (assuming openai.APIError exists if new client used)
         logger.error(f"OpenAI API error during intro/outro summarization: {e}", exc_info=True)
         st.error(f"OpenAI API error during summarization: {e}")
         return ("*Summary failed due to API error.*", "*Summary failed due to API error.*")
    except Exception as e:
        logger.error(f"Unexpected error during intro/outro summarization: {e}", exc_info=True)
        st.error(f"Unexpected error during summarization: {e}")
        return ("*Summary failed due to an unexpected error.*", "*Summary failed due to an unexpected error.*")
# --- END OF CORRECTED FUNCTION ---

def summarize_script(script_text):
    """Summarizes the full script text using OpenAI."""
    if not script_text or not script_text.strip():
        return "No script text provided to summarize."

    if not OPENAI_API_KEY:
         st.error("OpenAI API key not configured. Cannot summarize script.")
         return "Summary failed: OpenAI key missing."

    # Hash the first/last parts of the script for caching (more stable than full hash if minor edits)
    hash_prefix = script_text[:1000]
    hash_suffix = script_text[-1000:]
    hashed = hashlib.sha256((hash_prefix + hash_suffix).encode("utf-8")).hexdigest()

    # Initialize cache if needed
    if "script_summary_cache" not in st.session_state:
        st.session_state["script_summary_cache"] = {}
    # Check cache
    if hashed in st.session_state["script_summary_cache"]:
        logger.info(f"Using cached full script summary for hash {hashed[-8:]}")
        return st.session_state["script_summary_cache"][hashed]

    logger.info("Generating new full script summary using OpenAI...")
    # Truncate long scripts for API call to manage cost/tokens
    # Max tokens for gpt-3.5-turbo is often 4096 for prompt+completion
    # Aim for ~3000 tokens for the prompt (~12000 chars)
    max_chars = 12000
    truncated_script = script_text[:max_chars]
    if len(script_text) > max_chars:
         logger.warning(f"Full script text truncated from {len(script_text)} to {max_chars} characters for summarization.")
         truncated_script += "\n[... Script truncated ...]" # Indicate truncation

    prompt_content = (
        f"Please provide a concise, neutral summary (around 150-200 words) "
        f"highlighting the main topics and key information presented in the following video script:\n\n"
        f"'''\n{truncated_script}\n'''\n\n"
        f"Focus on the subject matter, core arguments, or steps described. Avoid adding opinions."
    )

    try:
         # Use updated client syntax if possible
         try:
             from openai import OpenAI, APIError
             client = OpenAI(api_key=OPENAI_API_KEY)
             response = client.chat.completions.create(
                 model="gpt-3.5-turbo",
                 messages=[{"role": "user", "content": prompt_content}],
                 max_tokens=300, # Limit output size
                 temperature=0.5,
             )
             summary = response.choices[0].message.content.strip()

         except ImportError:
             # Fallback to older openai<1.0 syntax
             logger.debug("Using legacy OpenAI < 1.0 syntax for full script summary.")
             try:
                 from openai.error import APIError as OldAPIError
             except ImportError:
                 OldAPIError = Exception
             global APIError # Make accessible
             APIError = OldAPIError

             response = openai.ChatCompletion.create(
                 model="gpt-3.5-turbo",
                 messages=[{"role": "user", "content": prompt_content}],
                 max_tokens=300,
                 temperature=0.5,
                 api_key=OPENAI_API_KEY
             )
             summary = response.choices[0].message['content'].strip() # type: ignore

         summary = summary if summary else "*Summary generation failed or returned empty.*"
         # Cache the result
         st.session_state["script_summary_cache"][hashed] = summary
         logger.info(f"Stored new full script summary in cache for hash {hashed[-8:]}")
         return summary

    # Error handling
    except (APIError, openai.APIError) as e:
        logger.error(f"OpenAI API error during full script summarization: {e}", exc_info=True)
        st.error(f"OpenAI API error during script summary: {e}")
        return "*Script summary failed due to API error.*"
    except Exception as e:
        logger.error(f"Unexpected error during full script summarization: {e}", exc_info=True)
        st.error(f"Unexpected error during script summary: {e}")
        return "*Script summary failed due to an unexpected error.*"


# =============================================================================
# 8. Searching & Calculating Outliers (INTEGRATED & MODIFIED)
# =============================================================================
def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# --- The MODIFIED calculate_metrics function (Integrated from previous step) ---
def calculate_metrics(df):
    """
    Calculates performance metrics for videos in the DataFrame.
    INTEGRATION: Uses a simplified relative performance score inspired by app.py.
                 Does NOT implement full view trajectory simulation.
    """
    if df.empty:
        return df, None # Return empty dataframe if input is empty

    now_utc = datetime.now(timezone.utc) # Use timezone-aware now

    # Convert published_at to datetime objects, handling potential errors
    # Ensure published_at exists before conversion
    if 'published_at' not in df.columns:
         logger.error("Missing 'published_at' column in DataFrame for metric calculation.")
         st.error("Internal data error: Missing publish date. Cannot calculate metrics.")
         return pd.DataFrame(), None # Return empty if essential column missing

    df['published_at_dt'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)

    # Drop rows where published_at couldn't be parsed
    initial_rows = len(df)
    df.dropna(subset=['published_at_dt'], inplace=True)
    if len(df) < initial_rows:
         logger.warning(f"Dropped {initial_rows - len(df)} rows due to invalid publish dates.")
    if df.empty:
        logger.warning("DataFrame empty after dropping rows with invalid publish dates.")
        return pd.DataFrame(), None

    # Calculate age consistently
    df['hours_since_published'] = ((now_utc - df['published_at_dt']).dt.total_seconds() / 3600)
    # Ensure hours_since_published is not zero or negative to avoid division errors
    # Use a small positive minimum like 1/60th of an hour (1 minute)
    df['hours_since_published'] = df['hours_since_published'].apply(lambda x: max(x, 1/60))
    df['days_since_published'] = (df['hours_since_published'] / 24) # Keep fractional days

    # --- Standard Metrics ---
    # Ensure numeric types for calculation columns, coercing errors to 0
    stat_cols = ['views', 'like_count', 'comment_count']
    for col in stat_cols:
        if col not in df.columns: # Add column if missing, fill with 0
             df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # --- Views Per Hour (VPH) ---
    # Raw VPH (Views / Actual Hours)
    df['raw_vph'] = df['views'] / df['hours_since_published']

    # Peak VPH (Views / Hours capped at 30 days = 720 hours)
    df['peak_hours'] = df['hours_since_published'].apply(lambda x: min(x, 720.0))
    # Ensure peak_hours is also non-zero for division
    df['peak_vph'] = df['views'] / df['peak_hours'].apply(lambda x: max(x, 1/60))

    # Effective VPH (Use raw for recent <90 days, peak for older >=90 days)
    df['effective_vph'] = df.apply(
        lambda row: row['raw_vph'] if row['days_since_published'] < 90 else row['peak_vph'],
        axis=1
    )

    # --- Engagement Metrics ---
    # Weighted Engagement Metric = Likes + 5 * Comments
    df['engagement_metric'] = df['like_count'] + 5 * df['comment_count']
    # Engagement Rate = Weighted Engagement / Views (safe division)
    df['engagement_rate'] = df['engagement_metric'] / df['views'].apply(lambda x: max(x, 1))

    # Basic Ratios (as floats for calculation)
    df['cvr_float'] = df['comment_count'] / df['views'].apply(lambda x: max(x, 1))
    df['clr_float'] = df['like_count'] / df['views'].apply(lambda x: max(x, 1))

    # --- Channel Averages and Relative Metrics (Grouped by Channel) ---
    channel_metrics = {} # To store calculated averages per channel
    results_list = []    # To store processed groups

    # Ensure channelId exists for grouping
    if 'channelId' not in df.columns:
         logger.error("Missing 'channelId' column. Cannot calculate channel-relative metrics.")
         st.error("Internal data error: Missing channel ID. Cannot calculate relative metrics.")
         # Add placeholder metrics? Or return? Returning is safer.
         return pd.DataFrame(), None

    for channel_id, group in df.groupby('channelId'):
        group = group.copy() # Avoid SettingWithCopyWarning

        # Determine videos to use for averaging (prefer last 30d, fallback to 90d, then all)
        recent_videos = group[group['days_since_published'] <= 30].copy()
        avg_period_days = 30
        if len(recent_videos) < 5: # Require at least 5 videos for 30d avg
            recent_videos = group[group['days_since_published'] <= 90].copy()
            avg_period_days = 90
            if len(recent_videos) < 5: # Still not enough? Use all available
                 recent_videos = group.copy()
                 avg_period_days = int(group['days_since_published'].max()) if not group.empty else 0
                 logger.warning(f"Channel {channel_id}: Using all {len(group)} videos for avg (needed >5, found {len(recent_videos)} in 90d)")
        num_videos_for_avg = len(recent_videos)
        logger.debug(f"Channel {channel_id}: Using {num_videos_for_avg} videos from last {avg_period_days} days for averaging.")

        # Calculate Channel Average Metrics (handle empty recent_videos case)
        if num_videos_for_avg > 0:
            channel_avg_vph = recent_videos['effective_vph'].mean()
            channel_avg_engagement = recent_videos['engagement_rate'].mean()
            channel_avg_cvr = recent_videos['cvr_float'].mean()
            channel_avg_clr = recent_videos['clr_float'].mean()
        else: # No videos found for averaging, use defaults (avoids division errors later)
             channel_avg_vph = 0.1
             channel_avg_engagement = 0.0001
             channel_avg_cvr = 0.0001
             channel_avg_clr = 0.0001
             logger.warning(f"Channel {channel_id}: No videos available to calculate averages.")


        # Store averages for potential debugging or later use
        channel_metrics[channel_id] = {
            'avg_vph': channel_avg_vph, 'avg_engagement': channel_avg_engagement,
            'avg_cvr': channel_avg_cvr, 'avg_clr': channel_avg_clr,
            'num_videos_for_avg': num_videos_for_avg, 'avg_period_days': avg_period_days
        }
        logger.debug(f"Channel {channel_id} Avgs: VPH={channel_avg_vph:.2f}, Eng={channel_avg_engagement:.4f}, CVR={channel_avg_cvr:.4f}, CLR={channel_avg_clr:.4f}")

        # --- Calculate Ratios Relative to Channel Average ---
        # Apply averages to the current group (all videos, not just recent ones)
        # Use safe division (max with a small non-zero number)
        group['channel_avg_vph'] = channel_avg_vph
        group['channel_avg_engagement'] = channel_avg_engagement
        group['channel_avg_cvr'] = channel_avg_cvr
        group['channel_avg_clr'] = channel_avg_clr

        group['vph_ratio'] = group['effective_vph'] / max(channel_avg_vph, 0.1)
        group['engagement_ratio'] = group['engagement_rate'] / max(channel_avg_engagement, 0.0001)
        group['outlier_cvr'] = group['cvr_float'] / max(channel_avg_cvr, 0.0001)
        group['outlier_clr'] = group['clr_float'] / max(channel_avg_clr, 0.0001)

        # --- **** NEW OUTLIER SCORE CALCULATION **** ---
        # Combined performance ratio: Weighted average of VPH ratio and Engagement ratio.
        # Weights can be adjusted (e.g., 0.8 VPH, 0.2 Engagement)
        vph_weight = 0.85
        eng_weight = 0.15
        group['combined_performance'] = (vph_weight * group['vph_ratio']) + (eng_weight * group['engagement_ratio'])

        # Define 'outlier_score' as this combined performance ratio.
        # Score > 1.0 means above average weighted performance.
        # Score < 1.0 means below average weighted performance.
        group['outlier_score'] = group['combined_performance']

        # Keep breakout_score alias for backward compatibility in UI if needed
        group['breakout_score'] = group['outlier_score']

        # --- Formatting for Display ---
        group['formatted_views'] = group['views'].apply(format_number)
        group['comment_to_view_ratio'] = group['cvr_float'].apply(lambda x: f"{x*100:.2f}%")
        # Renamed from app(1), CLR is Like/View. Keep consistent name.
        group['like_to_view_ratio'] = group['clr_float'].apply(lambda x: f"{x*100:.2f}%")
        # Keep the old name as well if UI depends on it? Or just update UI. Let's just use the correct one.
        # group['comment_to_like_ratio'] = group['clr_float'].apply(lambda x: f"{(x*100):.2f}%") # Incorrect name

        group['vph_display'] = group['effective_vph'].apply(lambda x: f"{int(round(x,0))} VPH" if x>0 else "0 VPH")

        # Append processed group to the list
        results_list.append(group)

    # --- Final Concatenation & Return ---
    if not results_list:
         logger.warning("No results generated after processing channel groups.")
         return pd.DataFrame(), None # Return empty df

    final_df = pd.concat(results_list).reset_index(drop=True) # Reset index after concat

    # Drop intermediate calculation columns if desired (optional)
    # final_df = final_df.drop(columns=['published_at_dt', 'peak_hours', 'engagement_metric'])

    logger.info(f"Calculated metrics for {len(final_df)} videos across {len(channel_metrics)} channels.")
    return final_df, None # Second element was unused placeholder
# --- END OF MODIFIED calculate_metrics ---


def fetch_all_snippets(channel_id, order_param, timeframe, query, published_after):
    """Fetches basic video snippet data (ID, title, publish time) for a channel via Search API."""
    all_videos = []
    page_token = None
    try:
        key = get_youtube_api_key()
    except ValueError as e:
        st.error(f"Cannot fetch snippets: {e}")
        logger.error(f"API key error in fetch_all_snippets: {e}")
        return []

    base_url = (
        f"https://www.googleapis.com/youtube/v3/search?part=snippet"
        f"&channelId={channel_id}&maxResults=50&type=video&order={order_param}&key={key}"
    )
    if published_after:
        base_url += f"&publishedAfter={published_after}"
    if query:
        from urllib.parse import quote_plus
        base_url += f"&q={quote_plus(query)}"

    max_results_limit = 200 # Limit total results per channel
    fetched_count = 0
    max_pages = (max_results_limit + 49) // 50 # Max pages to fetch

    logger.info(f"Fetching snippets for channel {channel_id} (Order: {order_param}, Query: '{query}', After: {published_after}, Limit: {max_results_limit})")

    for page_num in range(max_pages):
        url = base_url
        if page_token:
            url += f"&pageToken={page_token}"

        try:
            logger.debug(f"Requesting Snippets URL (Page {page_num+1}): {url.split('&key=')[0]}...")
            resp = requests.get(url, timeout=20)
            logger.debug(f"Response status: {resp.status_code}")

            if resp.status_code == 403:
                 logger.error(f"API Key/Quota Error (Forbidden) fetching snippets for {channel_id}: {resp.text}")
                 st.error(f"YouTube API error (Forbidden/Quota?). Check key and usage.")
                 break # Stop fetching for this channel
            elif resp.status_code == 400:
                 logger.error(f"API Bad Request fetching snippets for {channel_id}: {resp.text}")
                 st.error(f"YouTube API error (Bad Request): {resp.text}. Check parameters.")
                 break
            resp.raise_for_status() # Raise other HTTP errors

            data = resp.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during snippet request for {channel_id} (Page {page_num+1}): {e}", exc_info=True)
            st.error(f"Network error fetching video list for channel {channel_id}. Results may be incomplete.")
            break # Stop fetching for this channel

        items = data.get("items", [])
        if not items and fetched_count == 0 and page_num == 0:
             logger.warning(f"No video snippets found for channel {channel_id} on first page with current filters.")
             break # No videos found at all

        for it in items:
            vid_id = it.get("id", {}).get("videoId")
            snippet = it.get("snippet")
            if not vid_id or not snippet:
                logger.warning(f"Skipping invalid item in snippet response: {it}")
                continue

            published_at_raw = snippet.get("publishedAt")
            if not published_at_raw or not isinstance(published_at_raw, str):
                 logger.warning(f"Missing or invalid publishedAt for video {vid_id}: {published_at_raw}. Skipping video.")
                 continue

            # Extract medium thumbnail URL safely
            thumbnail_url = snippet.get("thumbnails", {}).get("medium", {}).get("url", "")

            all_videos.append({
                "video_id": vid_id,
                "title": snippet.get("title", "No Title Provided"),
                "channel_name": snippet.get("channelTitle", "Unknown Channel"),
                "channelId": snippet.get("channelId", channel_id), # Use snippet channelId if available, else fallback
                "publish_date": format_date(published_at_raw), # Formatted for display
                "published_at": published_at_raw, # ISO format for calculations
                "thumbnail": thumbnail_url
            })
            fetched_count += 1
            if fetched_count >= max_results_limit:
                break # Exit inner loop if limit reached

        # Check if limit reached after processing items
        if fetched_count >= max_results_limit:
            logger.info(f"Reached max results limit ({max_results_limit}) for channel {channel_id}.")
            break

        # Check for next page token
        page_token = data.get("nextPageToken")
        if not page_token:
            logger.debug(f"No more pages for channel {channel_id}.")
            break # No more pages

    logger.info(f"Fetched {len(all_videos)} total snippets for channel {channel_id}.")
    return all_videos


def search_youtube(query, channel_ids, timeframe, content_filter, ttl=600):
    """
    Searches YT, fetches details, calculates metrics (incl. new outlier score), caches results.
    Returns data as a list of dictionaries.
    """
    query = query.strip()
    # Use tuple for channel_ids in cache key for consistency
    channel_ids_tuple = tuple(sorted(channel_ids))
    is_broad_scan = not query and timeframe == "3 months" and content_filter.lower() == "both"
    effective_ttl = 7776000 if is_broad_scan else ttl # 90 days TTL for broad scans

    cache_key_params = [query, channel_ids_tuple, timeframe, content_filter]
    cache_key = build_cache_key(*cache_key_params)

    # --- Cache Check ---
    cached_result_list = get_cached_result(cache_key, ttl=effective_ttl)
    if cached_result_list is not None:
        # Ensure cached data is a list of dicts (our standard return format)
        if isinstance(cached_result_list, list):
             # Quick check if it looks like video data (has video_id and outlier_score)
             if not cached_result_list or (isinstance(cached_result_list[0], dict) and 'video_id' in cached_result_list[0] and 'outlier_score' in cached_result_list[0]):
                  logger.info(f"Cache hit for key {cache_key[-8:]}. Returning {len(cached_result_list)} items from cache.")
                  # Apply content filter *after* loading from cache (as cache stores 'both')
                  if content_filter.lower() == "shorts":
                      return [r for r in cached_result_list if r.get("content_category") == "Short"]
                  elif content_filter.lower() == "videos":
                      return [r for r in cached_result_list if r.get("content_category") == "Video"]
                  else: # 'both'
                      return cached_result_list
             else:
                  logger.warning(f"Cached data for {cache_key[-8:]} is a list but not in expected format. Refetching.")
                  delete_cache_key(cache_key) # Invalidate bad cache
        else:
             logger.warning(f"Cached data for {cache_key[-8:]} is not a list ({type(cached_result_list)}). Refetching.")
             delete_cache_key(cache_key) # Invalidate bad cache

    # --- API Fetching ---
    logger.info(f"Cache miss or invalid for key {cache_key[-8:]}. Fetching fresh data.")
    st_fetch_placeholder = st.info("Fetching fresh data from YouTube API...")

    order_param = "relevance" if query else "date"
    pub_after_iso = None
    if timeframe != "Lifetime":
        now_utc = datetime.now(timezone.utc)
        delta_map = {
            "Last 24 hours": timedelta(days=1), "Last 48 hours": timedelta(days=2),
            "Last 4 days": timedelta(days=4), "Last 7 days": timedelta(days=7),
            "Last 15 days": timedelta(days=15), "Last 28 days": timedelta(days=28),
            "3 months": timedelta(days=90),
        }
        delta = delta_map.get(timeframe)
        if delta:
            pub_after_dt = now_utc - delta
            pub_after_iso = pub_after_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Fetch snippets for all channels
    all_snippets = []
    with st.spinner(f"Fetching video list for {len(channel_ids)} channel(s)..."):
        for cid in channel_ids:
            channel_snippets = fetch_all_snippets(cid, order_param, timeframe, query, pub_after_iso)
            all_snippets.extend(channel_snippets)
            # time.sleep(0.05) # Small delay


    if not all_snippets:
        logger.warning("No snippets found for any selected channel with criteria.")
        st_fetch_placeholder.warning("No videos found matching your criteria.")
        set_cached_result(cache_key, []) # Cache the empty result
        return []

    # Deduplicate snippets by video_id
    unique_snippets_dict = {s['video_id']: s for s in all_snippets}
    unique_snippet_list = list(unique_snippets_dict.values())
    vid_ids = list(unique_snippets_dict.keys())
    logger.info(f"Found {len(vid_ids)} unique video IDs.")

    # Fetch statistics and content details in chunks
    all_stats_details = {}
    max_ids_per_request = 50
    video_id_chunks = list(chunk_list(vid_ids, max_ids_per_request))

    try:
        key = get_youtube_api_key()
        with st.spinner(f"Fetching details for {len(vid_ids)} videos...") as details_spinner:
             total_fetched_details = 0
             for i, chunk in enumerate(video_id_chunks):
                 logger.info(f"Fetching details chunk {i+1}/{len(video_id_chunks)} ({len(chunk)} IDs)")
                 ids_str = ','.join(chunk)
                 # Request snippet, contentDetails, statistics
                 stats_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={ids_str}&key={key}"
                 try:
                      resp = requests.get(stats_url, timeout=25)
                      if resp.status_code == 403: raise requests.exceptions.RequestException(f"API Forbidden (403): {resp.text}")
                      if resp.status_code == 400: raise requests.exceptions.RequestException(f"API Bad Request (400): {resp.text}")
                      resp.raise_for_status()

                      details_data = resp.json()
                      items_in_response = details_data.get("items", [])
                      total_fetched_details += len(items_in_response)

                      for item in items_in_response:
                          vid = item["id"]
                          stats = item.get("statistics", {})
                          content = item.get("contentDetails", {})
                          snippet = item.get("snippet", {}) # Get snippet for reliable publish time & duration def

                          duration_str = content.get("duration", "PT0S")
                          duration_sec = parse_iso8601_duration(duration_str)
                          # Use official 'definition' field for shorts if available, else fallback to duration <= 60
                          definition = content.get("definition", "") # e.g., 'hd', 'sd'
                          # Refined check for Shorts: duration <= 60s AND not a live stream/premiere
                          is_short_definition = (snippet.get("liveBroadcastContent", "none") == "none" and duration_sec > 0 and duration_sec <= 60)

                          category = "Short" if is_short_definition else "Video"

                          all_stats_details[vid] = {
                              "views": int(stats.get("viewCount", 0)),
                              "like_count": int(stats.get("likeCount", 0)),
                              "comment_count": int(stats.get("commentCount", 0)),
                              "duration_seconds": duration_sec,
                              "content_category": category,
                              "published_at": snippet.get("publishedAt") # Use timestamp from video details API
                          }
                 except requests.exceptions.RequestException as e:
                      logger.error(f"Details request failed for chunk {i+1}: {e}")
                      st.warning(f"Network/API error fetching details (chunk {i+1}). Some data may be missing.")
                      # Continue to next chunk if possible
                 # time.sleep(0.05) # Small delay

    except ValueError as e: # Catch API key error from get_youtube_api_key
         st_fetch_placeholder.error(f"Cannot fetch details: {e}")
         logger.error(f"API key error preventing details fetch: {e}")
         set_cached_result(cache_key, [])
         return []
    except Exception as e: # Catch other unexpected errors during loop setup etc.
         st_fetch_placeholder.error(f"Unexpected error during details fetch setup: {e}")
         logger.error(f"Unexpected error setting up details fetch: {e}", exc_info=True)
         set_cached_result(cache_key, [])
         return []


    # Combine snippet data with stats/details
    combined_results = []
    missing_details_count = 0
    for snippet_data in unique_snippet_list:
        vid = snippet_data["video_id"]
        if vid in all_stats_details:
            stats_info = all_stats_details[vid]
            # Use the more reliable published_at from the details API
            if stats_info.get("published_at"):
                 snippet_data["published_at"] = stats_info["published_at"]
                 snippet_data["publish_date"] = format_date(stats_info["published_at"])
            else:
                 logger.warning(f"Missing published_at from details API for {vid}. Using snippet version.")

            combined = {**snippet_data, **stats_info}
            combined_results.append(combined)
        else:
            missing_details_count += 1
            logger.warning(f"Missing stats/details for video ID: {vid}. Skipping.")

    if missing_details_count > 0:
         st.warning(f"Could not fetch details for {missing_details_count} videos (may be deleted or API issue).")

    if not combined_results:
         logger.warning("No results after combining snippets and details.")
         st_fetch_placeholder.warning("Could not retrieve details for any found videos.")
         set_cached_result(cache_key, [])
         return []

    # Create DataFrame and calculate metrics
    st_fetch_placeholder.info("Calculating performance metrics...")
    try:
         df_results = pd.DataFrame(combined_results)
         df_with_metrics, _ = calculate_metrics(df_results) # Call the integrated function

         if df_with_metrics is None or df_with_metrics.empty:
              logger.warning("Metrics calculation returned empty or None.")
              st_fetch_placeholder.warning("Failed to calculate performance metrics.")
              final_results_list = []
         else:
              # Convert DataFrame back to list of dicts for caching and return
              final_results_list = df_with_metrics.to_dict("records")
              # Cache the final list (before content filtering)
              try:
                  set_cached_result(cache_key, final_results_list)
                  logger.info(f"Successfully cached {len(final_results_list)} results for key {cache_key[-8:]}")
              except Exception as e:
                  logger.error(f"Failed to cache results: {e}", exc_info=True)
                  st.warning(f"Could not cache results due to: {e}") # Non-critical error

    except Exception as e:
         logger.error(f"Error during metrics calculation or DataFrame conversion: {e}", exc_info=True)
         st_fetch_placeholder.error(f"Error calculating metrics: {e}")
         final_results_list = [] # Return empty on error


    st_fetch_placeholder.empty() # Clear the "Fetching..." message

    # Apply content filter *after* calculation and caching
    if content_filter.lower() == "shorts":
        filtered_list = [r for r in final_results_list if r.get("content_category") == "Short"]
    elif content_filter.lower() == "videos":
        filtered_list = [r for r in final_results_list if r.get("content_category") == "Video"]
    else: # "Both"
        filtered_list = final_results_list

    logger.info(f"Returning {len(filtered_list)} videos after content filtering ('{content_filter}').")
    return filtered_list


# =============================================================================
# 9. Comments & Analysis
# =============================================================================
def analyze_comments(comments):
    """Analyzes comments using OpenAI, with caching."""
    if not comments:
        return "No comments provided for analysis."

    if not OPENAI_API_KEY:
         st.error("OpenAI API key not configured. Cannot analyze comments.")
         return "Analysis failed: OpenAI key missing."

    # Create a representative string/hash from comments for caching
    # Use a subset of comments if too many, and hash the text content + like counts?
    sample_size = min(len(comments), 100) # Analyze up to 100 comments
    comments_to_hash = sorted(comments, key=lambda c: c.get("likeCount", 0), reverse=True)[:sample_size] # Use top comments for hash
    comments_text = "\n".join([f"{c.get('likeCount',0)}:{c.get('text', '')[:100]}" for c in comments_to_hash]) # Include likes and snippet
    hashed = hashlib.sha256(comments_text.encode("utf-8")).hexdigest()

    # Initialize cache if needed
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}
    # Check cache
    if hashed in st.session_state["analysis_cache"]:
        logger.info(f"Using cached comment analysis for hash {hashed[-8:]}")
        return st.session_state["analysis_cache"][hashed]

    logger.info(f"Generating new comment analysis for {len(comments)} comments using OpenAI...")

    # Prepare the prompt for GPT
    # Use top comments for analysis input as well
    analysis_input_comments = "\n".join([f"- {c.get('text', '')}" for c in comments_to_hash])
    prompt_content = (
        f"Analyze the following YouTube comments (sampled, ordered by likes) and provide a concise summary in three sections:\n"
        "1.  **Positive Sentiment:** Briefly summarize the main positive feedback or appreciation (2-4 bullet points).\n"
        "2.  **Negative/Critical Sentiment:** Briefly summarize the main criticisms, complaints, or negative points (2-4 bullet points).\n"
        "3.  **Questions & Suggestions:** List the top 3-5 recurring questions or suggested topics/improvements mentioned.\n\n"
        "Focus on recurring themes. Keep summaries brief and neutral. If sentiment is very one-sided, mention it.\n\n"
        "--- Sampled Comments ---\n"
        f"{analysis_input_comments}"
        "\n--- End Comments ---"
    )

    try:
        # Use updated client syntax if possible
         try:
             from openai import OpenAI, APIError
             client = OpenAI(api_key=OPENAI_API_KEY)
             response = client.chat.completions.create(
                 model="gpt-3.5-turbo",
                 messages=[{"role": "user", "content": prompt_content}],
                 max_tokens=450, # Allow slightly more space
                 temperature=0.6,
             )
             analysis_text = response.choices[0].message.content.strip()

         except ImportError:
             # Fallback to older openai<1.0 syntax
             logger.debug("Using legacy OpenAI < 1.0 syntax for comment analysis.")
             try: from openai.error import APIError as OldAPIError
             except ImportError: OldAPIError = Exception
             global APIError
             APIError = OldAPIError

             response = openai.ChatCompletion.create(
                 model="gpt-3.5-turbo",
                 messages=[{"role": "user", "content": prompt_content}],
                 max_tokens=450,
                 temperature=0.6,
                 api_key=OPENAI_API_KEY
             )
             analysis_text = response.choices[0].message['content'].strip() # type: ignore

         analysis_text = analysis_text if analysis_text else "*AI analysis returned empty.*"
         # Cache the result
         st.session_state["analysis_cache"][hashed] = analysis_text
         logger.info(f"Stored new comment analysis in cache for hash {hashed[-8:]}")
         return analysis_text

    # Error Handling
    except (APIError, openai.APIError) as e:
        logger.error(f"OpenAI API error during comment analysis: {e}", exc_info=True)
        st.error(f"OpenAI API error during comment analysis: {e}")
        return "*Comment analysis failed due to API error.*"
    except Exception as e:
        logger.error(f"Unexpected error during comment analysis: {e}", exc_info=True)
        st.error(f"Unexpected error during comment analysis: {e}")
        return "*Comment analysis failed due to an unexpected error.*"

def get_video_comments(video_id, max_comments=100):
    """Fetches top or relevant comments for a video."""
    comments = []
    page_token = None
    try:
        key = get_youtube_api_key()
    except ValueError as e: # Catch specific error for missing key
        st.error(f"Cannot fetch comments: {e}")
        return []

    # Fetch by relevance first, as it often surfaces interesting comments
    order = "relevance"
    logger.info(f"Fetching up to {max_comments} comments for video {video_id} (order: {order})")

    while len(comments) < max_comments:
        try:
            url = (
                "https://www.googleapis.com/youtube/v3/commentThreads"
                f"?part=snippet&videoId={video_id}&maxResults={min(50, max_comments - len(comments))}"
                f"&order={order}&textFormat=plainText&key={key}" # Use plain text
            )
            if page_token:
                url += f"&pageToken={page_token}"

            resp = requests.get(url, timeout=15)

            if resp.status_code == 403:
                 # Handle comments disabled specifically
                 try:
                      error_details = resp.json()
                      if any(err.get('reason') == 'commentsDisabled' for err in error_details.get('error',{}).get('errors',[])):
                           logger.warning(f"Comments are disabled for video {video_id}.")
                           # Don't show error to user, just return empty
                           return []
                 except Exception: pass # Ignore parsing errors, fall through to general error
                 logger.error(f"API Key/Quota Error (Forbidden) fetching comments for {video_id}: {resp.text}")
                 st.error("YouTube API error (Forbidden/Quota?) fetching comments.")
                 break # Stop trying

            # Handle 404 Not Found (e.g., video deleted)
            if resp.status_code == 404:
                 logger.warning(f"Video {video_id} not found (404) when fetching comments.")
                 st.warning("Video not found, cannot fetch comments.")
                 return []

            resp.raise_for_status() # Raise other HTTP errors (e.g., 5xx)

            data = resp.json()
            items = data.get("items", [])

            if not items and len(comments) == 0 and not page_token: # No items on first page
                 logger.info(f"No comments found for video {video_id}.")
                 # Verify comment count via Videos API if possible (more reliable)
                 try:
                      stats_url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={video_id}&key={key}"
                      stats_resp = requests.get(stats_url, timeout=5).json()
                      if stats_resp.get('items'):
                           comment_count = int(stats_resp['items'][0].get('statistics', {}).get('commentCount', -1))
                           if comment_count == 0:
                                logger.info(f"Confirmed zero comments for video {video_id} via stats API.")
                                return [] # Definitely no comments
                           else: # Count > 0 but API returned none? Odd.
                                logger.warning(f"Stats API shows {comment_count} comments, but commentThreads API returned none for {video_id}.")
                      else: # Video details not found?
                           logger.warning(f"Could not get video stats to verify comment count for {video_id}.")
                 except Exception as e:
                      logger.warning(f"Error verifying comment count via stats API: {e}")
                 # If count check fails or > 0, maybe temporary API issue?
                 st.warning("Could not retrieve comments (API issue or none exist).")
                 break


            for item in items:
                try:
                    top_comment = item.get("snippet", {}).get("topLevelComment", {})
                    snippet = top_comment.get("snippet")
                    if not snippet: continue # Skip if essential snippet missing

                    text_cleaned = snippet.get("textOriginal", "").strip() # Use textOriginal (plain text)
                    if not text_cleaned: continue # Skip empty comments

                    comments.append({
                        "text": text_cleaned,
                        "likeCount": int(snippet.get("likeCount", 0)),
                        "author": snippet.get("authorDisplayName", "Unknown Author"),
                        "published_at": snippet.get("publishedAt"), # ISO timestamp
                        "comment_id": top_comment.get("id")
                        })
                except Exception as e:
                     logger.error(f"Error processing comment item: {item} - Error: {e}")
                     continue # Skip malformed comment

            # Check for next page
            page_token = data.get("nextPageToken")
            if not page_token:
                break # No more pages

        except requests.exceptions.Timeout:
             logger.error(f"Timeout fetching comments for {video_id}.")
             st.error("Timeout fetching comments. Please try again.")
             break
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching comments for {video_id}: {e}", exc_info=True)
            st.error(f"Error retrieving comments: {e}")
            break # Stop trying on error

    logger.info(f"Fetched {len(comments)} comments successfully for {video_id}.")
    return comments


# =============================================================================
# 10. Retention Analysis
# =============================================================================
# NOTE: Selenium-based retention requires a specific environment setup.
# These functions remain largely unchanged from the previous version.

# Use absolute path for Chromium if running in standard Linux env like Streamlit Cloud
CHROMIUM_PATH_STANDARD = "/usr/bin/chromium"

def check_selenium_setup():
     """Checks if required components for Selenium are likely present."""
     # 1. Check for Chrome/Chromium
     browser_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium-browser") or shutil.which("chromium")
     if not browser_path and os.path.exists(CHROMIUM_PATH_STANDARD):
          browser_path = CHROMIUM_PATH_STANDARD # Use standard path if others fail
     if not browser_path:
          st.warning("⚠️ Chrome/Chromium browser not found. Retention analysis requires it.")
          logger.warning("Chrome/Chromium executable not found.")
          return False
     logger.info(f"Using browser found at: {browser_path}")

     # 2. Check ChromeDriver using webdriver-manager
     try:
          # Let webdriver-manager find the appropriate driver version
          logger.info("Checking/Installing ChromeDriver via webdriver-manager...")
          os.environ['WDM_LOG_LEVEL'] = '0' # Suppress excessive WDM logs
          driver_path = ChromeDriverManager().install()
          if not driver_path or not os.path.exists(driver_path):
               st.warning("⚠️ ChromeDriver could not be installed/found by webdriver-manager.")
               logger.warning("webdriver-manager failed to install/find driver.")
               return False
          logger.info(f"ChromeDriver seems available at: {driver_path}")
     except Exception as e:
          # Catch specific errors? e.g., network error downloading driver
          st.warning(f"⚠️ Error setting up ChromeDriver: {e}. Retention analysis might fail.")
          logger.error(f"Error installing/checking ChromeDriver via webdriver-manager: {e}", exc_info=True)
          return False

     # 3. Check ffmpeg (redundant check, but good for completeness)
     if not shutil.which("ffmpeg"):
          st.warning("⚠️ ffmpeg not found in PATH. Required for video snippet extraction.")
     # 4. Check yt-dlp (redundant check)
     if not check_ytdlp_installed():
          st.warning("⚠️ yt-dlp not found. Required for downloading video snippets.")

     logger.info("Selenium setup dependencies appear to be met.")
     return True


def load_cookies(driver, cookie_file="youtube_cookies.json"):
    """Loads cookies from a JSON file into the Selenium driver."""
    if not os.path.exists(cookie_file):
        logger.warning(f"Cookie file '{cookie_file}' not found. Proceeding without cookies.")
        st.info("YouTube cookie file not found. Retention graph might be less accurate or unavailable.")
        return False

    logger.info(f"Loading cookies from {cookie_file}")
    try:
        with open(cookie_file, "r", encoding="utf-8") as f:
            cookies = json.load(f)
        if not isinstance(cookies, list):
             raise ValueError("Cookie file should contain a list of cookie objects.")
    except (json.JSONDecodeError, ValueError, IOError) as e:
        logger.error(f"Error reading or parsing cookie file {cookie_file}: {e}")
        st.error(f"Error reading cookie file: {e}")
        return False

    # Navigate to the base domain *before* adding cookies related to that domain
    try:
        driver.get("https://www.youtube.com/")
        time.sleep(2) # Allow page to start loading
    except Exception as e:
         logger.error(f"Failed to navigate to youtube.com before loading cookies: {e}")
         st.warning("Could not navigate to YouTube before loading cookies.")
         # Continue trying to load cookies anyway? Might work for some.

    added_count = 0
    skipped_count = 0
    current_domain = ".youtube.com" # Target domain for cookies

    for cookie in cookies:
        # Basic validation and cleaning
        if not isinstance(cookie, dict) or 'name' not in cookie or 'value' not in cookie:
            skipped_count += 1
            continue

        # Ensure domain matches or is parent domain
        cookie_domain = cookie.get('domain', '').lower()
        if not cookie_domain.endswith(current_domain):
             # logger.debug(f"Skipping cookie for wrong domain: {cookie.get('name')} ({cookie_domain})")
             skipped_count += 1
             continue

        # Clean attributes known to cause issues with `add_cookie`
        if 'sameSite' in cookie:
             valid_same_site = ['Strict', 'Lax', 'None']
             if cookie['sameSite'] not in valid_same_site:
                 # Default to Lax or remove? Removing is safer.
                 del cookie['sameSite']
        if "expiry" in cookie and cookie["expiry"] is not None:
             # Convert expiry (often float timestamp) to integer if needed
             try: cookie["expires"] = int(cookie.pop("expiry"))
             except (ValueError, TypeError): del cookie['expires'] # Remove if conversion fails
        if "expires" in cookie and cookie["expires"] is None:
             del cookie["expires"] # Remove None value for expires

        # Remove httpOnly and secure if they are not boolean? (Less common issue)

        try:
            driver.add_cookie(cookie)
            added_count += 1
        except Exception as e:
            skipped_count += 1
            # Log only first few errors to avoid flooding
            if skipped_count < 5:
                 logger.warning(f"Skipping cookie '{cookie.get('name', 'N/A')}' due to error: {e}")

    logger.info(f"Attempted to load cookies: Added {added_count}, Skipped {skipped_count}.")
    if added_count > 0:
         # Refresh page *after* adding cookies
         logger.info("Refreshing page after loading cookies.")
         try:
             driver.refresh()
             time.sleep(3) # Allow refresh to complete
             return True
         except Exception as e:
              logger.error(f"Failed to refresh page after loading cookies: {e}")
              st.warning("Failed to refresh page after cookie load.")
              return False # Indicate potential issue
    else:
         logger.warning("No valid cookies were added.")
         return False


def capture_player_screenshot_with_hover(video_url, timestamp, output_path="player_retention.png", use_cookies=True):
    """Captures screenshot of YouTube player with retention graph hover."""
    if not check_selenium_setup():
         raise EnvironmentError("Selenium/Browser setup is invalid. Cannot capture retention.")

    options = Options()
    # Standard headless options
    options.add_argument("--headless=new") # Use new headless mode
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    # Mimic browser environment
    options.add_argument("--lang=en-US,en")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36") # Example UA
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Try to reduce console noise

    # Browser path (use standard Linux path if others not found)
    browser_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium-browser") or shutil.which("chromium")
    if not browser_path and os.path.exists(CHROMIUM_PATH_STANDARD): browser_path = CHROMIUM_PATH_STANDARD
    if browser_path:
        options.binary_location = browser_path
        logger.info(f"Using browser binary at: {browser_path}")

    driver = None
    effective_duration = 0.0 # Initialize duration

    try:
        logger.info("Initializing Chrome driver for retention capture...")
        # Use webdriver-manager to find/install driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(45)
        logger.info("Driver initialized.")

        # Load cookies if requested
        if use_cookies:
             load_cookies(driver, "youtube_cookies.json") # Ignore return value, proceed anyway

        # Navigate to the video URL
        logger.info(f"Navigating to video URL: {video_url}")
        driver.get(video_url)
        logger.info("Waiting for video player element...")
        try:
             # Wait for essential player elements to load
             video_element = WebDriverWait(driver, 20).until(
                  EC.presence_of_element_located((By.CSS_SELECTOR, "video.html5-main-video"))
             )
             player_container = WebDriverWait(driver, 10).until(
                  EC.presence_of_element_located((By.CSS_SELECTOR, "#movie_player"))
             )
             logger.info("Video player elements found.")
             # Give extra time for UI animations/overlays to potentially settle
             time.sleep(5)
        except TimeoutException:
             logger.error("Timeout waiting for video player elements.")
             st.error("Failed to load video player interface in time.")
             try: driver.save_screenshot("debug_timeout_screenshot.png")
             except: pass
             raise TimeoutError("Video player did not load.")

        # Try to dismiss consent/popups (use more specific selectors if possible)
        try:
            consent_button = driver.find_element(By.XPATH, "//button[.//span[contains(text(), 'Accept all')]] | //button[contains(@aria-label, 'Accept')] | //button[contains(text(), 'Agree')]")
            if consent_button.is_displayed():
                consent_button.click()
                logger.info("Clicked a potential consent/agreement button.")
                time.sleep(2)
        except NoSuchElementException:
            logger.debug("No common consent button found.")
        except Exception as e:
            logger.warning(f"Could not click consent button: {e}")


        # --- Interaction Logic ---
        video_selector = "video.html5-main-video"
        try:
             # 1. Get initial duration
             initial_duration = driver.execute_script(f"return document.querySelector('{video_selector}')?.duration || 0;")
             logger.info(f"Initial JS duration: {initial_duration:.2f}s")
             if initial_duration <= 0: time.sleep(3); initial_duration = driver.execute_script(f"return document.querySelector('{video_selector}')?.duration || 0;") # Retry
             effective_duration = initial_duration if initial_duration > 0 else timestamp # Use input timestamp as fallback

             # 2. Pause video
             driver.execute_script(f"document.querySelector('{video_selector}')?.pause();")
             time.sleep(0.5)

             # 3. Seek near the end (but not exactly end)
             seek_target = effective_duration - 5 if effective_duration > 10 else effective_duration * 0.9
             seek_target = max(0, seek_target) # Ensure non-negative
             logger.info(f"Seeking video to: {seek_target:.2f} seconds (Effective duration: {effective_duration:.2f}s)")
             driver.execute_script(f"document.querySelector('{video_selector}').currentTime = {seek_target};")
             time.sleep(3) # Wait for seek & UI update

             # 4. Find progress bar (robust selector)
             progress_bar_container = WebDriverWait(player_container, 10).until(
                 EC.presence_of_element_located((By.CSS_SELECTOR, ".ytp-progress-bar-container"))
             )

             # 5. Hover over progress bar
             logger.info("Hovering over progress bar...")
             ActionChains(driver).move_to_element(progress_bar_container).perform()
             time.sleep(3) # Increased wait for graph stability

             # 6. Capture screenshot of the player container
             logger.info(f"Capturing player screenshot to: {output_path}")
             player_container.screenshot(output_path)
             logger.info("Screenshot captured.")

             # 7. Get final duration (might be more accurate)
             final_duration = driver.execute_script(f"return document.querySelector('{video_selector}')?.duration || 0;")
             if final_duration > 0: effective_duration = final_duration
             logger.info(f"Final JS duration: {final_duration:.2f}s. Using effective duration: {effective_duration:.2f}s")


        except (NoSuchElementException, TimeoutException) as e:
             logger.error(f"Selenium interaction failed: Element not found or timed out. Error: {e}")
             st.error(f"Error interacting with YouTube player: {e}. Retention graph might be missing.")
             try: driver.save_screenshot("debug_interaction_error.png")
             except: pass
             raise RuntimeError("Failed Selenium interaction for retention.") from e
        except Exception as e:
             logger.error(f"Unexpected error during Selenium interaction: {e}", exc_info=True)
             st.error(f"Unexpected error during retention analysis: {e}")
             try: driver.save_screenshot("debug_unexpected_error.png")
             except: pass
             raise # Re-raise

    finally:
        if driver:
            logger.info("Quitting Selenium driver (retention capture).")
            driver.quit()

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
         raise RuntimeError(f"Screenshot file '{output_path}' not created or is empty.")

    return effective_duration if effective_duration > 0 else 0.0 # Return 0 if duration unknown


def detect_retention_peaks(image_path, crop_ratio=0.15, height_threshold=100, distance=25, top_n=5):
    """Analyzes the retention graph screenshot to find peaks."""
    if not os.path.exists(image_path):
        logger.error(f"Retention screenshot file not found: {image_path}")
        raise FileNotFoundError(f"Screenshot file not found: {image_path}")

    logger.info(f"Analyzing retention peaks in: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image from {image_path}")
        raise ValueError(f"Failed to read image: {image_path}")

    height, width, _ = img.shape
    if height == 0 or width == 0:
         logger.error(f"Invalid image dimensions: {width}x{height}")
         raise ValueError("Image has zero width or height.")

    # --- ROI Definition ---
    roi_start_y = int(height * (1 - crop_ratio))
    # Add a small safety margin from the absolute bottom edge?
    roi_end_y = height - max(1, int(height * 0.01)) # Avoid absolute bottom edge
    roi_start_y = max(0, roi_start_y) # Ensure not negative
    if roi_start_y >= roi_end_y: # Ensure valid ROI height
         logger.error(f"Invalid ROI height after calculation: StartY={roi_start_y}, EndY={roi_end_y}. Image Height={height}")
         roi_start_y = max(0, height - 50) # Fallback to last 50 pixels
         roi_end_y = height

    roi = img[roi_start_y:roi_end_y, 0:width]
    logger.info(f"ROI: Y={roi_start_y}-{roi_end_y} (H:{roi.shape[0]}), W={width}")

    if roi.shape[0] == 0 or roi.shape[1] == 0:
         logger.error(f"ROI has invalid dimensions after cropping: {roi.shape}")
         st.warning("Could not define valid ROI for retention graph. Check screenshot.")
         return np.array([]), None, None, width, np.array([])

    # --- Image Processing ---
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding (Gaussian is common)
    # Adjust block size and constant C based on expected graph appearance
    binary_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 4) # Invert to make graph white on black bg, block 15, C=4
    logger.debug("Applied adaptive thresholding.")

    # Optional: Morphological operations to clean up noise or connect broken lines
    # kernel = np.ones((2,2),np.uint8)
    # binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel) # Remove small noise
    # binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel) # Close small gaps

    # --- Peak Detection ---
    col_sums = np.sum(binary_roi, axis=0) # Sum white pixels vertically
    if col_sums.size == 0:
         logger.error("Column sums array is empty after processing ROI.")
         return np.array([]), roi, binary_roi, width, np.array([])

    # Adjust threshold dynamically based on max sum? Or keep fixed?
    # effective_height_threshold = max(height_threshold, int(np.max(col_sums) * 0.2)) # Example: min 100 or 20% of max peak
    effective_height_threshold = height_threshold # Use parameter directly for now
    logger.info(f"Finding peaks: Height >= {effective_height_threshold}, Min Distance >= {distance}px")

    peaks, properties = find_peaks(col_sums, height=effective_height_threshold, distance=distance)

    if peaks.size == 0:
        logger.warning(f"No peaks found with initial settings. Try lowering threshold?")
        # Optional: Retry with lower threshold? Be careful not to find noise.
        # lower_threshold = max(50, height_threshold // 2)
        # peaks, properties = find_peaks(col_sums, height=lower_threshold, distance=distance)
        # if peaks.size > 0: logger.info(f"Found {len(peaks)} peaks with reduced threshold {lower_threshold}.")
        # else: logger.warning("Still no peaks with reduced threshold.")

    # Select top N peaks by height if needed
    if len(peaks) > top_n:
        logger.info(f"Found {len(peaks)} peaks, selecting top {top_n} by height.")
        peak_heights = properties["peak_heights"]
        top_indices = np.argsort(peak_heights)[-top_n:]
        top_peaks = np.sort(peaks[top_indices]) # Sort selected peaks by time (x-axis)
    else:
        top_peaks = peaks # Already sorted

    logger.info(f"Detected {len(top_peaks)} final peak(s) at x-coordinates: {top_peaks.tolist()}")
    return top_peaks, roi, binary_roi, width, col_sums


def capture_frame_at_time(video_url, target_time, output_path="frame_retention.png", use_cookies=True):
    """Captures a single frame screenshot at a specific video timestamp using Selenium."""
    if not check_selenium_setup():
         raise EnvironmentError("Selenium/Browser setup invalid. Cannot capture frame.")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en-US,en")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    browser_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium-browser") or shutil.which("chromium")
    if not browser_path and os.path.exists(CHROMIUM_PATH_STANDARD): browser_path = CHROMIUM_PATH_STANDARD
    if browser_path: options.binary_location = browser_path

    driver = None
    try:
        logger.info("Initializing Chrome driver for frame capture...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(45)
        logger.info("Driver initialized.")

        if use_cookies:
             load_cookies(driver, "youtube_cookies.json")

        logger.info(f"Navigating to video URL: {video_url}")
        driver.get(video_url)
        logger.info("Waiting for video element...")
        try:
             video_element = WebDriverWait(driver, 20).until(
                  EC.presence_of_element_located((By.CSS_SELECTOR, "video.html5-main-video"))
             )
             player_container = WebDriverWait(driver, 10).until(
                  EC.presence_of_element_located((By.CSS_SELECTOR, "#movie_player"))
             )
             logger.info("Video player elements found.")
             time.sleep(5)
        except TimeoutException:
             logger.error("Timeout waiting for video player for frame capture.")
             st.error("Failed to load video player interface for frame capture.")
             try: driver.save_screenshot("debug_frame_timeout.png"); logger.info("Saved debug screenshot.")
             except: pass
             raise TimeoutError("Video player did not load for frame capture.")

        # Dismiss consent
        try:
            consent_button = driver.find_element(By.XPATH, "//button[.//span[contains(text(), 'Accept all')]] | //button[contains(@aria-label, 'Accept')] | //button[contains(text(), 'Agree')]")
            if consent_button.is_displayed(): consent_button.click(); time.sleep(2)
        except Exception: pass


        # --- Frame Capture Logic ---
        video_selector = "video.html5-main-video"
        try:
             # 1. Pause
             driver.execute_script(f"document.querySelector('{video_selector}')?.pause();")
             time.sleep(0.5)

             # 2. Seek
             logger.info(f"Seeking video to target time: {target_time:.2f} seconds")
             # Ensure time is within valid range (JS handles <0, but check >duration?)
             # duration = driver.execute_script(f"return document.querySelector('{video_selector}')?.duration || 0;")
             # safe_target_time = min(target_time, duration - 0.1) if duration > 0 else target_time
             safe_target_time = max(0, target_time) # Ensure non-negative
             driver.execute_script(f"document.querySelector('{video_selector}').currentTime = {safe_target_time};")
             # Wait for seek & frame update. A small play/pause might be needed for accuracy in headless.
             time.sleep(1.5)
             logger.debug("Attempting brief play/pause cycle for frame accuracy...")
             driver.execute_script(f"document.querySelector('{video_selector}')?.play();")
             time.sleep(0.1) # Very short play
             driver.execute_script(f"document.querySelector('{video_selector}')?.pause();")
             time.sleep(0.5) # Wait for pause

             # 3. Capture screenshot of player
             logger.info(f"Capturing frame screenshot to: {output_path}")
             player_container.screenshot(output_path)
             logger.info("Frame screenshot captured.")

             if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                  raise RuntimeError("Screenshot failed - file empty or not created.")

        except (NoSuchElementException, TimeoutException) as e:
             logger.error(f"Selenium frame capture failed: {e}")
             st.error(f"Error interacting with YouTube player for frame capture: {e}.")
             try: driver.save_screenshot("debug_frame_interaction_error.png"); logger.info("Saved debug screenshot.")
             except: pass
             raise RuntimeError("Failed Selenium interaction for frame capture.") from e
        except Exception as e:
             logger.error(f"Unexpected error during Selenium frame capture: {e}", exc_info=True)
             st.error(f"Unexpected error during frame capture: {e}")
             try: driver.save_screenshot("debug_frame_unexpected_error.png"); logger.info("Saved debug screenshot.")
             except: pass
             raise

    finally:
        if driver:
            logger.info("Quitting Selenium driver (frame capture).")
            driver.quit()

    return output_path


def plot_brightness_profile(col_sums, peaks):
    """Generates a plot of the column brightness sums and detected peaks."""
    if col_sums is None or col_sums.size == 0:
         logger.warning("Cannot plot brightness profile: empty column sums.")
         return None

    buf = BytesIO()
    fig = None # Initialize fig to None for finally block
    try:
        fig, ax = plt.subplots(figsize=(10, 3)) # Use a consistent figure size
        ax.plot(col_sums, label="Retention Profile", color="#4285f4", linewidth=1.5)

        if peaks is not None and peaks.size > 0:
            # Ensure peaks are within the valid index range of col_sums
            valid_peaks_indices = peaks < len(col_sums)
            valid_peaks = peaks[valid_peaks_indices]
            if valid_peaks.size > 0:
                 ax.plot(valid_peaks, col_sums[valid_peaks], "x", label="Detected Peaks",
                         markersize=7, color="#ea4335", markeredgewidth=1.5)

        ax.set_xlabel("Horizontal Position (Pixels)")
        ax.set_ylabel("Relative Audience Retention") # More descriptive Y-axis
        ax.set_title("Audience Retention Graph Profile")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.5)
        # Improve y-axis ticks/labels (e.g., remove them or set specific range?)
        ax.set_yticks([]) # Remove Y-axis ticks as absolute brightness isn't meaningful
        ax.set_ylim(bottom=0) # Start y-axis at 0
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120) # Slightly higher DPI
        buf.seek(0)
        logger.info("Generated brightness profile plot.")
        return buf
    except Exception as e:
        logger.error(f"Error generating brightness profile plot: {e}", exc_info=True)
        return None # Return None on error
    finally:
        if fig: plt.close(fig) # Ensure figure is closed


def filter_transcript(transcript, target_time, window=5):
    """Extracts transcript text around a specific target time (+/- window seconds)."""
    if not transcript: return ""

    snippet_texts = []
    target_start = target_time - window
    target_end = target_time + window

    # Handle list of dicts format (YouTube or segmented Whisper)
    if isinstance(transcript, list) and transcript and isinstance(transcript[0], dict):
         # Check if timestamps ('start') are available
         if 'start' in transcript[0]:
             for seg in transcript:
                 try:
                     start = float(seg.get("start", -1))
                     if start == -1: continue # Skip segment if start time missing
                     duration = float(seg.get("duration", 2.0)) # Estimate duration if missing
                     end = start + duration
                     text = seg.get("text", "").strip()
                     if not text: continue

                     # Check for overlap: max(segment_start, target_start) < min(segment_end, target_end)
                     if max(start, target_start) < min(end, target_end):
                          snippet_texts.append(text)
                 except (ValueError, TypeError):
                     logger.warning(f"Skipping transcript segment due to invalid time/text: {seg}")
                     continue
             result = " ".join(snippet_texts).strip()

         else: # List of dicts but no 'start' key (e.g., simple Whisper output)
             logger.warning("Cannot filter transcript by time: Timestamps missing.")
             result = "(Transcript snippet unavailable due to missing timestamps)"

    elif isinstance(transcript, str): # Single string transcript
         logger.warning("Cannot filter transcript by time: Transcript is a single string.")
         result = "(Transcript snippet unavailable: Format lacks timestamps)"
    else:
         logger.warning(f"Unrecognized transcript format for filtering: {type(transcript)}")
         result = "(Transcript snippet unavailable: Unrecognized format)"


    # Limit result length and provide appropriate message if empty
    max_len = 500
    if not result:
         return "(No transcript text found near this time)"
    elif len(result) > max_len:
         return result[:max_len] + "..."
    else:
         return result


def check_ytdlp_installed():
    """Checks if yt-dlp command is available in the system PATH."""
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, check=True, timeout=5)
        logger.info(f"yt-dlp found, version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.warning("yt-dlp command not found in PATH.")
        return False
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning(f"yt-dlp --version command failed or timed out: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking yt-dlp: {e}", exc_info=True)
        return False


def download_video_snippet(video_url, start_time, duration=10, output_path="snippet.mp4"):
    """Downloads a short video snippet using yt-dlp and ffmpeg integration."""
    if not check_ytdlp_installed():
        st.error("yt-dlp is required to download video snippets.")
        raise EnvironmentError("yt-dlp not found.")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
         st.error("ffmpeg is required to extract video snippets.")
         raise EnvironmentError("ffmpeg not found or path invalid.")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try: os.makedirs(output_dir); logger.info(f"Created output directory: {output_dir}")
        except OSError as e: raise OSError(f"Failed to create output directory {output_dir}: {e}") from e

    # Use yt-dlp's download_ranges feature which leverages ffmpeg
    start_s = max(0, start_time)
    end_s = start_s + duration

    # Temporary filename template base (before final rename/remux)
    temp_output_base = output_path.replace('.mp4', '')

    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best', # Prefer mp4/m4a, reasonable quality
        'outtmpl': f'{temp_output_base}.%(ext)s', # Output with original extension first
        'quiet': True, 'no_warnings': True, 'noprogress': True,
        'ffmpeg_location': ffmpeg_path,
        'socket_timeout': 90, # Longer timeout for potential video download
        'ignoreconfig': True,
        # Use download_ranges for segment extraction
        'download_ranges': lambda info_dict, ydl: ydl.download_range_func(info_dict, [(start_s, end_s)]),
        # Postprocessor to ensure final output is MP4 and named correctly
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4', # Ensure MP4 container
        }],
        # Force filename after postprocessing to match desired output_path
        'outtmpl': {'default': output_path}, # Force final output name
        # Keep fragments might be needed if intermediate formats are used by yt-dlp
        'keepfragments': True, # Changed from keep_fragments
    }


    try:
        import yt_dlp
        logger.info(f"Downloading snippet for {video_url} [{start_s:.1f}s - {end_s:.1f}s] -> {output_path}...")
        st_info_placeholder = st.info(f"Downloading snippet ({duration}s)...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Ensure cookies are passed if needed? yt-dlp might need separate cookie handling.
            # ydl.params['cookiefile'] = 'youtube_cookies.txt' # Example if needed
            result_code = ydl.download([video_url]) # 0 on success

        # Check final output file after download and postprocessing attempt
        if result_code == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100: # Check size > 100 bytes
            st_info_placeholder.success(f"Snippet downloaded: {os.path.basename(output_path)}")
            logger.info(f"Snippet downloaded successfully to {output_path}")
            return output_path
        else:
            # Check for intermediate files if final failed
            found_files = [f for f in os.listdir(output_dir or '.') if f.startswith(os.path.basename(temp_output_base))]
            logger.error(f"yt-dlp process finished (code {result_code}) but expected output '{output_path}' not found or invalid. Temp files: {found_files}")
            st.error("Snippet download failed: Final file creation error.")
            raise RuntimeError("yt-dlp failed to create valid final snippet file.")

    except yt_dlp.utils.DownloadError as e:
         logger.error(f"yt-dlp DownloadError during snippet download: {e}")
         st.error(f"Error downloading snippet: {e}")
         raise RuntimeError(f"yt-dlp download error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during snippet download for {video_url}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during snippet download: {e}")
        raise # Re-raise the original exception
    finally:
        # Attempt cleanup of intermediate fragments if they exist
        try:
             base_name = os.path.basename(temp_output_base)
             for f in os.listdir(output_dir or '.'):
                 if f.startswith(base_name) and f != os.path.basename(output_path):
                      f_path = os.path.join(output_dir or '.', f)
                      logger.debug(f"Cleaning up intermediate file: {f_path}")
                      try: os.remove(f_path)
                      except OSError: pass
        except Exception as cleanup_e:
             logger.warning(f"Error during snippet fragment cleanup: {cleanup_e}")
        if 'st_info_placeholder' in locals(): st_info_placeholder.empty() # Clear message

# =============================================================================
# 14. UI Pages
# =============================================================================
def show_search_page():
    st.title("🚀 YouTube Niche Search")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Search Filters")

        # Folder Selection
        folders = load_channel_folders()
        available_folders = list(folders.keys())
        folder_choice = None
        if not available_folders:
             st.warning("No channel folders found.")
             st.caption("Use 'Manage Folders' below.")
        else:
             folder_choice = st.selectbox("Select Channel Folder", available_folders, index=0 if available_folders else -1, key="folder_choice_sb")

        # Timeframe
        timeframe_options = ["Last 24 hours", "Last 48 hours", "Last 4 days", "Last 7 days", "Last 15 days", "Last 28 days", "3 months", "Lifetime"]
        selected_timeframe = st.selectbox("Timeframe", timeframe_options, index=timeframe_options.index("3 months"), key="timeframe_sb")

        # Content Type Filter
        content_filter_options = ["Both", "Videos", "Shorts"]
        content_filter = st.selectbox("Filter Content Type", content_filter_options, index=0, key="content_filter_sb")

        # Minimum Outlier Score Filter
        min_outlier_score = st.number_input("Minimum Outlier Score", value=0.0, min_value=0.0, step=0.1, format="%.2f", key="min_outlier_sb",
                                            help="Filter results by performance relative to channel average (1.0 = avg, >1.0 = above avg). Based on weighted VPH & Engagement.")

        # Keyword Search
        search_query = st.text_input("Keyword Search (optional)", key="query_sb", placeholder="e.g., AI tools, market update")

        # Search Button
        search_button_pressed = st.button("🔍 Search Videos", key="search_button_sb", type="primary", use_container_width=True)

        st.divider()

        # --- Other Sidebar Actions ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Cache", help="Force fetch fresh data from YouTube API for the next search."):
                 try:
                      with sqlite3.connect(DB_PATH, timeout=10) as conn:
                          conn.execute("DELETE FROM youtube_cache")
                      # Clear relevant session state related to search results
                      for key in list(st.session_state.keys()):
                           if key.startswith('search_') or key.startswith('_search_'):
                                del st.session_state[key]
                      st.success("Cache cleared. Search again for fresh data.")
                      logger.info("SQLite cache cleared by user.")
                 except Exception as e:
                      st.error(f"Failed to clear cache: {e}")
                      logger.error(f"Failed to clear cache: {e}", exc_info=True)

        # Optionally add other buttons like export results etc. here

        st.divider()
        with st.expander("📂 Manage Folders", expanded=False):
             show_channel_folder_manager()

    # --- Main Page Content ---
    selected_channel_ids = []
    if folder_choice and folder_choice in folders:
        st.subheader(f"Folder: {folder_choice}")
        selected_channel_ids = [ch["channel_id"] for ch in folders[folder_choice]]
        if not selected_channel_ids:
             st.warning(f"Folder '{folder_choice}' is empty. Add channels via 'Manage Folders'.")
        else:
             # Show channels in folder (optional)
             with st.expander("Channels in this folder", expanded=False):
                  ch_names = [ch['channel_name'] for ch in folders[folder_choice]]
                  if len(ch_names) < 20:
                       st.write(" | ".join(ch_names))
                  else: # Use dataframe for many channels
                       st.dataframe({'Channel Name': ch_names}, hide_index=True, use_container_width=True)
    elif not available_folders:
         st.info("Create a channel folder in the sidebar to get started.")
    else:
         st.info("Select a channel folder from the sidebar.")


    # --- Search Execution Logic ---
    if search_button_pressed:
        if not folder_choice or not selected_channel_ids:
            st.error("Please select a folder containing channels before searching.")
        else:
            # Store search parameters to trigger fetch/display block
            st.session_state.search_params = {
                "query": search_query, "channel_ids": selected_channel_ids,
                "timeframe": selected_timeframe, "content_filter": content_filter,
                "min_outlier_score": min_outlier_score, "folder_choice": folder_choice
            }
            # Clear previous results to force refetch if params changed, or use cache if same
            if st.session_state.get('_search_params_for_results') != st.session_state.search_params:
                 if 'search_results' in st.session_state: del st.session_state['search_results']
                 logger.info("Search params changed or no previous results, will fetch/use cache.")
            else:
                 logger.info("Search params unchanged, will use existing results if available.")

            st.session_state.page = "search" # Ensure we stay/return to search page
            st.rerun()


    # --- Display Results Block ---
    # Check if search params are set (meaning a search was intended)
    if 'search_params' in st.session_state and st.session_state.get("page") == "search":
        search_params = st.session_state.search_params

        # Fetch results if not already in session state for current params
        if 'search_results' not in st.session_state or st.session_state.get('_search_params_for_results') != search_params:
            try:
                 # search_youtube handles caching internally
                 results_list = search_youtube(
                     search_params["query"], search_params["channel_ids"],
                     search_params["timeframe"], search_params["content_filter"]
                 )
                 st.session_state.search_results = results_list
                 st.session_state._search_params_for_results = search_params # Mark results as current
            except Exception as e:
                 st.error(f"An error occurred during search: {e}")
                 logger.error(f"Error during search_youtube call from UI: {e}", exc_info=True)
                 st.session_state.search_results = [] # Set empty results on error

        # --- Process and Display Results ---
        results_to_display = st.session_state.get('search_results', [])
        min_score_filter = search_params["min_outlier_score"]

        # Apply client-side outlier score filter
        filtered_results = []
        if min_score_filter > 0:
            initial_count = len(results_to_display)
            for r in results_to_display:
                 score = pd.to_numeric(r.get("outlier_score"), errors='coerce')
                 if score is not None and score >= min_score_filter:
                      filtered_results.append(r)
            filtered_count = len(filtered_results)
            if initial_count > 0 and filtered_count < initial_count:
                 st.caption(f"Filtered {initial_count - filtered_count} results below outlier score {min_score_filter:.2f}.")
        else:
            filtered_results = results_to_display


        if not filtered_results:
            # Show message only if search was actually performed
            if search_button_pressed or '_search_params_for_results' in st.session_state:
                st.info("No videos found matching all your criteria.")
        else:
            # --- Sorting ---
            sort_options_map = {
                "Outlier Score": "outlier_score", "Upload Date": "published_at",
                "Views": "views", "Effective VPH": "effective_vph",
                "VPH Ratio": "vph_ratio", "Comment/View Ratio": "cvr_float",
                "Like/View Ratio": "clr_float", "Comment Count": "comment_count",
                "Like Count": "like_count",
            }
            sort_label = st.selectbox("Sort results by:", list(sort_options_map.keys()), index=0, key="sort_by_select")
            sort_key = sort_options_map[sort_label]

            try:
                if sort_key == "published_at":
                    sorted_data = sorted(filtered_results, key=lambda x: x.get(sort_key, "1970-01-01T00:00:00Z"), reverse=True)
                else:
                    sorted_data = sorted(filtered_results, key=lambda x: pd.to_numeric(x.get(sort_key), errors='coerce') or 0, reverse=True)
            except Exception as e:
                 st.error(f"Error sorting results by {sort_label}: {e}")
                 logger.error(f"Sorting error: {e}", exc_info=True)
                 sorted_data = filtered_results # Fallback


            st.subheader(f"📊 Found {len(sorted_data)} Videos", anchor=False)
            st.caption(f"Sorted by {sort_label} (descending)")

            # --- Display Cards ---
            num_columns = 3 # Use 3 columns for card layout
            for i in range(0, len(sorted_data), num_columns):
                 cols = st.columns(num_columns)
                 row_items = sorted_data[i : i + num_columns]

                 for j, item in enumerate(row_items):
                      with cols[j]:
                           # Safely get display values
                           days_ago = int(round(item.get("days_since_published", 0)))
                           days_ago_text = "today" if days_ago == 0 else (f"yesterday" if days_ago == 1 else f"{days_ago} days ago")
                           score = pd.to_numeric(item.get('outlier_score'), errors='coerce')
                           score_str = f"{score:.2f}x" if score is not None else "N/A"

                           # Determine color based on score
                           if score is None: score_color = "#9aa0a6" # Grey
                           elif score >= 1.75: score_color = "#1e8e3e" # Dark Green (Strong Outlier)
                           elif score >= 1.15: score_color = "#34a853" # Green (Above Avg)
                           elif score >= 0.85: score_color = "#4285f4" # Blue (Avg)
                           elif score >= 0.5: score_color = "#fbbc04" # Orange (Below Avg)
                           else: score_color = "#ea4335" # Red (Significantly Below)

                           outlier_badge = f"""<span style="background-color:{score_color}; color:white; padding: 2px 7px; border-radius:10px; font-size:0.8em; font-weight:bold; display: inline-block; line-height: 1.2;">{score_str}</span>"""
                           watch_url = f"https://www.youtube.com/watch?v={item['video_id']}"
                           thumb_url = item.get('thumbnail', '') or 'https://placehold.co/320x180/grey/white?text=No+Thumb'


                           # --- Card HTML (Corrected) ---
                           card_style = """
                               border: 1px solid #dfe1e5; border-radius: 8px; padding: 12px; margin-bottom: 16px;
                               height: 430px; display: flex; flex-direction: column; background-color: #ffffff;
                               box-shadow: 0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15);
                               transition: box-shadow .2s ease-in-out;
                           """
                           card_html = f"""
                           <div style="{card_style}" onmouseover="this.style.boxShadow='0 1px 3px 0 rgba(60,64,67,.3), 0 4px 8px 3px rgba(60,64,67,.15)'" onmouseout="this.style.boxShadow='0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15)'">
                               <a href="{watch_url}" target="_blank" style="text-decoration: none; color: inherit; display:block;">
                                   <img src="{thumb_url}" alt="Thumbnail" style="width:100%; border-radius:4px; margin-bottom:10px; object-fit: cover; aspect-ratio: 16/9;" />
                                   <div title="{item.get('title', '')}" style="font-weight: 600; font-size: 1rem; line-height: 1.35; height: 48px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; margin-bottom: 6px; color: #202124;">
                                     {item.get('title', 'No Title')}
                                   </div>
                               </a>
                               <div style="font-size: 0.88rem; color: #5f6368; margin-bottom: 10px; height: 18px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                   {item.get('channel_name', 'Unknown Channel')}
                               </div>
                               <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                   <span style="font-weight: 500; color: #3c4043; font-size: 0.9rem;">{item.get('formatted_views', 'N/A')} views</span>
                                   {outlier_badge}
                               </div>
                               <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #5f6368; margin-bottom: 12px;">
                                   <span>{item.get('vph_display', 'N/A')}</span>
                                   <span>{days_ago_text}</span>
                               </div>
                               <div style="margin-top: auto;"> {{/* Button pushed to bottom */}} </div>
                           </div>
                           """
                           # Render card and button separately for interactivity
                           st.markdown(card_html, unsafe_allow_html=True)
                           # Place button seemingly inside the card's bottom area
                           button_key = f"view_details_{item['video_id']}"
                           if st.button("View Details", key=button_key, use_container_width=True):
                                st.session_state.selected_video_id = item["video_id"]
                                st.session_state.selected_video_title = item["title"]
                                # Store full item data for details page
                                st.session_state.selected_video_data = item
                                st.session_state.page = "details"
                                st.rerun()


def show_details_page():
    """Displays detailed analysis for the selected video."""
    # --- Retrieve selected video data ---
    video_id = st.session_state.get("selected_video_id")
    video_title = st.session_state.get("selected_video_title")
    # Get the full data dictionary stored during selection
    video_data = st.session_state.get("selected_video_data")

    # --- Navigation and Validation ---
    if st.button("⬅️ Back to Search Results", key="details_back_button"):
        st.session_state.page = "search"
        # Optionally clear selected video state to avoid showing stale data if user navigates back differently
        # if "selected_video_id" in st.session_state: del st.session_state["selected_video_id"]
        # if "selected_video_data" in st.session_state: del st.session_state["selected_video_data"]
        st.rerun()

    if not video_id or not video_title or not video_data:
        st.error("Video data not found. Please return to the search page and select a video.")
        return # Stop execution if no valid data

    st.header(f"🔬 Analysis: {video_title}")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    st.caption(f"Video ID: {video_id} | [Watch on YouTube]({video_url})")

    # --- Display Basic Info & Metrics ---
    col_thumb, col_metrics = st.columns([1, 2])
    with col_thumb:
         thumb_url = video_data.get('thumbnail', '') or 'https://placehold.co/320x180/grey/white?text=No+Thumb'
         st.image(thumb_url, use_column_width=True)
         st.caption(f"Channel: **{video_data.get('channel_name', 'N/A')}**")
         duration_sec = video_data.get('duration_seconds', 0)
         duration_min, duration_s = divmod(int(duration_sec), 60)
         st.caption(f"Duration: **{duration_min}m {duration_s}s** ({video_data.get('content_category', 'N/A')})")
         st.caption(f"Published: **{video_data.get('publish_date', 'N/A')}** ({int(round(video_data.get('days_since_published', 0)))} days ago)")

    with col_metrics:
        st.subheader("Performance Snapshot", anchor=False)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Views", video_data.get('formatted_views', 'N/A'))
        m_col2.metric("Likes 👍", format_number(video_data.get('like_count', 0)))
        m_col3.metric("Comments 💬", format_number(video_data.get('comment_count', 0)))

        # Display Outlier Score prominently
        score = pd.to_numeric(video_data.get('outlier_score'), errors='coerce')
        score_disp = f"{score:.2f}x" if score is not None else "N/A"
        delta_disp = f"{(score - 1.0):+.1%}" if score is not None and score != 1.0 else None
        st.metric("Outlier Score (vs Channel Avg)", score_disp, delta=delta_disp,
                  help="Weighted performance (VPH & Engagement) vs channel average. 1.0x = Average.")

        # Other relevant metrics
        st.markdown("**Performance Ratios (vs Channel Avg):**")
        vph_r = pd.to_numeric(video_data.get('vph_ratio'), errors='coerce')
        eng_r = pd.to_numeric(video_data.get('engagement_ratio'), errors='coerce')
        cvr_r = pd.to_numeric(video_data.get('outlier_cvr'), errors='coerce')
        clr_r = pd.to_numeric(video_data.get('outlier_clr'), errors='coerce')

        st.text(f" • VPH Ratio: {vph_r:.2f}x" if vph_r is not None else " • VPH Ratio: N/A")
        st.text(f" • Engagement Ratio: {eng_r:.2f}x" if eng_r is not None else " • Engagement Ratio: N/A")
        st.text(f" • Comment/View Ratio: {cvr_r:.2f}x" if cvr_r is not None else " • Comment/View Ratio: N/A")
        st.text(f" • Like/View Ratio: {clr_r:.2f}x" if clr_r is not None else " • Like/View Ratio: N/A")

        st.caption(f"Raw C/V: {video_data.get('comment_to_view_ratio', 'N/A')}, Raw L/V: {video_data.get('like_to_view_ratio', 'N/A')}")
        st.caption(f"Effective VPH: {video_data.get('vph_display', 'N/A')}")

    st.divider()

    # --- Tabs for Analysis Sections ---
    tab_comments, tab_script, tab_retention = st.tabs(["💬 Comments", "📜 Script", "📈 Retention"])

    # --- Comments Tab ---
    with tab_comments:
        st.subheader("Comments Analysis", anchor=False)
        comments_key = f"comments_{video_id}"
        analysis_key = f"analysis_{video_id}"

        # Fetch comments if not cached in session
        if comments_key not in st.session_state:
            with st.spinner(f"Fetching comments for {video_id}..."):
                st.session_state[comments_key] = get_video_comments(video_id, max_comments=100)

        comments = st.session_state.get(comments_key, [])

        if not comments:
            st.info("No comments found or comments are disabled.")
        else:
            st.caption(f"Analysis based on ~{len(comments)} comments (ordered by relevance).")

            # Perform AI analysis if not cached
            if analysis_key not in st.session_state:
                with st.spinner("Analyzing comments with AI..."):
                    st.session_state[analysis_key] = analyze_comments(comments)

            analysis_result = st.session_state.get(analysis_key, "Analysis failed or not performed.")
            st.markdown("**AI Comments Summary:**")
            st.markdown(analysis_result) # Display analysis text (uses markdown formatting from prompt)

            # Expander for top comments
            with st.expander("Show Top 5 Comments (by Likes)"):
                 top_5 = sorted([c for c in comments if c.get('likeCount', 0) > 0], key=lambda c: c.get('likeCount', 0), reverse=True)[:5]
                 if not top_5:
                      st.write("_No comments with likes found._")
                 else:
                      for i, c in enumerate(top_5):
                           st.markdown(f"**{i+1}. {c.get('likeCount', 0)} 👍** by *{c.get('author', 'Anon')}*")
                           # Use st.text or st.caption for comment text to prevent markdown injection issues
                           st.caption(f'"{c.get("text", "")[:300]}"' + ('...' if len(c.get("text", "")) > 300 else ''))
                           if i < len(top_5) - 1: st.markdown("---")


    # --- Script Tab ---
    with tab_script:
        st.subheader("Script Analysis", anchor=False)
        total_duration = video_data.get('duration_seconds', 0)
        is_short = video_data.get('content_category') == "Short"

        # Get transcript (uses fallback & session caching)
        transcript_key = f"transcript_data_{video_id}"
        transcript_source_key = f"transcript_source_{video_id}"
        if transcript_key not in st.session_state:
             with st.spinner("Fetching transcript (may use Whisper)..."):
                  get_transcript_with_fallback(video_id) # Updates session state

        transcript = st.session_state.get(transcript_key)
        source = st.session_state.get(transcript_source_key)

        if not transcript:
            st.warning("Transcript is unavailable for this video.")
        else:
            st.caption(f"Transcript source: {source or 'Unknown'}")
            full_script_text = " ".join([seg.get("text", "") for seg in transcript])

            if is_short:
                st.markdown("**Short Video Script Summary (AI):**")
                summary_key_short = f"summary_short_{video_id}"
                if summary_key_short not in st.session_state:
                     with st.spinner("Summarizing short's script..."):
                          st.session_state[summary_key_short] = summarize_script(full_script_text)
                st.write(st.session_state.get(summary_key_short, "*Summary failed.*"))
                with st.expander("Show Full Script (Short)", expanded=False):
                     st.text_area("", full_script_text, height=200, key="short_script_area")
            else: # Long-form
                st.markdown("**Intro & Outro Snippets & Summary (AI):**")
                intro_outro_key = f"intro_outro_data_{video_id}" # Changed key name
                if intro_outro_key not in st.session_state:
                     with st.spinner("Analyzing intro/outro..."):
                          intro_txt, outro_txt = get_intro_outro_transcript(video_id, total_duration)
                          summary_txt, _ = summarize_intro_outro(intro_txt, outro_txt)
                          st.session_state[intro_outro_key] = {
                               "intro_raw": intro_txt, "outro_raw": outro_txt, "summary": summary_txt
                          }
                intro_outro_data = st.session_state[intro_outro_key]
                st.markdown(intro_outro_data.get("summary", "*Summary unavailable.*"))

                with st.expander("Show Raw Intro/Outro Snippets"):
                     st.markdown("**Intro Snippet:**")
                     st.caption(intro_outro_data.get("intro_raw") or "*Not available*")
                     st.markdown("**Outro Snippet:**")
                     st.caption(intro_outro_data.get("outro_raw") or "*Not available*")

                # Button for full summary (long form only)
                summary_key_long = f"summary_long_{video_id}"
                if st.button("Summarize Full Script (AI)", key="summarize_full_btn"):
                     with st.spinner("Summarizing full script..."):
                          st.session_state[summary_key_long] = summarize_script(full_script_text)
                     st.rerun() # Rerun to display the summary

                if summary_key_long in st.session_state:
                     st.markdown("**Full Script Summary (AI):**")
                     st.write(st.session_state[summary_key_long])

                with st.expander("Show Full Script (Long-form)", expanded=False):
                     st.text_area("", full_script_text, height=300, key="long_script_area")


    # --- Retention Tab ---
    with tab_retention:
        st.subheader("Retention Analysis (Experimental)", anchor=False)
        retention_state_prefix = f"retention_{video_id}_" # Prefix for session state keys

        # Eligibility Checks
        published_at_iso = video_data.get("published_at")
        can_run_retention = False
        eligibility_msg = ""
        if is_short:
            eligibility_msg = "ℹ️ Retention analysis is typically less meaningful for Shorts."
        elif not published_at_iso:
            eligibility_msg = "⚠️ Cannot determine video age (publish date missing)."
        else:
            try:
                published_dt = datetime.fromisoformat(published_at_iso.replace('Z', '+00:00'))
                age_delta = datetime.now(timezone.utc) - published_dt
                if age_delta < timedelta(days=2):
                    eligibility_msg = f"ℹ️ Video is too recent ({age_delta.days} days old). Retention data needs ~2+ days to populate."
                elif not os.path.exists("youtube_cookies.json"):
                     eligibility_msg = "⚠️ `youtube_cookies.json` not found. Login cookies required."
                     st.markdown("See [cookie instructions](https://github.com/demberto/youtube-studio-api/blob/main/youtube_studio_api/auth.py#L13) (uses browser extension).", unsafe_allow_html=True)
                elif not check_selenium_setup(): # Checks browser/driver
                     eligibility_msg = "⚠️ Browser/Driver setup failed. Check logs/environment."
                else:
                     can_run_retention = True # Eligible
            except Exception as e:
                logger.error(f"Error checking retention eligibility: {e}", exc_info=True)
                eligibility_msg = f"⚠️ Error checking eligibility: {e}"

        if not can_run_retention:
            st.info(eligibility_msg)
        else: # Eligible, show button or results
            # --- Control and Execution ---
            analysis_running = st.session_state.get(retention_state_prefix + 'running', False)
            analysis_done = st.session_state.get(retention_state_prefix + 'done', False)
            analysis_error = st.session_state.get(retention_state_prefix + 'error', None)

            if not analysis_running and not analysis_done:
                if st.button("▶️ Run Retention Analysis", key="run_retention_btn"):
                    # Clear previous state for this video before starting
                    for key in list(st.session_state.keys()):
                         if key.startswith(retention_state_prefix): del st.session_state[key]
                    st.session_state[retention_state_prefix + 'running'] = True
                    st.rerun()

            # --- Analysis Execution Block ---
            if analysis_running:
                st.info("Retention analysis in progress...")
                progress_bar = st.progress(0, text="Starting analysis...")
                analysis_success = False
                error_message = None
                try:
                    # 1. Capture Screenshot
                    progress_bar.progress(10, text="Initializing browser...")
                    screenshot_key = retention_state_prefix + 'screenshot_path'
                    duration_key = retention_state_prefix + 'duration'
                    temp_retention_dir = tempfile.mkdtemp(prefix=f"retention_{video_id}_")
                    screenshot_path = os.path.join(temp_retention_dir, "player_retention.png")
                    st.session_state[retention_state_prefix + 'temp_dir'] = temp_retention_dir # Store dir for cleanup

                    progress_bar.progress(25, text="Capturing retention graph...")
                    base_ts = total_duration if total_duration > 0 else 120
                    vid_duration = capture_player_screenshot_with_hover(video_url, timestamp=base_ts, output_path=screenshot_path, use_cookies=True)

                    if not os.path.exists(screenshot_path): raise RuntimeError("Screenshot not saved.")
                    st.session_state[screenshot_key] = screenshot_path
                    st.session_state[duration_key] = vid_duration
                    effective_duration = vid_duration if vid_duration > 0 else (total_duration if total_duration > 0 else 1)
                    progress_bar.progress(50, text="Analyzing peaks...")

                    # 2. Detect Peaks
                    peaks, roi, bin_roi, roi_w, col_sums = detect_retention_peaks(screenshot_path)
                    st.session_state[retention_state_prefix + 'peaks'] = peaks
                    st.session_state[retention_state_prefix + 'roi_width'] = roi_w
                    st.session_state[retention_state_prefix + 'col_sums'] = col_sums

                    progress_bar.progress(70, text="Processing peak details...")

                    # 3. Process Peaks (Frames, Snippets - store paths, don't display yet)
                    peak_details = {}
                    if len(peaks) > 0 and roi_w > 0:
                         # Get transcript now if needed for filtering later
                         if transcript is None and transcript_key in st.session_state:
                             transcript = st.session_state.get(transcript_key)

                         for idx, peak_x in enumerate(peaks):
                             peak_time_s = (peak_x / roi_w) * effective_duration
                             peak_id = f"peak_{idx+1}"
                             peak_details[peak_id] = {'time_sec': peak_time_s}

                             # Capture Frame
                             frame_path = os.path.join(temp_retention_dir, f"{peak_id}_frame.png")
                             capture_frame_at_time(video_url, target_time=peak_time_s, output_path=frame_path, use_cookies=True)
                             if os.path.exists(frame_path): peak_details[peak_id]['frame_path'] = frame_path

                             # Transcript Snippet (text only)
                             if transcript:
                                  peak_details[peak_id]['transcript_text'] = filter_transcript(transcript, peak_time_s, window=5)

                             # Video Snippet Path (download happens later if displayed)
                             peak_details[peak_id]['snippet_start'] = max(0, peak_time_s - 4) # Default 8s snippet centered
                             peak_details[peak_id]['snippet_path'] = os.path.join(temp_retention_dir, f"{peak_id}_snippet.mp4")

                    st.session_state[retention_state_prefix + 'peak_details'] = peak_details
                    analysis_success = True
                    progress_bar.progress(100, text="Analysis complete!")
                    time.sleep(1) # Show complete message briefly

                except Exception as e:
                     error_message = f"Retention Analysis Failed: {e}"
                     logger.error(f"Retention analysis failed for {video_id}: {e}", exc_info=True)
                     st.error(error_message)
                finally:
                     # Update state regardless of success/failure
                     st.session_state[retention_state_prefix + 'running'] = False
                     st.session_state[retention_state_prefix + 'done'] = analysis_success
                     st.session_state[retention_state_prefix + 'error'] = error_message if not analysis_success else None
                     progress_bar.empty() # Clear progress bar
                     st.rerun() # Rerun to display results or error

            # --- Display Results Block ---
            elif analysis_done:
                 st.success("Retention analysis completed.")
                 screenshot_path = st.session_state.get(retention_state_prefix + 'screenshot_path')
                 peaks = st.session_state.get(retention_state_prefix + 'peaks')
                 col_sums = st.session_state.get(retention_state_prefix + 'col_sums')
                 peak_details = st.session_state.get(retention_state_prefix + 'peak_details', {})

                 if screenshot_path and os.path.exists(screenshot_path):
                      st.image(screenshot_path, caption="Retention Graph Screenshot")
                 else:
                      st.warning("Retention screenshot missing.")

                 if peaks is not None:
                      st.write(f"Detected {len(peaks)} peak(s).")
                      profile_plot_buf = plot_brightness_profile(col_sums, peaks)
                      if profile_plot_buf:
                           st.image(profile_plot_buf, caption="Brightness Profile")

                      if peak_details:
                           st.markdown("**Peak Details:**")
                           # Allow user to adjust snippet duration for display/download
                           snippet_dur = st.slider("Video Snippet Duration (sec)", 4, 20, 8, 2, key="ret_snippet_dur")

                           for peak_id, details in peak_details.items():
                                peak_time = details.get('time_sec', 0)
                                st.markdown(f"--- \n**{peak_id.replace('_', ' ').title()} at ~ {peak_time:.1f}s**")

                                col_frame, col_info = st.columns([1, 1])
                                with col_frame:
                                     frame_path = details.get('frame_path')
                                     if frame_path and os.path.exists(frame_path):
                                          st.image(frame_path, caption=f"Frame at {peak_time:.1f}s")
                                     else: st.caption("_Frame capture failed_")

                                with col_info:
                                     st.markdown("**Transcript Snippet:**")
                                     st.caption(details.get('transcript_text') or "_Not available_")

                                     # Video Snippet Download/Display
                                     snippet_out_path = details.get('snippet_path')
                                     if snippet_out_path:
                                          # Check if already downloaded (use existence as check)
                                          snippet_key_dl = f"{retention_state_prefix}{peak_id}_snippet_downloaded" # Simpler key
                                          if st.session_state.get(snippet_key_dl) and os.path.exists(snippet_out_path):
                                               try:
                                                    st.video(snippet_out_path)
                                               except Exception as video_e:
                                                    st.error(f"Could not display video snippet: {video_e}")
                                                    logger.error(f"Error displaying video {snippet_out_path}: {video_e}")
                                          else:
                                               if st.button(f"Load Snippet ({snippet_dur}s)", key=f"load_snippet_{peak_id}"):
                                                    if check_ytdlp_installed():
                                                         with st.spinner("Downloading snippet..."):
                                                             try:
                                                                  start_t = max(0, peak_time - snippet_dur / 2)
                                                                  download_video_snippet(video_url, start_time=start_t, duration=snippet_dur, output_path=snippet_out_path)
                                                                  st.session_state[snippet_key_dl] = True # Mark as downloaded
                                                                  st.rerun() # Rerun to display video
                                                             except Exception as snip_e:
                                                                  st.error(f"Snippet download failed: {snip_e}")
                                                    else:
                                                         st.error("yt-dlp not found, cannot download snippet.")


                 if st.button("Clear Retention Results", key="clear_retention_btn"):
                      # Clean up files and session state for this video's retention
                      temp_dir = st.session_state.get(retention_state_prefix + 'temp_dir')
                      if temp_dir and os.path.isdir(temp_dir):
                           try: shutil.rmtree(temp_dir)
                           except Exception as e: logger.error(f"Error cleaning retention temp dir {temp_dir}: {e}")
                      for key in list(st.session_state.keys()):
                           if key.startswith(retention_state_prefix): del st.session_state[key]
                      st.rerun()

            elif analysis_error:
                 st.error(f"Analysis failed: {analysis_error}")
                 if st.button("Retry Analysis?", key="retry_retention_btn"):
                      # Clear error state and trigger again
                      for key in list(st.session_state.keys()):
                           if key.startswith(retention_state_prefix): del st.session_state[key]
                      st.session_state[retention_state_prefix + 'running'] = True
                      st.rerun()


def main():
    """Main function to run the Streamlit application."""
    # Initialize database on first load if not already done
    if 'db_initialized' not in st.session_state:
         init_db(DB_PATH)
         st.session_state.db_initialized = True

    # Set page configuration (should ideally be the first Streamlit command)
    # Moved here for clarity, though might cause rerun issues if not first. Best practice: put at top level.
    # st.set_page_config(page_title="YouTube Niche Analysis", layout="wide", page_icon="📊")

    # Simple Page Navigation
    if "page" not in st.session_state:
        st.session_state.page = "search"

    current_page = st.session_state.get("page")

    if current_page == "search":
        show_search_page()
    elif current_page == "details":
        show_details_page()
    else: # Default or invalid state -> search page
        logger.warning(f"Invalid page state '{current_page}', defaulting to search.")
        st.session_state.page = "search"
        show_search_page()

# Application Shutdown Hook
def app_shutdown():
     logger.info("======== Application Shutting Down ========")
     # Add any other cleanup tasks here (e.g., close DB connections if not using 'with')

if __name__ == "__main__":
    # --- Page Config (BEST place for it) ---
    st.set_page_config(page_title="YouTube Niche Analysis", layout="wide", page_icon="📊")
    # --- Main Execution ---
    try:
        main()
    except Exception as e:
        # Catch unexpected errors in the main flow and log them
        logger.critical(f"FATAL: Unhandled exception in Streamlit main execution: {e}", exc_info=True)
        # Display a user-friendly error message
        st.error(f"A critical application error occurred: {e}. Please check logs or contact support.")
    finally:
         # Register the shutdown function to be called when the script exits
         atexit.register(app_shutdown)


