import requests  # <-- ADD THIS
import json
import streamlit as st

# --- File paths and URLs ---
# This is the URL for the header animation you liked
LOTTIE_URL_HEADER = "https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json"

# These are the local files for the status box
PROCESSING_ANIMATION_PATH = "processing_animation.json"
COMPLETE_ANIMATION_PATH = "complete_animation.json"


@st.cache_data
def load_lottieurl(url: str):
    """Fetches a Lottie JSON from a URL with error handling and caching."""
    try:
        r = requests.get(url)
        r.raise_for_status() # Raises an error for bad responses (4xx or 5xx)
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading Lottie animation: {e}")
        return None
    except requests.exceptions.JSONDecodeError:
        st.error("Error: Failed to decode Lottie JSON. Is this a valid URL?")
        return None

@st.cache_data
def load_lottielocal(filepath: str):
    """Loads a Lottie JSON from a local file path."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Animation file '{filepath}' not found. Please download it.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Failed to decode Lottie JSON from {filepath}.")
        return None