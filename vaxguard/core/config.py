import os
from dotenv import load_dotenv

load_dotenv()

# Load up to 5 Groq keys from environment variables
GROQ_API_KEYS = []
for i in range(1, 6):
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if key:
        GROQ_API_KEYS.append(key)

if not GROQ_API_KEYS:
    # Fallback to standard GROQ_API_KEY if specific numbered ones aren't found
    default_key = os.getenv("GROQ_API_KEY")
    if default_key:
        GROQ_API_KEYS.append(default_key)
