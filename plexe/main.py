"""
Application entry point for using the plexe package as a conversational agent.
"""

import threading
import time
import logging
import os

import uvicorn
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Launch the Plexe assistant with a web UI."""
    load_dotenv()
    host = "127.0.0.1"
    port = int(os.environ.get("BACKEND_PORT", 8000))

    # If the user exported GEMINI_API_KEY but not GOOGLE_API_KEY, map it so
    # litellm/Google provider can pick it up (common naming mismatch).
    # This does not persist anything to disk and only affects the current process.
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_API_KEY"] = gemini_key
        logging.getLogger(__name__).info("Mapped GEMINI_API_KEY -> GOOGLE_API_KEY for this process")

    # Configure uvicorn to run in a thread
    config = uvicorn.Config("plexe.server:app", host=host, port=port, log_level="info", reload=False)
    server = uvicorn.Server(config)

    # Start server in a background thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Give the server a moment to start
    time.sleep(4)

    # Open the browser
    url = f"http://{host}:{port}"
    logger.info(f"Opening browser at {url}")
    # webbrowser.open(url)

    # Keep the main thread alive
    try:
        logger.info("Plexe Assistant is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down Plexe Assistant...")
        server.should_exit = True


if __name__ == "__main__":
    main()
