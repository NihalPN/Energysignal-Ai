import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def async_telegram_push(message: str):
    """Phase 4: True asynchronous, non-blocking push notification."""
    if not TOKEN or not CHAT_ID:
        print("Telegram credentials missing in.env file.")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}

    # httpx allows the network request to run in the background
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=5.0)
            response.raise_for_status()
        except Exception as e:
            print(f"Async Alert Failed: {e}")

def send_alert(message: str):
    """Wrapper to trigger the async task from synchronous scripts."""
    asyncio.run(async_telegram_push(message))
