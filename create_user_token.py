import os
import datetime
from dotenv import load_dotenv
from livekit import api

load_dotenv()

def create_user_token():
    """Create a user token for the test room"""
    
    room_name = "healthcare-ai-session-test-session"
    
    # Generate user token (same logic as app.py)
    user_token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    ).with_identity("user") \
     .with_ttl(datetime.timedelta(seconds=3600)) \
     .with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True
        )
    ).to_jwt()
    
    print("🎫 User Token for Test Room:")
    print(f"Room: {room_name}")
    print(f"URL: {os.getenv('LIVEKIT_URL')}")
    print(f"Token: {user_token}")
    print()
    print("👉 Copy this token and paste it in your web client!")

if __name__ == "__main__":
    create_user_token()