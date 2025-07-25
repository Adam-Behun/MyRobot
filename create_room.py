import os
import asyncio
import datetime
from dotenv import load_dotenv
from livekit import api

load_dotenv()

async def main():
    # Initialize LiveKit API client
    lk_api = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL").replace("wss://", "https://"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET")
    )

    # Create a room
    room_name = "healthcare-ai-session"  # Customize as needed
    room = await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
    print(f"Room created: {room.sid}")
    print(f"Room URL: {os.getenv('LIVEKIT_URL')}?room={room_name}")

    # Generate a token for the bot (the AI agent)
    bot_token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    ).with_identity("healthcare-bot") \
     .with_ttl(datetime.timedelta(seconds=3600)) \
     .with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True
        )
    ).to_jwt()
    print(f"Bot Token: {bot_token}")

    # Generate a token for yourself (the user/client)
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
    print(f"User Token: {user_token}")

    # Close the API client
    await lk_api.aclose()

asyncio.run(main())