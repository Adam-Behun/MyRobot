import asyncio
import os
import logging
from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_livekit_direct():
    """Test direct LiveKit connection to see if we can receive audio"""
    
    # Get a fresh session from your server first
    session_id = "test-session"
    room_name = f"healthcare-ai-session-{session_id}"
    
    # Create room and tokens (same logic as your app.py)
    lk_api = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL").replace("wss://", "https://"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET")
    )
    
    try:
        room = await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        logger.info(f"Created room: {room_name}")
        
        # Generate bot token
        bot_token = api.AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET")
        ).with_identity("test-bot") \
         .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True
            )
        ).to_jwt()
        
        await lk_api.aclose()
        
        logger.info(f"Bot token: {bot_token[:50]}...")
        
        # Now test direct RTC connection
        room_client = rtc.Room()
        
        @room_client.on("participant_connected")
        def on_participant_connected(participant):
            logger.info(f"🟢 Participant connected: {participant.identity}")
        
        @room_client.on("track_subscribed") 
        def on_track_subscribed(track, publication, participant):
            logger.info(f"🎵 Track subscribed: {track.kind} from {participant.identity}")
            
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"🔊 Audio track received! Setting up audio stream...")
                
                # Create audio stream to receive frames
                audio_stream = rtc.AudioStream(track)
                
                async def process_audio():
                    logger.info("🎤 Starting audio processing...")
                    async for frame in audio_stream:
                        logger.info(f"📥 Received audio frame: {len(frame.data)} bytes, {frame.samples_per_channel} samples")
                        # This proves we're getting audio data
                
                asyncio.create_task(process_audio())
        
        # Connect to room
        logger.info(f"Connecting to: {os.getenv('LIVEKIT_URL')}")
        await room_client.connect(os.getenv("LIVEKIT_URL"), bot_token)
        logger.info("✅ Connected to LiveKit room")
        
        # Wait for connections
        logger.info("🔄 Waiting for user to connect and start speaking...")
        logger.info("👉 Now connect your web client and speak!")
        
        # Keep alive for testing
        await asyncio.sleep(60)
        
    except Exception as e:
        logger.error(f"❌ LiveKit test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await room_client.disconnect()
        logger.info("Disconnected from test")

if __name__ == "__main__":
    asyncio.run(test_livekit_direct())