import asyncio
from livekit import rtc

async def join_room():
    room = rtc.Room()
    await room.connect(
        url="wss://myrobot-ndwwn90o.livekit.cloud?room=healthcare-ai-session-1f2e0376",  # Use your room_url
        token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ2aWRlbyI6eyJyb29tSm9pbiI6dHJ1ZSwicm9vbSI6ImhlYWx0aGNhcmUtYWktc2Vzc2lvbi0xZjJlMDM3NiIsImNhblB1Ymxpc2giOnRydWUsImNhblN1YnNjcmliZSI6dHJ1ZSwiY2FuUHVibGlzaERhdGEiOnRydWV9LCJzdWIiOiJ1c2VyIiwiaXNzIjoiQVBJb3JLR0RqNjJNSndZIiwibmJmIjoxNzUzNDc0OTQ5LCJleHAiOjE3NTM0Nzg1NDl9._rXKI8wzH35cg5xsG0d3pW4LMoYMiyPT4RIEKXRWSlk"  # Use your user_token
    )
    print("Connected to room")
    
    # Publish local microphone audio
    mic_source = rtc.MicrophoneSource()
    audio_track = rtc.LocalAudioTrack.create_track(mic_source, name="mic")
    await room.local_participant.publish_track(audio_track)
    
    # Keep the connection open (adjust sleep duration as needed)
    await asyncio.sleep(3600)
    await room.disconnect()

asyncio.run(join_room())