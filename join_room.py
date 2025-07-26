import asyncio
from livekit import rtc

async def join_room():
    room = rtc.Room()
    
    # Event handlers
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        print(f"Participant connected: {participant.identity}")
    
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        print(f"Track subscribed from {participant.identity}: {track.kind}")
        if isinstance(track, rtc.RemoteAudioTrack):
            print("Audio track received - you should hear the AI agent")
    
    # Connect to room
    await room.connect(
        url="wss://myrobot-ndwwn90o.livekit.cloud?room=healthcare-ai-session-8bbbd917",
        token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ2aWRlbyI6eyJyb29tSm9pbiI6dHJ1ZSwicm9vbSI6ImhlYWx0aGNhcmUtYWktc2Vzc2lvbi04YmJiZDkxNyIsImNhblB1Ymxpc2giOnRydWUsImNhblN1YnNjcmliZSI6dHJ1ZSwiY2FuUHVibGlzaERhdGEiOnRydWV9LCJzdWIiOiJ1c2VyIiwiaXNzIjoiQVBJb3JLR0RqNjJNSndZIiwibmJmIjoxNzUzNDc2MDI0LCJleHAiOjE3NTM0Nzk2MjR9.r9d5649iRgdCV9i5kU7LD3kxAroYrWUSGkkwlAyo2S8"
    )
    print("Connected to room")
    
    # Publish microphone audio
    try:
        # Create microphone audio source directly
        mic_source = rtc.MicrophoneAudioSource(sample_rate=48000, num_channels=1)
        local_audio_track = rtc.LocalAudioTrack.create_audio_track("microphone", mic_source)
        
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        
        publication = await room.local_participant.publish_track(local_audio_track, options)
        print(f"Published microphone track: {publication.sid}")
        print("Microphone is active")
        
    except Exception as e:
        print(f"Error setting up microphone: {e}")
        # Try alternative approach
        try:
            local_audio_track = rtc.LocalAudioTrack.create_microphone_track("microphone")
            publication = await room.local_participant.publish_track(local_audio_track)
            print(f"Published microphone track (alternative): {publication.sid}")
        except Exception as e2:
            print(f"Alternative microphone setup also failed: {e2}")
    
    print("Ready - speak into your microphone to talk to the AI")
    print("Press Ctrl+C to disconnect")
    
    try:
        await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("Disconnecting...")
    finally:
        await room.disconnect()

if __name__ == "__main__":
    asyncio.run(join_room())