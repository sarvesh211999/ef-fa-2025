import asyncio
import logging
import os
import io
import wave
from typing import Optional

import pyaudio
import numpy as np
from dotenv import load_dotenv

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleVoiceAgent:
    """Ultra-simple voice agent for testing"""

    def __init__(self):
        # Initialize services
        self.stt_service = DeepgramSTTService(
            api_key="",
            model="nova-2",
            language="en-US",
        )

        self.llm_service = OpenAILLMService(
            api_key="",
            model="gpt-4",
        )

        self.tts_service = CartesiaTTSService(
            api_key="",
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
            model="sonic-english",
        )

        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.record_seconds = 3  # Record for 3 seconds at a time

        # PyAudio setup
        self.pyaudio = pyaudio.PyAudio()

        # Conversation history
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful AI voice assistant. Keep responses concise and conversational.",
            }
        ]

    def record_audio(self, duration: float = 3.0) -> bytes:
        """Record audio for specified duration"""
        logger.info(f"Recording for {duration} seconds...")

        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames = []
        for _ in range(int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        audio_data = b''.join(frames)
        logger.info("Recording complete")
        return audio_data

    def play_audio(self, audio_data: bytes):
        """Play audio data"""
        try:
            # Convert to numpy array and play
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Create output stream
            stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
            )

            # Play audio
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()

            logger.info("Audio playback complete")

        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio to text"""
        try:
            logger.info("Transcribing audio...")

            # Create audio frame
            audio_frame = AudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=self.channels,
            )

            # Process through STT
            result = await self.stt_service.process_frame(
                audio_frame, FrameDirection.DOWNSTREAM
            )

            if isinstance(result, TranscriptionFrame) and result.text.strip():
                text = result.text.strip()
                logger.info(f"Transcribed: {text}")
                return text

            logger.info("No transcription result")
            return None

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    async def get_response(self, user_text: str) -> Optional[str]:
        """Get AI response"""
        try:
            logger.info("Getting AI response...")

            # Add user message
            self.messages.append({"role": "user", "content": user_text})

            # Create text frame with the user's message
            text_frame = TextFrame(user_text)

            # Get LLM response
            result = await self.llm_service.process_frame(
                text_frame, FrameDirection.DOWNSTREAM
            )

            if isinstance(result, TextFrame):
                response = result.text
                logger.info(f"AI response: {response}")

                # Add to conversation history
                self.messages.append({"role": "assistant", "content": response})
                return response

            logger.info("No LLM response")
            return None

        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return None

    async def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech"""
        try:
            logger.info("Converting text to speech...")

            # Create text frame
            text_frame = TextFrame(text)

            # Convert to speech
            result = await self.tts_service.process_frame(
                text_frame, FrameDirection.DOWNSTREAM
            )

            if isinstance(result, TTSAudioRawFrame):
                logger.info("Text-to-speech complete")
                return result.audio

            logger.info("No TTS result")
            return None

        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            return None

    async def conversation_loop(self):
        """Main conversation loop"""
        logger.info("Starting conversation loop...")

        try:
            while True:
                # Record audio
                print("\nPress Enter to start recording, or 'q' to quit:")
                user_input = input()

                if user_input.lower() == 'q':
                    break

                # Record audio
                audio_data = self.record_audio(self.record_seconds)

                # Transcribe
                transcribed_text = await self.transcribe_audio(audio_data)

                if transcribed_text:
                    print(f"You said: {transcribed_text}")

                    # Get AI response
                    response_text = await self.get_response(transcribed_text)

                    if response_text:
                        print(f"AI: {response_text}")

                        # Convert to speech
                        speech_audio = await self.text_to_speech(response_text)

                        if speech_audio:
                            # Play audio
                            self.play_audio(speech_audio)
                        else:
                            print("Failed to generate speech")
                    else:
                        print("Failed to get AI response")
                else:
                    print("No speech detected or transcription failed")

        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.pyaudio.terminate()


async def main():
    """Main entry point"""
    # Check environment variables

    logger.info("All environment variables found")

    # Create and run voice agent
    agent = SimpleVoiceAgent()

    print("=" * 50)
    print("Simple Voice Agent Test")
    print("=" * 50)
    print("- Press Enter to record for 3 seconds")
    print("- Speak during recording")
    print("- AI will respond with voice")
    print("- Type 'q' to quit")
    print("=" * 50)

    await agent.conversation_loop()
    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
