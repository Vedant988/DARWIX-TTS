"""
Audio Stitcher: Combines multiple audio chunks with precise silence injection using pydub.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioStitcher:
    """Stitches multiple audio chunks together with controlled silence."""

    @staticmethod
    def stitch_chunks(chunk_files: list[Path], pause_durations_ms: list[int], output_path: Path) -> Path:
        """
        Stitch audio chunks together with specified pauses between them.
        
        Args:
            chunk_files: List of paths to WAV files (in order)
            pause_durations_ms: List of pause durations in milliseconds (same length as chunk_files)
            output_path: Path where final audio will be saved
            
        Returns:
            Path to the stitched audio file
        """
        try:
            from pydub import AudioSegment
        except ImportError as e:
            logger.error(f"pydub not installed: {e}")
            raise RuntimeError("pydub is required for audio stitching") from e

        if not chunk_files:
            raise ValueError("No audio chunks provided")

        if len(chunk_files) != len(pause_durations_ms):
            raise ValueError("Mismatch between chunk files and pause durations")

        try:
            # Load the first chunk
            final_audio = AudioSegment.from_wav(str(chunk_files[0]))

            # Process remaining chunks
            for i in range(1, len(chunk_files)):
                pause_ms = pause_durations_ms[i - 1]  # Pause AFTER previous chunk
                
                if pause_ms > 0:
                    silence = AudioSegment.silent(duration=pause_ms)
                    final_audio = final_audio + silence

                # Load and append next chunk
                next_chunk = AudioSegment.from_wav(str(chunk_files[i]))
                final_audio = final_audio + next_chunk

            # Export final stitched audio
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_audio.export(str(output_path), format="wav")

            logger.info(f"Audio stitched successfully: {output_path} ({len(final_audio)}ms total)")
            return output_path

        except Exception as e:
            logger.error(f"Audio stitching failed: {e}", exc_info=True)
            raise RuntimeError(f"Audio stitching failed: {e}") from e


audio_stitcher = AudioStitcher()
