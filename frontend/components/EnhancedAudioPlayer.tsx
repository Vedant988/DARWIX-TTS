import { useEffect, useRef, useState } from "react";
import { Play, Pause, Volume2, Download } from "lucide-react";

interface EnhancedAudioPlayerProps {
  audioUrl: string | null;
  audioRef: React.RefObject<HTMLAudioElement>;
  audioDuration: number;
  audioTime: number;
  onTimeUpdate: (time: number) => void;
  setAudioDuration: (duration: number) => void;
}

export default function EnhancedAudioPlayer({
  audioUrl,
  audioRef,
  audioDuration,
  audioTime,
  onTimeUpdate,
  setAudioDuration,
}: EnhancedAudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const [volume, setVolume] = useState(1);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationRef = useRef<number | null>(null);

  // Initialize Web Audio API
  useEffect(() => {
    if (!audioRef.current) return;

    const setupAudio = () => {
      if (!audioContextRef.current) {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = audioContext;

        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;

        const source = audioContext.createMediaElementAudioSource(audioRef.current);
        const gainNode = audioContext.createGain();

        source.connect(analyser);
        analyser.connect(gainNode);
        gainNode.connect(audioContext.destination);
        gainNode.gain.value = volume;
      }
    };

    const audio = audioRef.current;
    audio.addEventListener("play", setupAudio);
    audio.addEventListener("play", () => setIsPlaying(true));
    audio.addEventListener("pause", () => setIsPlaying(false));

    return () => {
      audio.removeEventListener("play", setupAudio);
      audio.removeEventListener("play", () => setIsPlaying(true));
      audio.removeEventListener("pause", () => setIsPlaying(false));
    };
  }, []);

  // Update volume
  useEffect(() => {
    if (audioContextRef.current && audioRef.current) {
      const gainNode = audioContextRef.current.createGain?.();
      if (gainNode) {
        gainNode.gain.value = volume;
      }
    }
    if (audioRef.current) {
      audioRef.current.volume = volume;
    }
  }, [volume]);

  // Draw waveform
  const drawWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;

    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    // Clear canvas
    ctx.fillStyle = "rgb(249, 249, 249)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw frequency bars
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height;

      // Gradient color based on frequency
      const hue = (i / bufferLength) * 60; // Green to blue range
      ctx.fillStyle = `hsl(${140 + hue}, 60%, 45%)`;
      ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

      x += barWidth + 1;
    }

    if (isPlaying) {
      animationRef.current = requestAnimationFrame(drawWaveform);
    }
  };

  // Play/Pause handler
  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
      animationRef.current = requestAnimationFrame(drawWaveform);
    }
  };

  // Timeline click handler
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    audioRef.current.currentTime = percent * audioDuration;
  };

  // Format time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  if (!audioUrl) return null;

  return (
    <section className="rounded-lg bg-gradient-to-br from-white via-[#f2f4f4] to-white p-8 shadow-sm border border-[#dfe3e4]">
      <audio
        ref={audioRef}
        src={audioUrl}
        onLoadedMetadata={(e) => setAudioDuration(e.currentTarget.duration)}
        onTimeUpdate={(e) => onTimeUpdate(e.currentTarget.currentTime)}
        crossOrigin="anonymous"
      />

      <div className="mb-6">
        <h2 className="text-lg font-semibold text-[#2f3334] mb-4">Generated Audio</h2>

        {/* Waveform Canvas */}
        <div className="mb-6 rounded-lg overflow-hidden bg-[#f9f9f9] border border-[#dfe3e4]">
          <canvas
            ref={canvasRef}
            width={600}
            height={100}
            className="w-full h-24 bg-[#f9f9f9]"
          />
        </div>

        {/* Timeline */}
        <div className="mb-4">
          <div
            onClick={handleTimelineClick}
            className="relative h-2 w-full cursor-pointer rounded-full bg-[#dfe3e4] group"
          >
            {/* Progress bar */}
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#5a6344] to-[#8e9677] transition-all"
              style={{ width: `${(audioTime / audioDuration) * 100}%` }}
            />
            {/* Playhead */}
            <div
              className="absolute top-1/2 h-4 w-4 -translate-y-1/2 rounded-full bg-[#5a6344] shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"
              style={{ left: `${(audioTime / audioDuration) * 100}%` }}
            />
          </div>
        </div>

        {/* Time display */}
        <div className="mb-6 flex items-center justify-between text-[12px] font-semibold text-[#5b6061]">
          <span className="font-mono">{formatTime(audioTime)}</span>
          <span className="font-mono">{formatTime(audioDuration)}</span>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Play/Pause Button */}
          <button
            onClick={togglePlay}
            className="flex h-12 w-12 items-center justify-center rounded-full bg-[#5a6344] text-white transition-all hover:bg-[#4a5334] active:scale-95 shadow-md"
          >
            {isPlaying ? (
              <Pause className="h-5 w-5" />
            ) : (
              <Play className="h-5 w-5 ml-0.5" />
            )}
          </button>

          {/* Volume Control */}
          <div className="flex items-center gap-2">
            <Volume2 className="h-4 w-4 text-[#5b6061]" />
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="w-24 h-1 rounded-full bg-[#dfe3e4] appearance-none cursor-pointer accent-[#5a6344]"
            />
            <span className="text-xs font-semibold text-[#5b6061] w-8 text-right">
              {Math.round(volume * 100)}%
            </span>
          </div>

          {/* Download Button */}
          <a
            href={audioUrl}
            download="synthesis.wav"
            className="ml-auto flex items-center gap-2 rounded-lg bg-[#f2f4f4] px-4 py-2 text-[12px] font-bold uppercase tracking-[0.15em] text-[#5a6344] transition-colors hover:bg-[#dee7c0]"
          >
            <Download className="h-4 w-4" />
            Download
          </a>
        </div>
      </div>

      {/* Real-time Metrics during playback */}
      {isPlaying && (
        <div className="mt-6 grid grid-cols-2 gap-4 p-4 bg-[#f2f4f4] rounded-lg border border-[#dfe3e4]">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#5b6061] mb-2">
              Playback Status
            </p>
            <p className="text-sm font-semibold text-[#5a6344]">
              {formatTime(audioTime)} / {formatTime(audioDuration)}
            </p>
          </div>
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#5b6061] mb-2">
              Progress
            </p>
            <p className="text-sm font-semibold text-[#5a6344]">
              {Math.round((audioTime / audioDuration) * 100)}%
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
