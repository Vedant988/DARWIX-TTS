import { useEffect, useRef, useState } from "react";
import { Activity } from "lucide-react";

interface ProsodyVisualizerProps {
  audioRef: React.RefObject<HTMLAudioElement>;
  audioTime: number;
  audioDuration: number;
  isPlaying: boolean;
}

export default function ProsodyVisualizer({
  audioRef,
  audioTime,
  audioDuration,
  isPlaying,
}: ProsodyVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const [currentPitch, setCurrentPitch] = useState(0);
  const [currentLoudness, setCurrentLoudness] = useState(0);
  const animationRef = useRef<number | null>(null);

  // Initialize Web Audio API for analysis
  useEffect(() => {
    if (!audioRef.current) return;

    const setupAnalyser = () => {
      if (!audioContextRef.current) {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = audioContext;

        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        analyserRef.current = analyser;

        try {
          const source = audioContext.createMediaElementAudioSource(audioRef.current);
          source.connect(analyser);
          analyser.connect(audioContext.destination);
        } catch (e) {
          // Source already connected, that's okay
        }
      }
    };

    audioRef.current.addEventListener("play", setupAnalyser);
    return () => audioRef.current?.removeEventListener("play", setupAnalyser);
  }, []);

  // Analyze audio and draw visualizer
  const analyzeAndDraw = () => {
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

    // Calculate loudness (average of all frequencies)
    const loudness = dataArray.reduce((a, b) => a + b) / dataArray.length / 255;
    setCurrentLoudness(loudness);

    // Calculate pitch (fundamental frequency - first peak)
    let maxFreq = 0;
    let maxValue = 0;
    for (let i = 0; i < bufferLength; i++) {
      if (dataArray[i] > maxValue) {
        maxValue = dataArray[i];
        maxFreq = i;
      }
    }
    const pitch = (maxFreq / bufferLength) * 0.5; // Normalize to 0-1
    setCurrentPitch(pitch);

    // Draw frequency spectrum
    const barWidth = (canvas.width / bufferLength) * 2;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height;

      // Color based on pitch (low = warm, high = cool)
      const hue = pitch * 60; // 0-60 degree hue range
      ctx.fillStyle = `hsl(${120 + hue}, 70%, ${50 - pitch * 20}%)`;
      ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

      x += barWidth + 1;
    }

    // Draw center line
    ctx.strokeStyle = `hsla(${120 + pitch * 60}, 70%, 50%, 0.3)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    if (isPlaying) {
      animationRef.current = requestAnimationFrame(analyzeAndDraw);
    }
  };

  useEffect(() => {
    if (isPlaying) {
      animationRef.current = requestAnimationFrame(analyzeAndDraw);
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying]);

  const getQuality = (value: number): string => {
    if (value < 0.3) return "Low";
    if (value < 0.6) return "Medium";
    if (value < 0.8) return "High";
    return "Very High";
  };

  if (!isPlaying) {
    return (
      <section className="rounded-lg bg-gradient-to-br from-[#f2f4f4] via-white to-[#f2f4f4] p-8 shadow-sm border border-[#dfe3e4]">
        <h2 className="flex items-center gap-3 text-lg font-semibold text-[#2f3334] mb-4">
          <Activity className="h-5 w-5 text-[#5a6344]" />
          Prosody Metrics
        </h2>
        <p className="text-sm text-[#5b6061]">Play the audio to see real-time pitch and loudness analysis</p>
      </section>
    );
  }

  return (
    <section className="rounded-lg bg-gradient-to-br from-[#f2f4f4] via-white to-[#f2f4f4] p-8 shadow-sm border border-[#dfe3e4]">
      <h2 className="flex items-center gap-3 text-lg font-semibold text-[#2f3334] mb-6">
        <Activity className="h-5 w-5 text-[#5a6344] animate-pulse" />
        Real-Time Prosody Metrics
      </h2>

      {/* Frequency Spectrum Canvas */}
      <div className="mb-6 rounded-lg overflow-hidden bg-[#f9f9f9] border border-[#dfe3e4]">
        <canvas
          ref={canvasRef}
          width={600}
          height={80}
          className="w-full h-20 bg-[#f9f9f9]"
        />
      </div>

      {/* Pitch and Loudness Meters */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* Pitch Meter */}
        <div>
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-[#5b6061]">
              Pitch Level
            </span>
            <span className="text-sm font-semibold text-[#5a6344] tabular-nums">
              {(currentPitch * 100).toFixed(0)}%
            </span>
          </div>
          <div className="h-3 w-full rounded-full bg-[#dfe3e4] overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#a73b21] via-[#5a6344] to-[#5a90c2] transition-all duration-75"
              style={{ width: `${Math.min(100, currentPitch * 100)}%` }}
            />
          </div>
          <p className="mt-2 text-xs text-[#5b6061]">{getQuality(currentPitch)} pitch</p>
        </div>

        {/* Loudness Meter */}
        <div>
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-[#5b6061]">
              Loudness Level
            </span>
            <span className="text-sm font-semibold text-[#5a6344] tabular-nums">
              {(currentLoudness * 100).toFixed(0)}%
            </span>
          </div>
          <div className="h-3 w-full rounded-full bg-[#dfe3e4] overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#5a6344] to-[#a73b21] transition-all duration-75"
              style={{ width: `${Math.min(100, currentLoudness * 100)}%` }}
            />
          </div>
          <p className="mt-2 text-xs text-[#5b6061]">{getQuality(currentLoudness)} loudness</p>
        </div>
      </div>

      {/* Progress indicator */}
      <div className="text-center text-[11px] text-[#5b6061] font-semibold">
        Time: {Math.floor(audioTime)}s / {Math.floor(audioDuration)}s
      </div>
    </section>
  );
}
