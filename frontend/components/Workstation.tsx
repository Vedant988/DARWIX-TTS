"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AudioLines,
  Brain,
  Download,
  HelpCircle,
  History,
  Library,
  Loader2,
  Mic,
  Play,
  Plus,
  Search,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Square,
  WandSparkles,
} from "lucide-react";
import EnhancedAudioPlayer from "./EnhancedAudioPlayer";
import ProsodyVisualizer from "./ProsodyVisualizer";

type Language = "en" | "hi";

type EmotionModel = {
  key: string;
  display_name: string;
  tradeoff: string;
  label_count: number;
  latency_hint_ms: string;
  multi_label: boolean;
  default?: boolean;
};

type EmotionScore = {
  label: string;
  score: number;
};

type AnalysisPayload = {
  primary_emotion: string;
  confidence: number;
  clarity: number;
  display_name: string;
  top_emotions: EmotionScore[];
  normalized_emotions: EmotionScore[];
};

type VoiceProfile = {
  speaker: string;
  pitch: number;
  pace: number;
  loudness: number;
  intensity: number;
  stability: number;
  dimensions: { valence: number; arousal: number; dominance: number };
  reason: string;
};

type SummaryPayload = {
  text: string;
  latency?: number;
  file_url?: string;
};

const FALLBACK_MODELS: EmotionModel[] = [
  {
    key: "hartmann",
    display_name: "j-hartmann/emotion-english-distilroberta-base",
    tradeoff: "Fastest premium-quality",
    label_count: 7,
    latency_hint_ms: "80-250",
    multi_label: false,
    default: true,
  },
  {
    key: "go_emotions",
    display_name: "SamLowe/roberta-base-go_emotions",
    tradeoff: "Best balance / 28 classes",
    label_count: 28,
    latency_hint_ms: "150-400",
    multi_label: true,
  },
  {
    key: "bhadresh",
    display_name: "bhadresh-savani/distilbert-base-uncased-emotion",
    tradeoff: "Very fast / compact",
    label_count: 6,
    latency_hint_ms: "60-200",
    multi_label: false,
  },
];

const METRIC_COLORS = [
  "bg-[#a73b21]",
  "bg-[#6f5e26]",
  "bg-[#777b7c]",
  "bg-[#5a6344]",
  "bg-[#5c624a]",
  "bg-[#8a8f80]",
];

function titleCase(value: string) {
  return value
    .split(/[_-\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function toneFromEmotion(emotion?: string) {
  if (!emotion) return "Velvet";
  if (emotion.includes("anger") || emotion.includes("disgust")) return "Grit";
  if (emotion.includes("joy") || emotion.includes("surprise")) return "Air";
  return "Velvet";
}

export default function Workstation() {
  const [models, setModels] = useState<EmotionModel[]>(FALLBACK_MODELS);
  const [selectedModel, setSelectedModel] = useState("hartmann");
  const [language, setLanguage] = useState<Language>("en");
  const [speaker, setSpeaker] = useState("manisha");
  const [text, setText] = useState("");
  const [search, setSearch] = useState("");
  const [connected, setConnected] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [voice, setVoice] = useState<VoiceProfile | null>(null);
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [submittedText, setSubmittedText] = useState("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioDuration, setAudioDuration] = useState(0);
  const [audioTime, setAudioTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);
  const [pipelineStage, setPipelineStage] = useState<string | null>(null);
  const [chunkCount, setChunkCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null!);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    const loadModels = async () => {
      try {
        const res = await fetch(`${apiUrl}/emotion-models`);
        if (!res.ok) return;
        const data = await res.json();
        if (Array.isArray(data.models) && data.models.length > 0) {
          setModels(data.models);
          const fallback = data.models.find((item: EmotionModel) => item.default) ?? data.models[0];
          setSelectedModel(fallback.key);
        }
      } catch {
        // Local fallback models keep the page usable without the endpoint.
      }
    };
    void loadModels();
  }, []);

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
    console.log("[WS] Connecting to:", wsUrl);
    const ws = new WebSocket(`${wsUrl}/ws/voice?language=${language}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("[WS] Connected");
      setConnected(true);
      setErrorDetails(null);
    };
    ws.onclose = (event) => {
      console.log("[WS] Disconnected", event);
      setConnected(false);
      if (event.code !== 1000) {
        setErrorDetails(
          event.reason
            ? `WebSocket closed unexpectedly: ${event.reason}`
            : `WebSocket closed with code ${event.code}`,
        );
      }
    };
    ws.onerror = () => {
      const details = "WebSocket connection failed";
      console.error("[WS] Error:", details);
      setErrorDetails(details);
      setConnected(false);
      setSubmitting(false);
    };
    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        console.log("[WS] Received audio blob, size:", event.data.byteLength);
        setAudioUrl((current) => {
          if (current) URL.revokeObjectURL(current);
          return URL.createObjectURL(new Blob([event.data], { type: "audio/wav" }));
        });
        setSubmitting(false);
        return;
      }

      if (typeof event.data !== "string") {
        console.warn("[WS] Non-string message received:", typeof event.data);
        return;
      }
      
      const payload = JSON.parse(event.data);
      console.log("[WS] Received message type:", payload.type);
      
      if (payload.type === "pipeline_status") {
        console.log("[WS] Pipeline stage:", payload.stage);
        setPipelineStage(payload.stage);
        if (payload.chunk_count) {
          setChunkCount(payload.chunk_count);
        }
      } else if (payload.type === "analysis") {
        console.log("[WS] Emotion analysis received:", payload.analysis.primary_emotion);
        setAnalysis(payload.analysis);
        setVoice(payload.voice_profile);
        setSubmittedText(payload.text ?? "");
      } else if (payload.type === "transcript" && payload.role === "assistant") {
        console.log("[WS] Transcript received:", payload.text.slice(0, 50));
        setSummary(payload);
        setPipelineStage(null);
        setSubmitting(false);
      } else if (payload.type === "interrupt") {
        console.log("[WS] Interrupt received");
        setPipelineStage(null);
        setSubmitting(false);
      } else if (payload.type === "error") {
        console.error("[WS] Error from backend:", payload.message);
        setPipelineStage(null);
        setSubmitting(false);
      }
    };

    wsRef.current = ws;
    return () => ws.close();
  }, [language]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  // Track audio play/pause state
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
    };
  }, []);

  const filteredModels = useMemo(() => {
    if (!search.trim()) return models;
    const q = search.toLowerCase();
    return models.filter(
      (item) =>
        item.display_name.toLowerCase().includes(q) ||
        item.tradeoff.toLowerCase().includes(q),
    );
  }, [models, search]);

  const bars = useMemo(
    () =>
      analysis?.normalized_emotions?.length
        ? analysis.normalized_emotions
        : [
            { label: "anger", score: 0.12 },
            { label: "frustration", score: 0.24 },
            { label: "neutral", score: 0.45 },
            { label: "joy", score: 0.18 },
            { label: "calm", score: 0.82 },
          ],
    [analysis],
  );

  const waveform = useMemo(
    () =>
      Array.from({ length: 50 }, (_, index) => {
        const sample = bars[index % bars.length]?.score ?? 0.5;
        return clamp(Math.round(sample * 100 + (index % 6) * 8), 18, 98);
      }),
    [bars],
  );

  const selectedModelData = models.find((item) => item.key === selectedModel) ?? models[0];
  const intensity = voice ? clamp(Math.round(voice.intensity * 100), 0, 100) : 85;
  const vocalTexture = toneFromEmotion(analysis?.primary_emotion);
  const hasSubmitted = Boolean(analysis || summary || audioUrl || submittedText);

  const submit = () => {
    console.log("[SUBMIT] Button clicked");
    console.log("[SUBMIT] WebSocket state:", wsRef.current?.readyState, "Text:", text.trim().slice(0, 50));
    
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || !text.trim()) {
      console.error("[SUBMIT] Cannot submit - WS not ready or empty text");
      return;
    }
    
    console.log("[SUBMIT] Submitting text to backend...");
    setSubmitting(true);
    setAnalysis(null);
    setVoice(null);
    setSummary(null);
    setSubmittedText("");
    setAudioDuration(0);
    setAudioTime(0);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    
    const payload = {
      type: "text",
      text: text.trim(),
      emotion_model: selectedModel,
      language,
      speaker,
    };
    console.log("[SUBMIT] Sending payload:", payload);
    wsRef.current.send(JSON.stringify(payload));
  };

  const stop = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "interrupt" }));
    }
    audioRef.current?.pause();
    setSubmitting(false);
  };

  const resetFlow = () => {
    setAnalysis(null);
    setVoice(null);
    setSummary(null);
    setSubmittedText("");
    setAudioDuration(0);
    setAudioTime(0);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
  };

  return hasSubmitted ? (
    <TelemetryPage
      analysis={analysis}
      audioDuration={audioDuration}
      audioRef={audioRef}
      audioTime={audioTime}
      audioUrl={audioUrl}
      bars={bars}
      onReset={resetFlow}
      setAudioDuration={setAudioDuration}
      setAudioTime={setAudioTime}
      submittedText={submittedText}
      summary={summary}
      voice={voice}
      waveform={waveform}
    />
  ) : (
    <AtelierInputPage
      connected={connected}
      errorDetails={errorDetails}
      filteredModels={filteredModels}
      intensity={intensity}
      language={language}
      pipelineStage={pipelineStage}
      chunkCount={chunkCount}
      search={search}
      selectedModel={selectedModel}
      setLanguage={setLanguage}
      setSearch={setSearch}
      setSelectedModel={setSelectedModel}
      setText={setText}
      submit={submit}
      submitting={submitting}
      text={text}
      vocalTexture={vocalTexture}
      onStop={stop}
      speaker={speaker}
      setSpeaker={setSpeaker}
    />
  );
}

function AtelierInputPage({
  connected,
  errorDetails,
  filteredModels,
  intensity,
  language,
  pipelineStage,
  chunkCount,
  search,
  selectedModel,
  setLanguage,
  setSearch,
  setSelectedModel,
  setText,
  submit,
  submitting,
  text,
  vocalTexture,
  onStop,
  speaker,
  setSpeaker,
}: {
  connected: boolean;
  errorDetails: string | null;
  filteredModels: EmotionModel[];
  intensity: number;
  language: Language;
  pipelineStage: string | null;
  chunkCount: number;
  search: string;
  selectedModel: string;
  setLanguage: (value: Language) => void;
  setSearch: (value: string) => void;
  setSelectedModel: (value: string) => void;
  setText: (value: string) => void;
  submit: () => void;
  submitting: boolean;
  text: string;
  vocalTexture: string;
  onStop: () => void;
  speaker: string;
  setSpeaker: (value: string) => void;
}) {
  return (
    <div className="min-h-screen bg-[#f9f9f9] text-[#2f3334]">
      <nav className="fixed top-0 z-50 flex w-full items-center justify-between bg-[#f9f9f9] px-8 py-6 lg:px-12">
        <div className="flex items-center gap-8">
          <span className="text-xl font-bold tracking-tight text-[#5a6344]">DARWIX AI</span>
          <div className="hidden items-center gap-6 md:flex">
            <span className="font-semibold text-[#5a6344]">DARWIX AI</span>
            <span className="rounded-full px-3 py-1 text-[#2f3334]">Library</span>
            <span className="rounded-full px-3 py-1 text-[#2f3334]">Voice Lab</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden items-center gap-2 rounded-full bg-[#f2f4f4] px-4 py-2 sm:flex">
            <Search className="h-4 w-4 text-[#777b7c]" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search workstation..."
              className="w-48 bg-transparent text-sm outline-none"
            />
          </div>
          <button className="rounded-full p-2 hover:bg-[#f2f4f4]">
            <Settings className="h-5 w-5 text-[#5a6344]" />
          </button>
          <button className="rounded-full p-2 hover:bg-[#f2f4f4]">
            <SlidersHorizontal className="h-5 w-5 text-[#5a6344]" />
          </button>
          <div className="grid h-8 w-8 place-items-center rounded-full bg-[#e0e6c7] text-[10px] font-bold text-[#5a6344]">
            AI
          </div>
        </div>
      </nav>

      <aside className="fixed left-0 top-0 hidden h-full w-72 flex-col rounded-r-lg bg-[#f2f4f4] shadow-[20px_0_40px_rgba(47,51,52,0.03)] md:flex">
        <div className="px-8 pb-10 pt-10">
          <div className="mb-6 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#5a6344] text-[#f4fdd6]">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-black uppercase tracking-tight text-[#5a6344]">DARWIX AI</h2>
              <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#5b6061] opacity-60">
                AI Voice Studio
              </p>
            </div>
          </div>
          <button className="flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-br from-[#5a6344] to-[#dee7c0] px-6 py-4 text-[11px] font-bold uppercase tracking-[0.24em] text-white shadow-sm transition-transform hover:scale-[0.98]">
            <Plus className="h-4 w-4" />
            New Generation
          </button>
        </div>

        <div className="flex-1 space-y-4 px-6">
          <nav className="flex flex-col gap-2">
            <span className="flex items-center gap-4 rounded-lg bg-gradient-to-br from-[#5a6344] to-[#dee7c0] px-6 py-3 text-white">
              <Sparkles className="h-4 w-4" />
              <span className="text-[11px] font-bold uppercase tracking-[0.24em]">DARWIX AI</span>
            </span>
            <SideLink icon={<Library className="h-4 w-4" />} label="Library" />
            <SideLink icon={<AudioLines className="h-4 w-4" />} label="Voice Lab" />
            <SideLink icon={<Activity className="h-4 w-4" />} label="Analytics" />
          </nav>
        </div>

        <div className="border-t border-[#afb3b3]/20 px-6 py-8">
          <nav className="flex flex-col gap-2">
            <SideLink icon={<HelpCircle className="h-4 w-4" />} label="Support" />
            <SideLink icon={<Settings className="h-4 w-4" />} label="Theme" />
          </nav>
        </div>
      </aside>

      <main className="px-8 pb-12 pt-28 md:ml-72">
        <div className="mx-auto max-w-7xl">
          <header className="mb-12">
            <h1 className="text-4xl font-bold tracking-tight text-[#5a6344]">Workstation Input Panel</h1>
            <p className="mt-2 text-[#5b6061]">
              Refine your creative intent and choose the emotional engine for your generation.
            </p>
          </header>

          <div className="grid grid-cols-1 items-start gap-10 lg:grid-cols-12">
            <div className="space-y-8 lg:col-span-7">
              <section className="rounded-lg bg-white p-10 shadow-sm">
                <div className="mb-8 flex items-center gap-3">
                  <Brain className="h-5 w-5 text-[#5a6344]" />
                  <h2 className="text-xl font-semibold text-[#2f3334]">Emotion Engine Selection</h2>
                </div>

                <div className="space-y-6">
                  {filteredModels.map((model) => {
                    const active = model.key === selectedModel;
                    return (
                      <button
                        key={model.key}
                        onClick={() => setSelectedModel(model.key)}
                        className={`w-full rounded-lg border p-8 text-left transition-all duration-300 ${
                          active
                            ? "border-[#5a6344] bg-[#dee7c0]/20 shadow-md"
                            : "border-[#afb3b3]/30 bg-[#f2f4f4]/30 hover:border-[#5a6344]/40 hover:shadow-sm"
                        }`}
                      >
                        <div className="mb-4 flex items-start justify-between gap-4">
                          <div>
                            <h3 className={`text-xl font-bold tracking-tight ${active ? "text-[#5a6344]" : "text-[#2f3334]"}`}>
                              {model.key === "hartmann" ? "Hartmann" : model.key === "go_emotions" ? "Go_Emotions" : "Bhadresh"}
                            </h3>
                            <p className="mt-1 inline-block rounded bg-[#eceeef] px-2 py-0.5 text-[11px] font-mono text-[#5b6061]">
                              {model.display_name}
                            </p>
                          </div>
                          <div className={`h-6 w-6 rounded-full ${active ? "border-4 border-[#5a6344] bg-white shadow-inner" : "border-2 border-[#afb3b3]"}`} />
                        </div>

                        <div className="my-4 grid grid-cols-3 gap-4 border-y border-[#5a6344]/10 py-4">
                          <MetricMini label="Classes" value={`${model.label_count} ${model.multi_label ? "Wide" : "Core"}`} />
                          <MetricMini label="Granularity" value={model.multi_label ? "Fine" : model.key === "bhadresh" ? "Moderate" : "High"} />
                          <MetricMini label="Speed" value={`${model.latency_hint_ms}ms`} />
                        </div>

                        <div className="space-y-4">
                          <div>
                            <p className="mb-1 text-[11px] font-bold uppercase tracking-[0.2em] text-[#2f3334] opacity-40">Best For</p>
                            <p className="text-sm leading-relaxed text-[#5b6061]">
                              {model.key === "hartmann" && "Deep creative prose, literary analysis, and scripts requiring nuanced emotional subtlety."}
                              {model.key === "go_emotions" && "Colloquial dialogue, social media analysis, and capturing complex human reactions."}
                              {model.key === "bhadresh" && "Real-time processing, simple chat applications, and high-volume data streams."}
                            </p>
                          </div>
                          <div className="grid grid-cols-2 gap-6">
                            <ProsCons
                              title="Pros"
                              tone="good"
                              items={
                                model.key === "hartmann"
                                  ? ["Highly accurate", "Minimal bias"]
                                  : model.key === "go_emotions"
                                    ? ["Diverse range", "Modern context"]
                                    : ["Blazing fast", "Low memory"]
                              }
                            />
                            <ProsCons
                              title="Cons"
                              tone="bad"
                              items={
                                model.key === "hartmann"
                                  ? ["Resource intensive", "Limited categories"]
                                  : model.key === "go_emotions"
                                    ? ["Slower inference", "Potential noise"]
                                    : ["Lower precision", "Simple labels"]
                              }
                            />
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </section>
            </div>

            <div className="space-y-6 lg:col-span-5">
              <section className="rounded-lg bg-[#f2f4f4] p-8">
                <div className="mb-8">
                  <label className="mb-4 block text-[11px] font-bold uppercase tracking-[0.22em] text-[#5a6344]">
                    Manifesto Input
                  </label>
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="min-h-[220px] w-full resize-none rounded-lg bg-white p-6 leading-relaxed text-[#2f3334] outline-none ring-1 ring-transparent placeholder:text-[#afb3b3] focus:ring-[#5a6344]/30"
                    placeholder="Type or paste your narrative text here..."
                    rows={8}
                  />
                </div>

                <div className="space-y-6">
                  <div>
                    <div className="mb-2 flex justify-between">
                      <label className="text-[11px] font-bold uppercase tracking-[0.18em] text-[#5b6061]">Intensity</label>
                      <span className="text-[11px] font-bold text-[#5a6344]">{intensity}%</span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#dfe3e4]">
                      <div className="h-full rounded-full bg-[#5a6344]" style={{ width: `${intensity}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="mb-2 flex justify-between">
                      <label className="text-[11px] font-bold uppercase tracking-[0.18em] text-[#5b6061]">Vocal Texture</label>
                      <span className="text-[11px] font-bold text-[#5a6344]">{vocalTexture}</span>
                    </div>
                    <div className="flex gap-2">
                      {["Velvet", "Grit", "Air"].map((tone) => (
                        <button
                          key={tone}
                          className={`flex-1 rounded-lg px-3 py-2 text-[10px] font-bold uppercase tracking-[0.2em] ${
                            vocalTexture === tone ? "bg-[#5a6344] text-[#f4fdd6]" : "border border-[#afb3b3]/20 bg-white text-[#5b6061]"
                          }`}
                        >
                          {tone}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="mb-2 flex justify-between">
                      <label className="text-[11px] font-bold uppercase tracking-[0.18em] text-[#5b6061]">Language</label>
                      <span className="text-[11px] font-bold text-[#5a6344]">{language === "en" ? "English" : "Hindi"}</span>
                    </div>
                    <div className="flex gap-2">
                      {(["en", "hi"] as Language[]).map((item) => (
                        <button
                          key={item}
                          onClick={() => setLanguage(item)}
                          className={`flex-1 rounded-lg px-3 py-2 text-[10px] font-bold uppercase tracking-[0.2em] ${
                            language === item ? "bg-[#5a6344] text-[#f4fdd6]" : "border border-[#afb3b3]/20 bg-white text-[#5b6061]"
                          }`}
                        >
                          {item === "en" ? "English" : "Hindi"}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="text-[11px] font-bold uppercase tracking-[0.18em] text-[#5b6061]">Voice Speaker</label>
                    <div className="mt-2 grid grid-cols-2 gap-2">
                      {(["manisha", "vidya", "arya", "abhilash", "karun", "hitesh", "anushka"] as const).map((s) => (
                        <button
                          key={s}
                          onClick={() => setSpeaker(s)}
                          className={`rounded-lg px-3 py-2 text-[10px] font-bold uppercase tracking-[0.2em] transition-colors ${
                            speaker === s ? "bg-[#5a6344] text-[#f4fdd6]" : "border border-[#afb3b3]/20 bg-white text-[#5b6061] hover:bg-[#f4fdd6]"
                          }`}
                        >
                          {titleCase(s)}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-lg border border-[#afb3b3]/20 bg-white px-4 py-3 text-xs text-[#5b6061]">
                    Backend status:
                    <span className={`ml-2 font-semibold ${connected ? "text-[#5a6344]" : "text-[#a73b21]"}`}>
                      {connected ? "Connected" : "Offline"}
                    </span>
                    {errorDetails ? (
                      <p className="mt-2 text-xs text-[#a73b21]">{errorDetails}</p>
                    ) : null}
                  </div>
                </div>

                <div className="mt-10 flex gap-3">
                  {submitting && (
                    <button
                      onClick={onStop}
                      className="flex items-center justify-center gap-2 rounded-lg border border-[#afb3b3]/30 px-5 py-4 text-sm font-semibold text-[#5a6344]"
                    >
                      <Square className="h-4 w-4" />
                      Stop
                    </button>
                  )}
                  <button
                    onClick={submit}
                    disabled={!connected || !text.trim() || submitting}
                    className="flex flex-1 items-center justify-center gap-3 rounded-lg bg-[#5a6344] py-5 text-lg font-bold text-[#f4fdd6] shadow-lg shadow-[#5a6344]/10 transition-transform hover:scale-[0.99] disabled:opacity-50"
                  >
                    {submitting ? <Loader2 className="h-5 w-5 animate-spin" /> : "Synthesize Emotion"}
                    <Sparkles className="h-5 w-5" />
                  </button>
                </div>

                {pipelineStage && (
                  <div className="mt-8 space-y-3 rounded-lg bg-[#f2f4f4] p-6">
                    <p className="text-xs font-bold uppercase tracking-[0.15em] text-[#5b6061]">Processing Pipeline</p>
                    <div className="space-y-2">
                      {[
                        "Optimizing Text for Emotion",
                        "Generating Semantic Micro-Chunks",
                        "Mapping Chunk-Level Prosody Parameters",
                        "Synthesizing Concurrent Audio Streams",
                        "Assembling Final Audio Vector",
                      ].map((stage, idx) => {
                        const isActive = pipelineStage === stage;
                        const isCompleted = [
                          "Optimizing Text for Emotion",
                          "Generating Semantic Micro-Chunks",
                          "Mapping Chunk-Level Prosody Parameters",
                          "Synthesizing Concurrent Audio Streams",
                        ].indexOf(stage) <
                          [
                            "Optimizing Text for Emotion",
                            "Generating Semantic Micro-Chunks",
                            "Mapping Chunk-Level Prosody Parameters",
                            "Synthesizing Concurrent Audio Streams",
                            "Assembling Final Audio Vector",
                          ].indexOf(pipelineStage || "");

                        return (
                          <div key={idx} className="flex items-center gap-3">
                            <div
                              className={`h-2 w-2 rounded-full transition-all ${
                                isActive
                                  ? "animate-pulse bg-[#5a6344]"
                                  : isCompleted
                                    ? "bg-[#5a6344]"
                                    : "bg-[#dfe3e4]"
                              }`}
                            />
                            <span
                              className={`text-xs font-semibold uppercase tracking-[0.1em] ${
                                isActive || isCompleted ? "text-[#5a6344]" : "text-[#afb3b3]"
                              }`}
                            >
                              [ {isActive ? "ACTIVE" : isCompleted ? "✓" : "PENDING"} ] {stage}
                              {isActive && chunkCount > 0 && ` (${chunkCount} chunks)`}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </section>

              <div className="group relative h-72 overflow-hidden rounded-lg shadow-sm">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(222,231,192,0.9),transparent_35%),linear-gradient(140deg,#5a6344_0%,#8e9677_45%,#f4fed6_100%)]" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent" />
                <div className="absolute bottom-0 z-10 p-8">
                  <span className="mb-2 block text-[10px] font-bold uppercase tracking-[0.25em] text-white/70">Active Atmosphere</span>
                  <h4 className="text-2xl font-bold tracking-tight text-white">Misty Morning Solitude</h4>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <button className="fixed bottom-8 right-8 flex h-14 w-14 items-center justify-center rounded-full bg-[#5a6344] text-[#f4fdd6] shadow-xl transition-transform hover:scale-105 md:hidden">
        <Plus className="h-5 w-5" />
      </button>
    </div>
  );
}

function SideLink({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button className="flex items-center gap-4 rounded-lg px-6 py-3 text-[11px] font-bold uppercase tracking-[0.24em] text-[#5b6061] transition-colors hover:bg-[#e0e6c7]/30 hover:text-[#5a6344]">
      {icon}
      <span>{label}</span>
    </button>
  );
}

function MetricMini({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#5b6061] opacity-60">{label}</p>
      <p className="mt-1 text-sm font-semibold text-[#2f3334]">{value}</p>
    </div>
  );
}

function ProsCons({ title, tone, items }: { title: string; tone: "good" | "bad"; items: string[] }) {
  const color = tone === "good" ? "text-[#5a6344]" : "text-[#a73b21]";
  const symbol = tone === "good" ? "✓" : "✕";

  return (
    <div>
      <p className={`mb-2 text-[11px] font-bold uppercase tracking-[0.2em] ${color}`}>{title}</p>
      <ul className="space-y-1">
        {items.map((item, idx) => (
          <li key={idx} className={`text-[13px] leading-relaxed ${color}`}>
            <span className="mr-2">{symbol}</span>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function TelemetryPage({
  analysis,
  audioDuration,
  audioRef,
  audioTime,
  audioUrl,
  bars,
  onReset,
  setAudioDuration,
  setAudioTime,
  submittedText,
  summary,
  voice,
  waveform,
}: {
  analysis: AnalysisPayload | null;
  audioDuration: number;
  audioRef: React.RefObject<HTMLAudioElement>;
  audioTime: number;
  audioUrl: string | null;
  bars: EmotionScore[];
  onReset: () => void;
  setAudioDuration: (value: number) => void;
  setAudioTime: (value: number) => void;
  submittedText: string;
  summary: SummaryPayload | null;
  voice: VoiceProfile | null;
  waveform: number[];
}) {
  return (
    <div className="min-h-screen bg-[#f9f9f9] text-[#2f3334]">
      <nav className="fixed top-0 z-50 flex w-full items-center justify-between bg-[#f9f9f9] px-8 py-6 lg:px-12 shadow-sm">
        <div className="flex items-center gap-4">
          <button onClick={onReset} className="rounded-full p-2 hover:bg-[#f2f4f4]">
            <span className="text-lg font-bold text-[#5a6344]">←</span>
          </button>
          <span className="text-xl font-bold tracking-tight text-[#5a6344]">DARWIX AI</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-[#5b6061]">Workstation</span>
          <div className="ml-4 grid h-8 w-8 place-items-center rounded-full bg-[#e0e6c7] text-[10px] font-bold text-[#5a6344]">
            AI
          </div>
        </div>
      </nav>

      <aside className="fixed left-0 top-0 hidden h-full w-72 flex-col border-r border-[#afb3b3]/20 bg-white md:flex pt-24">
        <div className="px-6 py-6">
          <h3 className="mb-6 text-sm font-bold uppercase tracking-[0.2em] text-[#5b6061]">Analysis</h3>
          <nav className="space-y-2">
            <SideLink icon={<Brain className="h-4 w-4" />} label="Emotion Matrix" />
            <SideLink icon={<Mic className="h-4 w-4" />} label="Voice Profile" />
            <SideLink icon={<WandSparkles className="h-4 w-4" />} label="Synthesis" />
          </nav>
        </div>
        <div className="border-t border-[#afb3b3]/20 px-6 py-6">
          <button
            onClick={onReset}
            className="w-full rounded-lg bg-[#f2f4f4] px-4 py-3 text-[11px] font-bold uppercase tracking-[0.2em] text-[#5a6344] transition-colors hover:bg-[#dee7c0]"
          >
            New Generation
          </button>
        </div>
      </aside>

      <main className="px-8 pb-12 pt-28 md:ml-72">
        <div className="mx-auto max-w-6xl">
          <header className="mb-12">
            <h1 className="text-4xl font-bold tracking-tight text-[#5a6344]">Telemetry Analysis</h1>
            <p className="mt-2 text-[#5b6061]">
              Detailed emotional and vocal analysis of your synthesis
            </p>
          </header>

          <div className="grid grid-cols-1 items-start gap-10 lg:grid-cols-12">
            <div className="space-y-8 lg:col-span-7">
              {/* Submitted Text Section */}
              <section className="rounded-lg bg-white p-8 shadow-sm">
                <h2 className="mb-4 text-lg font-semibold text-[#2f3334]">Original Input</h2>
                <p className="leading-relaxed text-[#5b6061]">{submittedText}</p>
              </section>

              {/* Enhanced Audio Player */}
              <EnhancedAudioPlayer
                audioUrl={audioUrl}
                audioRef={audioRef}
                audioDuration={audioDuration}
                audioTime={audioTime}
                onTimeUpdate={setAudioTime}
                setAudioDuration={setAudioDuration}
              />

              {/* Emotion Matrix */}
              {analysis && (
                <section className="rounded-lg bg-white p-8 shadow-sm">
                  <h2 className="mb-6 flex items-center gap-3 text-lg font-semibold text-[#2f3334]">
                    <Brain className="h-5 w-5 text-[#5a6344]" />
                    Emotion Matrix
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <div className="mb-2 flex items-center justify-between">
                        <span className="text-sm font-semibold text-[#2f3334]">Primary Emotion</span>
                        <span className="text-sm font-bold text-[#5a6344]">{pct(analysis.confidence)}</span>
                      </div>
                      <p className="text-lg font-bold text-[#5a6344]">{titleCase(analysis.primary_emotion)}</p>
                    </div>
                    <div className="border-t border-[#afb3b3]/20 pt-4">
                      <p className="mb-3 text-[11px] font-bold uppercase tracking-[0.15em] text-[#5b6061]">Top Emotions</p>
                      <div className="space-y-2">
                        {bars.slice(0, 5).map((emotion, idx) => (
                          <div key={idx}>
                            <div className="mb-1 flex items-center justify-between">
                              <span className="text-sm text-[#5b6061]">{titleCase(emotion.label)}</span>
                              <span className="text-sm font-semibold text-[#2f3334]">{pct(emotion.score)}</span>
                            </div>
                            <div className="h-2 rounded-full bg-[#dfe3e4]">
                              <div
                                className={`h-full rounded-full ${METRIC_COLORS[idx % METRIC_COLORS.length]}`}
                                style={{ width: `${emotion.score * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </section>
              )}
            </div>

            <div className="space-y-8 lg:col-span-5">
              {/* Voice Profile */}
              {voice && (
                <section className="rounded-lg bg-white p-8 shadow-sm">
                  <h2 className="mb-6 flex items-center gap-3 text-lg font-semibold text-[#2f3334]">
                    <Mic className="h-5 w-5 text-[#5a6344]" />
                    Voice Profile
                  </h2>
                  <div className="space-y-5">
                    <div>
                      <div className="mb-2 flex items-center justify-between">
                        <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-[#5b6061]">Speaker</span>
                        <span className="text-sm font-semibold text-[#2f3334]">{voice.speaker}</span>
                      </div>
                    </div>
                    {[
                      { label: "Pitch", value: voice.pitch },
                      { label: "Pace", value: voice.pace },
                      { label: "Loudness", value: voice.loudness },
                      { label: "Intensity", value: voice.intensity },
                      { label: "Stability", value: voice.stability },
                    ].map((item, idx) => (
                      <div key={idx}>
                        <div className="mb-2 flex items-center justify-between">
                          <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-[#5b6061]">{item.label}</span>
                          <span className="text-sm font-semibold text-[#2f3334] tabular-nums">{(item.value * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-[#dfe3e4] overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-[#5a6344] to-[#8e9677] transition-all duration-300"
                            style={{ width: `${Math.min(100, Math.max(0, item.value * 100))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                    {voice.dimensions && (
                      <div className="border-t border-[#afb3b3]/20 pt-4">
                        <p className="mb-3 text-[11px] font-bold uppercase tracking-[0.15em] text-[#5b6061]">Dimensions</p>
                        <div className="grid grid-cols-3 gap-3 text-center">
                          {Object.entries(voice.dimensions).map(([key, val]) => (
                            <div key={key}>
                              <p className="text-[10px] text-[#5b6061]">{titleCase(key)}</p>
                              <p className="text-lg font-bold text-[#5a6344]">{pct(val as number)}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Waveform Visualization */}
              <section className="rounded-lg bg-gradient-to-br from-[#f2f4f4] to-[#f9f9f9] p-8 shadow-sm">
                <h2 className="mb-6 flex items-center gap-3 text-lg font-semibold text-[#2f3334]">
                  <AudioLines className="h-5 w-5 text-[#5a6344]" />
                  Waveform
                </h2>
                <div className="flex items-end justify-center gap-0.5 h-40">
                  {waveform.map((height, idx) => (
                    <div
                      key={idx}
                      className="flex-1 rounded-t-sm bg-gradient-to-t from-[#5a6344] to-[#dee7c0] opacity-80 transition-all"
                      style={{ height: `${height}%` }}
                    />
                  ))}
                </div>
              </section>

              {/* Summary */}
              {summary && (
                <section className="rounded-lg bg-[#e0e6c7]/10 p-8 shadow-sm">
                  <h2 className="mb-4 flex items-center gap-3 text-lg font-semibold text-[#2f3334]">
                    <WandSparkles className="h-5 w-5 text-[#5a6344]" />
                    Summary
                  </h2>
                  <p className="leading-relaxed text-[#5b6061]">{summary.text}</p>
                  {summary.latency && (
                    <p className="mt-4 text-[12px] text-[#5b6061] opacity-60">
                      Generated in {summary.latency}ms
                    </p>
                  )}
                </section>
              )}
            </div>
          </div>

          <div className="mt-12 flex justify-center">
            <button
              onClick={onReset}
              className="flex items-center justify-center gap-2 rounded-lg bg-[#5a6344] px-8 py-4 text-[11px] font-bold uppercase tracking-[0.2em] text-[#f4fdd6] shadow-lg shadow-[#5a6344]/10 transition-transform hover:scale-[0.98]"
            >
              <Plus className="h-4 w-4" />
              New Generation
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

