import { X, WifiOff, FileX, AlertTriangle, Clock } from "lucide-react";

interface ErrorBannerProps {
  message:   string;
  onDismiss: () => void;
}

// Pick an icon based on the error message content
function getErrorIcon(message: string) {
  const m = message.toLowerCase();
  if (m.includes("network") || m.includes("server") || m.includes("reach"))
    return <WifiOff size={18} className="shrink-0" />;
  if (m.includes("file") || m.includes("5 mb") || m.includes("pdf") || m.includes("docx"))
    return <FileX size={18} className="shrink-0" />;
  if (m.includes("timeout") || m.includes("waking") || m.includes("timed"))
    return <Clock size={18} className="shrink-0" />;
  return <AlertTriangle size={18} className="shrink-0" />;
}

// One helpful hint per error type shown below the message
function getHint(message: string): string | null {
  const m = message.toLowerCase();
  if (m.includes("timeout") || m.includes("waking"))
    return "Free-tier servers sleep after inactivity. Your next request will succeed.";
  if (m.includes("5 mb"))
    return "Try compressing your PDF or saving a shorter version of your resume.";
  if (m.includes("pdf") || m.includes("docx"))
    return "Only .pdf and .docx files are supported. Save your resume in one of those formats.";
  if (m.includes("short") || m.includes("20 words"))
    return "Paste the full resume text — at least a few sentences are needed for an accurate prediction.";
  if (m.includes("network") || m.includes("reach"))
    return "The API server may be starting up. Wait 10 seconds and try again.";
  return null;
}

export default function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  const hint = getHint(message);

  return (
    <div
      role="alert"
      className="relative flex gap-3 rounded-2xl border border-red-500/30 bg-red-500/10 p-4 backdrop-blur-md text-red-300"
    >
      {/* Icon */}
      <span className="mt-0.5 text-red-400">
        {getErrorIcon(message)}
      </span>

      {/* Content */}
      <div className="flex-1 space-y-1">
        <p className="text-sm font-medium text-red-200">{message}</p>
        {hint && (
          <p className="text-xs text-red-400 leading-relaxed">{hint}</p>
        )}
      </div>

      {/* Dismiss */}
      <button
        onClick={onDismiss}
        aria-label="Dismiss error"
        className="absolute right-3 top-3 text-red-400 hover:text-red-200 transition-colors"
      >
        <X size={16} />
      </button>
    </div>
  );
}
