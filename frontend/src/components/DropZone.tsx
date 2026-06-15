import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, FileCode, CheckCircle, RefreshCw, XCircle } from 'lucide-react';

interface DropZoneProps {
  onAnalyzeFile: (file: File) => void;
  onAnalyzeText: (text: string) => void;
  loading: boolean;
  onReset: () => void;
  hasResult: boolean;
}

export const DropZone: React.FC<DropZoneProps> = ({
  onAnalyzeFile,
  onAnalyzeText,
  loading,
  onReset,
  hasResult,
}) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'paste'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState<string>('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxFiles: 1,
    disabled: loading || hasResult,
  });

  const handleAnalyze = () => {
    if (activeTab === 'upload' && file) {
      onAnalyzeFile(file);
    } else if (activeTab === 'paste' && text.trim()) {
      onAnalyzeText(text);
    }
  };

  const handleClear = () => {
    setFile(null);
    setText('');
    onReset();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const isButtonDisabled = loading || (activeTab === 'upload' && !file) || (activeTab === 'paste' && !text.trim());

  return (
    <div className="w-full max-w-[720px] mx-auto bg-white dark:bg-[#1E293B] border border-gray-200 dark:border-[#334155] rounded-2xl p-6 transition-all duration-300">
      {/* Tabs - transparent background, border colors */}
      <div className="flex border-b border-gray-100 dark:border-[#334155] bg-transparent mb-6">
        <button
          onClick={() => !loading && !hasResult && setActiveTab('upload')}
          disabled={loading || hasResult}
          className={`flex-1 pb-3 text-sm font-semibold border-b-2 flex items-center justify-center gap-2 transition-all duration-200 ${
            activeTab === 'upload'
              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-gray-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'
          } ${loading || hasResult ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <Upload className="w-4 h-4" />
          Upload File
        </button>
        <button
          onClick={() => !loading && !hasResult && setActiveTab('paste')}
          disabled={loading || hasResult}
          className={`flex-1 pb-3 text-sm font-semibold border-b-2 flex items-center justify-center gap-2 transition-all duration-200 ${
            activeTab === 'paste'
              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-gray-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'
          } ${loading || hasResult ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <FileText className="w-4 h-4" />
          Paste Text
        </button>
      </div>

      {/* Main Content Area */}
      <div className="mb-6">
        {activeTab === 'upload' ? (
          <div>
            {!file ? (
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-2xl p-8 md:p-12 text-center cursor-pointer transition-all duration-300 ${
                  isDragActive
                    ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/20 scale-[1.01]'
                    : 'border-[#3B82F6] dark:border-[#3B82F6] hover:bg-blue-50/30 dark:hover:bg-slate-800/20'
                } ${hasResult ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <input {...getInputProps()} />
                <div className="p-4 bg-[#EFF6FF] dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-inner">
                  <Upload className="w-8 h-8" />
                </div>
                <p className="text-base font-semibold text-slate-700 dark:text-slate-200 mb-1">
                  Drag & drop your resume here, or <span className="text-blue-500 dark:text-blue-400 font-bold hover:underline">browse</span>
                </p>
                <p className="text-xs text-slate-400 dark:text-slate-300">
                  Supports PDF and DOCX files up to 5MB
                </p>
              </div>
            ) : (
              /* File Preview Box - bg-gray-50/bg-[#0F172A], border-gray-100/border-[#334155] */
              <div className="bg-gray-50 dark:bg-[#0F172A] border border-gray-100 dark:border-[#334155] rounded-xl p-4 flex items-center justify-between gap-3 animate-fade-in">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="p-3 bg-blue-500/10 dark:bg-blue-400/10 text-blue-600 dark:text-blue-400 rounded-xl flex-shrink-0">
                    <FileCode className="w-8 h-8" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-slate-800 dark:text-slate-100 truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-slate-550 dark:text-slate-300">
                      {formatFileSize(file.size)} • {file.name.split('.').pop()?.toUpperCase()}
                    </p>
                  </div>
                </div>
                {!loading && !hasResult && (
                  <button
                    onClick={() => setFile(null)}
                    className="p-1.5 hover:bg-gray-200 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 rounded-full transition-colors focus:outline-none"
                  >
                    <XCircle className="w-5 h-5" />
                  </button>
                )}
              </div>
            )}

            {fileRejections && fileRejections.length > 0 && (
              <div className="mt-3 text-sm text-rose-500 flex items-center gap-1.5 px-3 py-2 bg-rose-500/10 rounded-xl border border-rose-500/20">
                <span className="font-semibold">Error:</span> Only PDF and DOCX file formats are supported.
              </div>
            )}
          </div>
        ) : (
          <div className="relative">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={loading || hasResult}
              placeholder="Paste the plain text of the resume here (e.g. contact info, work experience, education)..."
              rows={8}
              className="w-full px-4 py-3.5 border border-slate-300 dark:border-slate-700 bg-white/50 dark:bg-slate-900/50 rounded-2xl text-slate-850 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 text-sm resize-y min-h-[160px] max-h-[400px] transition-all duration-200 shadow-inner disabled:opacity-75 disabled:cursor-not-allowed"
            />
            {text.trim() && !loading && !hasResult && (
              <button
                onClick={() => setText('')}
                className="absolute top-3.5 right-3.5 p-1 bg-slate-100 hover:bg-slate-200 dark:bg-slate-850 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-300 rounded-full transition-colors focus:outline-none"
              >
                <XCircle className="w-4 h-4" />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Actions Section */}
      <div className="flex flex-col sm:flex-row items-center gap-3">
        {!hasResult ? (
          <button
            onClick={handleAnalyze}
            disabled={isButtonDisabled}
            className={`w-full py-3 rounded-xl font-semibold text-white bg-blue-600 hover:bg-blue-700 transition-colors duration-200 flex items-center justify-center gap-2.5 shadow-md ${
              isButtonDisabled ? 'opacity-50 cursor-not-allowed bg-blue-600/60' : 'cursor-pointer hover:scale-[1.01]'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <CheckCircle className="w-5 h-5" />
                <span>Analyze Resume</span>
              </>
            )}
          </button>
        ) : (
          <button
            onClick={handleClear}
            className="w-full py-3 px-6 rounded-xl bg-slate-105 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700/80 text-slate-700 dark:text-slate-200 font-semibold text-sm flex items-center justify-center gap-2 border border-slate-200 dark:border-slate-700 hover:shadow-md transition-all duration-300 cursor-pointer"
          >
            <RefreshCw className="w-4 h-4 animate-spin-reverse" />
            Analyze Another Resume
          </button>
        )}
      </div>
    </div>
  );
};
