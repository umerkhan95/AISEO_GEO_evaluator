import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ProgressUpdate {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  timestamp: string;
}

interface ChunkStatus {
  chunk_id: string;
  section_title: string;
  status: string;
  geo_score_after: number;
  geo_score_before?: number;
  processing_time_ms?: number;
}

interface CrawlStats {
  title: string;
  word_count: number;
  crawl_time_ms: number;
  pages_crawled: number;
  content_length: number;
  crawler: string;
  links_count: number;
  images_count: number;
  chunks_generated: number;
  pages_urls?: string[];
}

interface JobStats {
  industry: string | null;
  industry_confidence: number;
  total_chunks: number;
  completed_chunks: number;
  original_geo_score: number;
  optimized_geo_score: number;
  status: string;
  crawl_stats?: CrawlStats;
}

interface JobResult {
  job_id: string;
  final_markdown: string;
  report_json: {
    scores: {
      original_geo_score: number;
      optimized_geo_score: number;
      improvement_pct: number;
    };
  };
}

export default function Optimize() {
  const [url, setUrl] = useState('');
  const [maxPages, setMaxPages] = useState(5);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [message, setMessage] = useState('');
  const [chunks, setChunks] = useState<ChunkStatus[]>([]);
  const [result, setResult] = useState<JobResult | null>(null);
  const [_logs, setLogs] = useState<ProgressUpdate[]>([]);
  const [jobStats, setJobStats] = useState<JobStats | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  const API_BASE = 'http://localhost:8001';

  const startOptimization = async () => {
    if (!url.trim()) return;

    setIsProcessing(true);
    setProgress(0);
    setStatus('starting');
    setMessage('Initializing...');
    setChunks([]);
    setResult(null);
    setLogs([]);
    setJobStats(null);
    setStartTime(Date.now());
    setElapsedTime(0);

    try {
      const response = await fetch(`${API_BASE}/api/v2/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim(), max_pages: maxPages }),
      });

      const data = await response.json();
      setJobId(data.job_id);

      // Connect to WebSocket for updates
      connectWebSocket(data.job_id);

      // Start polling for status
      pollJobStatus(data.job_id);
    } catch (error) {
      console.error('Error starting optimization:', error);
      setStatus('error');
      setMessage('Failed to start optimization');
      setIsProcessing(false);
    }
  };

  const connectWebSocket = (jobId: string) => {
    const ws = new WebSocket(`ws://localhost:8001/ws/${jobId}`);

    ws.onmessage = (event) => {
      const update: ProgressUpdate = JSON.parse(event.data);
      setProgress(update.progress);
      setStatus(update.status);
      setMessage(update.message);
      setLogs((prev) => [...prev, update]);

      if (update.status === 'completed' || update.status === 'failed') {
        setIsProcessing(false);
        if (update.status === 'completed') {
          fetchResult(jobId);
        }
      }
    };

    ws.onerror = () => {
      console.log('WebSocket error, falling back to polling');
    };

    wsRef.current = ws;
  };

  const pollJobStatus = async (jobId: string) => {
    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/v2/jobs/${jobId}`);
        const data = await response.json();

        // Update job stats
        setJobStats({
          industry: data.industry,
          industry_confidence: data.industry_confidence || 0,
          total_chunks: data.total_chunks || 0,
          completed_chunks: data.completed_chunks || 0,
          original_geo_score: data.original_geo_score || 0,
          optimized_geo_score: data.optimized_geo_score || 0,
          status: data.status,
          crawl_stats: data.crawl_stats || undefined,
        });

        // Update status and message based on job status
        setStatus(data.status);
        if (data.status === 'crawling') {
          setMessage('Crawling website with Crawl4AI...');
        } else if (data.status === 'classifying') {
          setMessage(`Classifying industry${data.industry ? `: ${data.industry}` : '...'}`);
        } else if (data.status === 'processing') {
          setMessage(`Processing ${data.completed_chunks}/${data.total_chunks} chunks...`);
        } else if (data.status === 'assembling') {
          setMessage('Assembling final output...');
        }

        if (data.chunks) {
          setChunks(data.chunks);
        }

        // Calculate progress based on status
        let calculatedProgress = 0;
        if (data.status === 'crawling') calculatedProgress = 10;
        else if (data.status === 'classifying') calculatedProgress = 25;
        else if (data.status === 'processing') {
          calculatedProgress = 30 + (data.total_chunks > 0
            ? Math.round((data.completed_chunks / data.total_chunks) * 55)
            : 0);
        }
        else if (data.status === 'assembling') calculatedProgress = 90;
        else if (data.status === 'completed') calculatedProgress = 100;

        setProgress(calculatedProgress);

        // Update elapsed time
        if (startTime) {
          setElapsedTime(Math.round((Date.now() - startTime) / 1000));
        }

        if (data.status === 'completed') {
          setIsProcessing(false);
          fetchResult(jobId);
          return;
        }

        if (data.status === 'failed') {
          setIsProcessing(false);
          setStatus('failed');
          setMessage(data.error_message || 'Optimization failed');
          return;
        }

        // Continue polling
        setTimeout(poll, 1500);
      } catch (error) {
        console.error('Polling error:', error);
        setTimeout(poll, 3000);
      }
    };

    poll();
  };

  const fetchResult = async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/v2/results/${jobId}`);
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error fetching result:', error);
    }
  };

  const downloadMarkdown = () => {
    if (!result) return;
    const blob = new Blob([result.final_markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `geo_optimized_${jobId}.md`;
    a.click();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isProcessing) {
      startOptimization();
    }
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000" />
        <div className="absolute top-1/2 left-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000" />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
            GEO Optimizer
          </h1>
          <p className="text-slate-400 text-lg">
            Transform your content for AI citations
          </p>
          <p className="text-slate-500 text-sm mt-2">
            Deep crawl up to 10 pages per website for comprehensive optimization
          </p>
        </motion.div>

        {/* Main Input */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-3xl mx-auto mb-12"
        >
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500" />
            <div className="relative bg-slate-800/90 backdrop-blur-xl rounded-2xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="flex-1 flex items-center bg-slate-900/50 rounded-xl px-4">
                  <svg className="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                  </svg>
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Enter URL to optimize..."
                    className="flex-1 bg-transparent text-white placeholder-slate-500 py-4 px-3 focus:outline-none text-lg"
                    disabled={isProcessing}
                  />
                </div>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={startOptimization}
                  disabled={isProcessing || !url.trim()}
                  className={`px-8 py-4 rounded-xl font-semibold text-white transition-all ${
                    isProcessing
                      ? 'bg-slate-600 cursor-not-allowed'
                      : 'bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400'
                  }`}
                >
                  {isProcessing ? (
                    <span className="flex items-center gap-2">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Processing
                    </span>
                  ) : (
                    'Optimize'
                  )}
                </motion.button>
              </div>

              {/* Max Pages Slider */}
              <div className="flex items-center gap-4 px-2">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span className="text-sm text-slate-400">Pages to crawl:</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={maxPages}
                  onChange={(e) => setMaxPages(parseInt(e.target.value))}
                  disabled={isProcessing}
                  className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
                <span className="text-sm font-medium text-cyan-400 w-8 text-center">{maxPages}</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Agentic Progress Section */}
        <AnimatePresence>
          {(isProcessing || jobStats) && !result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-4xl mx-auto mb-8"
            >
              <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
                {/* Header with Timer */}
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <svg className="w-5 h-5 text-cyan-400 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Agentic Workflow
                  </h2>
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-slate-400">
                      Elapsed: <span className="text-cyan-400 font-mono">{elapsedTime}s</span>
                    </span>
                    <span className="text-sm bg-cyan-500/20 text-cyan-400 px-3 py-1 rounded-full font-medium">
                      {progress}%
                    </span>
                  </div>
                </div>

                {/* Main Progress Bar */}
                <div className="mb-6">
                  <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <p className="text-sm text-slate-400 mt-2">{message}</p>
                </div>

                {/* Workflow Steps */}
                <div className="grid grid-cols-5 gap-2 mb-6">
                  {[
                    { key: 'crawling', label: 'Crawl', icon: '🕷️', desc: 'Crawl4AI' },
                    { key: 'classifying', label: 'Classify', icon: '🏷️', desc: 'Industry' },
                    { key: 'processing', label: 'Process', icon: '⚙️', desc: 'GPT-4o' },
                    { key: 'assembling', label: 'Assemble', icon: '📝', desc: 'Markdown' },
                    { key: 'completed', label: 'Done', icon: '✅', desc: 'Complete' },
                  ].map((step, idx) => {
                    const steps = ['crawling', 'classifying', 'processing', 'assembling', 'completed'];
                    const currentIdx = steps.indexOf(status);
                    const isActive = status === step.key;
                    const isComplete = currentIdx > idx || status === 'completed';

                    return (
                      <div
                        key={step.key}
                        className={`text-center p-3 rounded-xl transition-all ${
                          isActive
                            ? 'bg-cyan-500/20 border border-cyan-500/50 scale-105'
                            : isComplete
                            ? 'bg-green-500/10 border border-green-500/30'
                            : 'bg-slate-700/30 border border-slate-600/30'
                        }`}
                      >
                        <div className={`text-2xl mb-1 ${isActive ? 'animate-bounce' : ''}`}>
                          {isComplete && !isActive ? '✓' : step.icon}
                        </div>
                        <div className={`text-xs font-medium ${
                          isActive ? 'text-cyan-400' : isComplete ? 'text-green-400' : 'text-slate-500'
                        }`}>
                          {step.label}
                        </div>
                        <div className="text-[10px] text-slate-500">{step.desc}</div>
                      </div>
                    );
                  })}
                </div>

                {/* Crawl Stats */}
                {jobStats?.crawl_stats && jobStats.crawl_stats.crawl_time_ms > 0 && (
                  <div className="mb-6 p-4 bg-slate-900/30 rounded-xl border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-lg">🕷️</span>
                      <h3 className="text-sm font-medium text-slate-300">Crawl4AI Statistics</h3>
                      <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full">
                        {jobStats.crawl_stats.crawler || 'Crawl4AI'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="text-center">
                        <div className="text-lg font-bold text-cyan-400">
                          {((jobStats.crawl_stats.crawl_time_ms || 0) / 1000).toFixed(2)}s
                        </div>
                        <div className="text-xs text-slate-500">Crawl Time</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-purple-400">
                          {(jobStats.crawl_stats.word_count || 0).toLocaleString()}
                        </div>
                        <div className="text-xs text-slate-500">Words Extracted</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-pink-400">
                          {jobStats.crawl_stats.chunks_generated || 0}
                        </div>
                        <div className="text-xs text-slate-500">Chunks Generated</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-amber-400">
                          {jobStats.crawl_stats.links_count || 0}
                        </div>
                        <div className="text-xs text-slate-500">Links Found</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Stats Grid */}
                {jobStats && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                    <div className="bg-slate-900/50 rounded-xl p-3">
                      <div className="text-xs text-slate-500 mb-1">Industry</div>
                      <div className="text-sm font-medium text-white">
                        {jobStats.industry || '—'}
                      </div>
                      {jobStats.industry_confidence > 0 && (
                        <div className="text-xs text-cyan-400">
                          {(jobStats.industry_confidence * 100).toFixed(0)}% confidence
                        </div>
                      )}
                    </div>
                    <div className="bg-slate-900/50 rounded-xl p-3">
                      <div className="text-xs text-slate-500 mb-1">Chunks</div>
                      <div className="text-sm font-medium text-white">
                        {jobStats.completed_chunks} / {jobStats.total_chunks}
                      </div>
                      <div className="text-xs text-purple-400">
                        {jobStats.total_chunks > 0 ? Math.round((jobStats.completed_chunks / jobStats.total_chunks) * 100) : 0}% processed
                      </div>
                    </div>
                    <div className="bg-slate-900/50 rounded-xl p-3">
                      <div className="text-xs text-slate-500 mb-1">Original Score</div>
                      <div className="text-sm font-medium text-red-400">
                        {jobStats.original_geo_score.toFixed(1)}/10
                      </div>
                      <div className="text-xs text-slate-500">Before GEO</div>
                    </div>
                    <div className="bg-slate-900/50 rounded-xl p-3">
                      <div className="text-xs text-slate-500 mb-1">Current Score</div>
                      <div className="text-sm font-medium text-green-400">
                        {jobStats.optimized_geo_score.toFixed(1)}/10
                      </div>
                      <div className="text-xs text-green-400">
                        {jobStats.optimized_geo_score > jobStats.original_geo_score
                          ? `+${((jobStats.optimized_geo_score - jobStats.original_geo_score) / Math.max(jobStats.original_geo_score, 0.1) * 100).toFixed(0)}%`
                          : 'Processing...'}
                      </div>
                    </div>
                  </div>
                )}

                {/* Chunk Processing */}
                {chunks.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <h3 className="text-sm font-medium text-slate-300">Chunk Processing</h3>
                      <span className="text-xs text-slate-500">
                        {chunks.filter(c => c.status === 'completed').length}/{chunks.length} complete
                      </span>
                    </div>
                    <div className="space-y-2">
                      {chunks.map((chunk, idx) => (
                        <motion.div
                          key={chunk.chunk_id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.1 }}
                          className={`p-3 rounded-lg border ${
                            chunk.status === 'completed'
                              ? 'bg-green-500/5 border-green-500/20'
                              : chunk.status === 'processing'
                              ? 'bg-cyan-500/5 border-cyan-500/20'
                              : 'bg-slate-800/50 border-slate-700/50'
                          }`}
                        >
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-3">
                              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                                chunk.status === 'completed'
                                  ? 'bg-green-500/20 text-green-400'
                                  : chunk.status === 'processing'
                                  ? 'bg-cyan-500/20 text-cyan-400 animate-spin'
                                  : 'bg-slate-700/50 text-slate-500'
                              }`}>
                                {chunk.status === 'completed' ? '✓' : chunk.status === 'processing' ? '◌' : idx + 1}
                              </div>
                              <div>
                                <div className="text-sm font-medium text-white">{chunk.section_title}</div>
                                <div className="text-xs text-slate-500">
                                  {chunk.status === 'completed'
                                    ? 'Optimized with GEO guidelines'
                                    : chunk.status === 'processing'
                                    ? 'Applying GEO optimization...'
                                    : 'Pending'}
                                </div>
                              </div>
                            </div>
                            {chunk.status === 'completed' && (
                              <div className="text-right">
                                <div className="text-lg font-bold text-green-400">
                                  {chunk.geo_score_after.toFixed(1)}
                                </div>
                                <div className="text-xs text-slate-500">GEO Score</div>
                              </div>
                            )}
                            {chunk.status === 'processing' && (
                              <div className="flex items-center gap-2">
                                <div className="w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
                                <span className="text-xs text-cyan-400">Processing</span>
                              </div>
                            )}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="max-w-4xl mx-auto"
            >
              {/* Score Cards */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl p-6 border border-slate-700/50 text-center">
                  <div className="text-3xl font-bold text-red-400">
                    {result.report_json.scores.original_geo_score.toFixed(1)}
                  </div>
                  <div className="text-slate-400 text-sm">Original Score</div>
                </div>
                <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl p-6 border border-green-500/30 text-center">
                  <div className="text-3xl font-bold text-green-400">
                    {result.report_json.scores.optimized_geo_score.toFixed(1)}
                  </div>
                  <div className="text-slate-400 text-sm">Optimized Score</div>
                </div>
                <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl p-6 border border-cyan-500/30 text-center">
                  <div className="text-3xl font-bold text-cyan-400">
                    +{result.report_json.scores.improvement_pct.toFixed(0)}%
                  </div>
                  <div className="text-slate-400 text-sm">Improvement</div>
                </div>
              </div>

              {/* Download Button */}
              <div className="flex justify-center gap-4 mb-6">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={downloadMarkdown}
                  className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl font-semibold text-white flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download Markdown
                </motion.button>
              </div>

              {/* Preview */}
              <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-lg font-semibold text-white mb-4">Preview</h3>
                <div className="bg-slate-900/50 rounded-xl p-4 max-h-96 overflow-y-auto">
                  <pre className="text-sm text-slate-300 whitespace-pre-wrap font-mono">
                    {result.final_markdown.slice(0, 2000)}
                    {result.final_markdown.length > 2000 && '...\n\n[Download full content]'}
                  </pre>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <style>{`
        @keyframes blob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </div>
  );
}
