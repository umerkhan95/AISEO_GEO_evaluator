import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { getShowcase, type ShowcaseWebsite } from '@/lib/api';

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function extractDomain(url: string): string {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname.replace('www.', '');
  } catch {
    return url;
  }
}

function WebsiteCard({ website, index }: { website: ShowcaseWebsite; index: number }) {
  const domain = extractDomain(website.url);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 hover:border-slate-600/50 transition-all group"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <a
            href={website.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-lg font-semibold text-white hover:text-cyan-400 transition-colors truncate block"
          >
            {domain}
          </a>
          <p className="text-sm text-slate-500 truncate mt-1">{website.url}</p>
        </div>
        <div className="ml-4 flex items-center gap-2">
          {website.industry && (
            <span className="px-3 py-1 bg-purple-500/20 text-purple-400 text-xs font-medium rounded-full">
              {website.industry}
            </span>
          )}
        </div>
      </div>

      {/* Scores */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center p-3 bg-slate-900/50 rounded-xl">
          <div className="text-2xl font-bold text-red-400">
            {website.original_score.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500">Original</div>
        </div>
        <div className="text-center p-3 bg-slate-900/50 rounded-xl border border-green-500/30">
          <div className="text-2xl font-bold text-green-400">
            {website.optimized_score.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500">Optimized</div>
        </div>
        <div className="text-center p-3 bg-slate-900/50 rounded-xl">
          <div className="text-2xl font-bold text-cyan-400">
            +{website.improvement_pct.toFixed(0)}%
          </div>
          <div className="text-xs text-slate-500">Improvement</div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>{website.chunks_processed} sections processed</span>
        <span>{formatDate(website.optimized_at)}</span>
      </div>
    </motion.div>
  );
}

export default function Showcase() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['showcase'],
    queryFn: () => getShowcase(100),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

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
            GEO Showcase
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Websites that have been optimized for AI citations using our GEO Optimizer.
            See the before and after scores.
          </p>
        </motion.div>

        {/* Stats Summary */}
        {data && data.total > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-4xl mx-auto mb-12"
          >
            <div className="bg-slate-800/30 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
                <div>
                  <div className="text-3xl font-bold text-white">{data.total}</div>
                  <div className="text-sm text-slate-400">Websites Optimized</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-green-400">
                    {(data.websites.reduce((acc, w) => acc + w.optimized_score, 0) / data.total).toFixed(1)}
                  </div>
                  <div className="text-sm text-slate-400">Avg. GEO Score</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-cyan-400">
                    +{(data.websites.reduce((acc, w) => acc + w.improvement_pct, 0) / data.total).toFixed(0)}%
                  </div>
                  <div className="text-sm text-slate-400">Avg. Improvement</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-purple-400">
                    {data.websites.reduce((acc, w) => acc + w.chunks_processed, 0)}
                  </div>
                  <div className="text-sm text-slate-400">Sections Processed</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-center items-center py-20">
            <div className="w-12 h-12 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {/* Error State */}
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <div className="text-6xl mb-4">:(</div>
            <p className="text-slate-400 text-lg">Failed to load showcase data</p>
            <p className="text-slate-500 text-sm mt-2">Please try again later</p>
          </motion.div>
        )}

        {/* Empty State */}
        {data && data.total === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <div className="text-6xl mb-4">:)</div>
            <p className="text-slate-400 text-lg">No websites optimized yet</p>
            <p className="text-slate-500 text-sm mt-2">
              Be the first to optimize your website!
            </p>
            <a
              href="/"
              className="inline-block mt-6 px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl font-semibold text-white hover:from-cyan-400 hover:to-purple-400 transition-all"
            >
              Optimize Now
            </a>
          </motion.div>
        )}

        {/* Website Grid */}
        {data && data.total > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
            {data.websites.map((website, index) => (
              <WebsiteCard key={website.job_id} website={website} index={index} />
            ))}
          </div>
        )}

        {/* Footer CTA */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center mt-16"
        >
          <p className="text-slate-500 mb-4">Want your website here?</p>
          <a
            href="/"
            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl font-semibold text-white hover:from-cyan-400 hover:to-purple-400 transition-all"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Optimize Your Website
          </a>
        </motion.div>
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
