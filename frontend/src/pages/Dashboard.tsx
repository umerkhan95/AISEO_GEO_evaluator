import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Database,
  FileText,
  TrendingUp,
  Sparkles,
  ArrowUpRight,
  Activity,
} from 'lucide-react'
import { getCollections, getStats } from '@/lib/api'
import { cn, formatNumber } from '@/lib/utils'

const stats = [
  {
    name: 'Total Guidelines',
    icon: FileText,
    color: 'from-purple-500 to-indigo-500',
    value: 0,
  },
  {
    name: 'Collections',
    icon: Database,
    color: 'from-blue-500 to-cyan-500',
    value: 5,
  },
  {
    name: 'Search Queries',
    icon: TrendingUp,
    color: 'from-green-500 to-emerald-500',
    value: 0,
  },
  {
    name: 'AI Responses',
    icon: Sparkles,
    color: 'from-orange-500 to-amber-500',
    value: 0,
  },
]

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
}

export default function Dashboard() {
  const { data: collectionsData } = useQuery({
    queryKey: ['collections'],
    queryFn: getCollections,
  })

  const { data: _statsData } = useQuery({
    queryKey: ['stats'],
    queryFn: getStats,
  })

  const totalGuidelines = collectionsData?.collections?.reduce(
    (acc, col) => acc + (col.points_count || 0),
    0
  ) || 0

  const updatedStats = stats.map((stat) => {
    if (stat.name === 'Total Guidelines') {
      return { ...stat, value: totalGuidelines }
    }
    return stat
  })

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">
          Knowledge Base Dashboard
        </h1>
        <p className="text-muted-foreground text-lg">
          Manage your GEO/SEO guidelines and research
        </p>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
      >
        {updatedStats.map((stat) => (
          <motion.div key={stat.name} variants={item}>
            <div className="glass rounded-2xl p-6 border border-white/10 hover:border-white/20 transition-all duration-300 group">
              <div className="flex items-center justify-between mb-4">
                <div
                  className={cn(
                    'flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-r',
                    stat.color
                  )}
                >
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
                <ArrowUpRight className="w-5 h-5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
              <p className="text-3xl font-bold text-foreground mb-1">
                {formatNumber(stat.value)}
              </p>
              <p className="text-sm text-muted-foreground">{stat.name}</p>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Collections Overview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass rounded-2xl p-6 border border-white/10"
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-foreground">
            Collections Overview
          </h2>
          <Activity className="w-5 h-5 text-muted-foreground" />
        </div>

        <div className="space-y-4">
          {collectionsData?.collections?.map((collection, index) => (
            <motion.div
              key={collection.collection_name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              className="flex items-center gap-4 p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                <Database className="w-5 h-5 text-primary" />
              </div>
              <div className="flex-1">
                <p className="font-medium text-foreground capitalize">
                  {collection.category.replace(/_/g, ' ')}
                </p>
                <p className="text-sm text-muted-foreground">
                  {collection.collection_name}
                </p>
              </div>
              <div className="text-right">
                <p className="font-semibold text-foreground">
                  {formatNumber(collection.points_count || 0)}
                </p>
                <p className="text-sm text-muted-foreground">guidelines</p>
              </div>
            </motion.div>
          )) || (
            <div className="text-center py-8 text-muted-foreground">
              <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No collections found. Start by uploading some PDFs!</p>
            </div>
          )}
        </div>
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8"
      >
        <a
          href="/upload"
          className="glass rounded-2xl p-6 border border-white/10 hover:border-primary/50 transition-all duration-300 group cursor-pointer"
        >
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-primary/10 mb-4 group-hover:scale-110 transition-transform">
            <FileText className="w-6 h-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">
            Upload Papers
          </h3>
          <p className="text-sm text-muted-foreground">
            Add new GEO/SEO research papers to the knowledge base
          </p>
        </a>

        <a
          href="/search"
          className="glass rounded-2xl p-6 border border-white/10 hover:border-primary/50 transition-all duration-300 group cursor-pointer"
        >
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-blue-500/10 mb-4 group-hover:scale-110 transition-transform">
            <TrendingUp className="w-6 h-6 text-blue-500" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">
            Search Guidelines
          </h3>
          <p className="text-sm text-muted-foreground">
            Find specific optimization strategies and best practices
          </p>
        </a>

        <a
          href="/chat"
          className="glass rounded-2xl p-6 border border-white/10 hover:border-primary/50 transition-all duration-300 group cursor-pointer"
        >
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-amber-500/10 mb-4 group-hover:scale-110 transition-transform">
            <Sparkles className="w-6 h-6 text-amber-500" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">
            Ask AI Agent
          </h3>
          <p className="text-sm text-muted-foreground">
            Get AI-powered insights and recommendations
          </p>
        </a>
      </motion.div>
    </div>
  )
}
