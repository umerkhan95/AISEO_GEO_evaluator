import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Database,
  Globe,
  Building2,
  Wrench,
  Quote,
  BarChart3,
  TrendingUp,
  Loader2,
} from 'lucide-react'
import { getCollections, type CollectionStats } from '@/lib/api'
import { cn, formatNumber } from '@/lib/utils'

const collectionIcons = {
  universal_seo_geo: { icon: Globe, color: 'from-purple-500 to-indigo-500' },
  industry_specific: { icon: Building2, color: 'from-blue-500 to-cyan-500' },
  technical: { icon: Wrench, color: 'from-orange-500 to-amber-500' },
  citation_optimization: { icon: Quote, color: 'from-green-500 to-emerald-500' },
  metrics: { icon: BarChart3, color: 'from-pink-500 to-rose-500' },
}

const collectionDescriptions = {
  universal_seo_geo:
    'Guidelines applicable to all industries and contexts for both SEO and GEO optimization',
  industry_specific:
    'Guidelines tailored to specific industries like healthcare, finance, and e-commerce',
  technical:
    'Technical implementation guidelines including schema markup, site architecture, and performance',
  citation_optimization:
    'Guidelines for citation formatting, source credibility, and reference optimization',
  metrics:
    'Measurement, analytics, and performance tracking guidelines for monitoring success',
}

function CollectionCard({ collection }: { collection: CollectionStats }) {
  const iconConfig = collectionIcons[collection.category as keyof typeof collectionIcons] || {
    icon: Database,
    color: 'from-gray-500 to-gray-600',
  }
  const Icon = iconConfig.icon
  const description =
    collectionDescriptions[collection.category as keyof typeof collectionDescriptions] ||
    'Collection of GEO/SEO guidelines'

  const isActive = collection.status !== 'not_created'
  const count = collection.points_count || 0

  return (
    <motion.div
      whileHover={{ y: -4 }}
      className="glass rounded-2xl p-6 border border-white/10 hover:border-white/20 transition-all duration-300"
    >
      <div className="flex items-start gap-4 mb-4">
        <div
          className={cn(
            'flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-r',
            iconConfig.color,
            !isActive && 'opacity-50'
          )}
        >
          <Icon className="w-7 h-7 text-white" />
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-foreground capitalize">
            {collection.category.replace(/_/g, ' ')}
          </h3>
          <p className="text-sm text-muted-foreground">
            {collection.collection_name}
          </p>
        </div>
      </div>

      <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
        {description}
      </p>

      <div className="flex items-center justify-between pt-4 border-t border-white/10">
        <div>
          <p className="text-2xl font-bold text-foreground">
            {formatNumber(count)}
          </p>
          <p className="text-xs text-muted-foreground">guidelines</p>
        </div>
        <div
          className={cn(
            'px-3 py-1 rounded-full text-xs font-medium',
            isActive
              ? 'bg-green-500/10 text-green-500'
              : 'bg-yellow-500/10 text-yellow-500'
          )}
        >
          {isActive ? 'Active' : 'Empty'}
        </div>
      </div>
    </motion.div>
  )
}

export default function Collections() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['collections'],
    queryFn: getCollections,
  })

  const totalGuidelines =
    data?.collections?.reduce((acc, col) => acc + (col.points_count || 0), 0) || 0

  const activeCollections =
    data?.collections?.filter((col) => col.status !== 'not_created').length || 0

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">Collections</h1>
        <p className="text-muted-foreground text-lg">
          Manage your knowledge base collections
        </p>
      </motion.div>

      {/* Stats Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
      >
        <div className="glass rounded-xl p-6 border border-white/10">
          <div className="flex items-center gap-3 mb-2">
            <Database className="w-5 h-5 text-primary" />
            <p className="text-sm text-muted-foreground">Total Collections</p>
          </div>
          <p className="text-3xl font-bold text-foreground">5</p>
        </div>
        <div className="glass rounded-xl p-6 border border-white/10">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-5 h-5 text-green-500" />
            <p className="text-sm text-muted-foreground">Active Collections</p>
          </div>
          <p className="text-3xl font-bold text-foreground">{activeCollections}</p>
        </div>
        <div className="glass rounded-xl p-6 border border-white/10">
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 className="w-5 h-5 text-amber-500" />
            <p className="text-sm text-muted-foreground">Total Guidelines</p>
          </div>
          <p className="text-3xl font-bold text-foreground">
            {formatNumber(totalGuidelines)}
          </p>
        </div>
      </motion.div>

      {/* Collections Grid */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
        </div>
      ) : error ? (
        <div className="text-center py-16">
          <Database className="w-16 h-16 mx-auto text-muted-foreground/30 mb-4" />
          <p className="text-muted-foreground">Failed to load collections</p>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {data?.collections?.map((collection, index) => (
            <motion.div
              key={collection.collection_name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
            >
              <CollectionCard collection={collection} />
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  )
}
