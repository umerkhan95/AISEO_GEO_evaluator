import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search as SearchIcon,
  Filter,
  Star,
  Clock,
  Tag,
  ChevronDown,
  BookOpen,
  Loader2,
} from 'lucide-react'
import { searchGuidelines, type Guideline } from '@/lib/api'
import { cn } from '@/lib/utils'

const categories = [
  { value: '', label: 'All Categories' },
  { value: 'universal_seo_geo', label: 'Universal SEO/GEO' },
  { value: 'industry_specific', label: 'Industry Specific' },
  { value: 'technical', label: 'Technical' },
  { value: 'citation_optimization', label: 'Citation Optimization' },
  { value: 'metrics', label: 'Metrics' },
]

const priorities = [
  { value: '', label: 'All Priorities' },
  { value: 'critical', label: 'Critical', color: 'text-red-500' },
  { value: 'high', label: 'High', color: 'text-orange-500' },
  { value: 'medium', label: 'Medium', color: 'text-yellow-500' },
  { value: 'low', label: 'Low', color: 'text-green-500' },
]

const complexities = [
  { value: '', label: 'All Complexities' },
  { value: 'easy', label: 'Easy' },
  { value: 'moderate', label: 'Moderate' },
  { value: 'complex', label: 'Complex' },
]

function GuidelineCard({ guideline }: { guideline: Guideline }) {
  const [expanded, setExpanded] = useState(false)

  const priorityColor = {
    critical: 'bg-red-500/10 text-red-500 border-red-500/30',
    high: 'bg-orange-500/10 text-orange-500 border-orange-500/30',
    medium: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30',
    low: 'bg-green-500/10 text-green-500 border-green-500/30',
  }[guideline.priority] || 'bg-gray-500/10 text-gray-500 border-gray-500/30'

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="glass rounded-xl p-5 border border-white/10 hover:border-white/20 transition-all duration-300"
    >
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={cn(
              'px-2.5 py-1 rounded-full text-xs font-medium border',
              priorityColor
            )}
          >
            {guideline.priority}
          </span>
          <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary border border-primary/30">
            {guideline.category.replace(/_/g, ' ')}
          </span>
          <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-white/5 text-muted-foreground border border-white/10">
            {guideline.implementation_complexity}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-amber-500">
          <Star className="w-4 h-4 fill-current" />
          <span className="text-sm font-medium">
            {(guideline.confidence_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <p
        className={cn(
          'text-foreground mb-3 leading-relaxed',
          !expanded && 'line-clamp-3'
        )}
      >
        {guideline.content}
      </p>

      {guideline.content.length > 200 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-primary text-sm font-medium hover:underline mb-3"
        >
          {expanded ? 'Show less' : 'Read more'}
        </button>
      )}

      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <BookOpen className="w-3.5 h-3.5" />
          {guideline.source_section}
        </span>
        {guideline.page_numbers.length > 0 && (
          <span className="flex items-center gap-1">
            <Tag className="w-3.5 h-3.5" />
            Pages: {guideline.page_numbers.join(', ')}
          </span>
        )}
        {guideline.industries.length > 0 && (
          <span className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5" />
            {guideline.industries.join(', ')}
          </span>
        )}
      </div>
    </motion.div>
  )
}

export default function Search() {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('')
  const [priority, setPriority] = useState('')
  const [complexity, setComplexity] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['search', query, category, priority, complexity],
    queryFn: () =>
      searchGuidelines(query, {
        category: category || undefined,
        priority: priority || undefined,
        complexity: complexity || undefined,
        limit: 20,
      }),
    enabled: query.length >= 2,
  })

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">
          Search Guidelines
        </h1>
        <p className="text-muted-foreground text-lg">
          Find optimization strategies and best practices
        </p>
      </motion.div>

      {/* Search Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-6"
      >
        <div className="relative">
          <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for GEO/SEO guidelines..."
            className="w-full h-14 pl-12 pr-4 rounded-2xl glass border border-white/10 focus:border-primary/50 focus:ring-2 focus:ring-primary/20 bg-transparent text-foreground placeholder:text-muted-foreground outline-none transition-all"
          />
          {(isLoading || isFetching) && (
            <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-primary animate-spin" />
          )}
        </div>
      </motion.div>

      {/* Filters Toggle */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-6"
      >
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl glass border border-white/10 hover:border-white/20 transition-all text-sm font-medium text-foreground"
        >
          <Filter className="w-4 h-4" />
          Filters
          <ChevronDown
            className={cn(
              'w-4 h-4 transition-transform',
              showFilters && 'rotate-180'
            )}
          />
        </button>

        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="h-12 px-4 rounded-xl glass border border-white/10 bg-transparent text-foreground outline-none focus:border-primary/50"
                >
                  {categories.map((cat) => (
                    <option key={cat.value} value={cat.value} className="bg-background">
                      {cat.label}
                    </option>
                  ))}
                </select>

                <select
                  value={priority}
                  onChange={(e) => setPriority(e.target.value)}
                  className="h-12 px-4 rounded-xl glass border border-white/10 bg-transparent text-foreground outline-none focus:border-primary/50"
                >
                  {priorities.map((p) => (
                    <option key={p.value} value={p.value} className="bg-background">
                      {p.label}
                    </option>
                  ))}
                </select>

                <select
                  value={complexity}
                  onChange={(e) => setComplexity(e.target.value)}
                  className="h-12 px-4 rounded-xl glass border border-white/10 bg-transparent text-foreground outline-none focus:border-primary/50"
                >
                  {complexities.map((c) => (
                    <option key={c.value} value={c.value} className="bg-background">
                      {c.label}
                    </option>
                  ))}
                </select>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Results */}
      <div className="space-y-4">
        {query.length < 2 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-16"
          >
            <SearchIcon className="w-16 h-16 mx-auto text-muted-foreground/30 mb-4" />
            <p className="text-muted-foreground text-lg">
              Start typing to search guidelines
            </p>
            <p className="text-muted-foreground/60 text-sm mt-2">
              Enter at least 2 characters
            </p>
          </motion.div>
        ) : data?.results?.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-16"
          >
            <SearchIcon className="w-16 h-16 mx-auto text-muted-foreground/30 mb-4" />
            <p className="text-muted-foreground text-lg">No results found</p>
            <p className="text-muted-foreground/60 text-sm mt-2">
              Try adjusting your search query or filters
            </p>
          </motion.div>
        ) : (
          <>
            {data?.total_found !== undefined && (
              <p className="text-sm text-muted-foreground mb-4">
                Found {data.total_found} guidelines
              </p>
            )}
            <AnimatePresence>
              {data?.results?.map((guideline) => (
                <GuidelineCard key={guideline.guideline_id} guideline={guideline} />
              ))}
            </AnimatePresence>
          </>
        )}
      </div>
    </div>
  )
}
