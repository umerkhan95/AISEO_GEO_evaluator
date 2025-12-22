import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard,
  Search,
  Upload,
  Database,
  MessageSquare,
  Moon,
  Sun,
  Sparkles,
} from 'lucide-react'
import { useTheme } from '@/hooks/useTheme'
import { cn } from '@/lib/utils'
import ThreeBackground from './ThreeBackground'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Search', href: '/search', icon: Search },
  { name: 'Upload', href: '/upload', icon: Upload },
  { name: 'Collections', href: '/collections', icon: Database },
  { name: 'AI Chat', href: '/chat', icon: MessageSquare },
]

export default function Layout() {
  const { resolvedTheme, setTheme } = useTheme()
  const location = useLocation()

  return (
    <div className="relative min-h-screen">
      <ThreeBackground />

      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-50 w-64 glass-dark border-r border-white/5">
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center gap-3 px-6 py-6 border-b border-white/5">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-indigo-500 rounded-xl blur-lg opacity-50" />
              <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-r from-purple-500 to-indigo-500">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-lg font-semibold text-foreground">GEO/SEO KB</h1>
              <p className="text-xs text-muted-foreground">Knowledge Base</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <NavLink
                  key={item.name}
                  to={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                    isActive
                      ? 'bg-primary/10 text-primary'
                      : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
                  )}
                >
                  <item.icon className={cn('w-5 h-5', isActive && 'text-primary')} />
                  <span>{item.name}</span>
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-indicator"
                      className="absolute left-0 w-1 h-6 bg-primary rounded-r-full"
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )}
                </NavLink>
              )
            })}
          </nav>

          {/* Theme Toggle */}
          <div className="px-4 py-4 border-t border-white/5">
            <button
              onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
              className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all duration-200"
            >
              <AnimatePresence mode="wait">
                {resolvedTheme === 'dark' ? (
                  <motion.div
                    key="moon"
                    initial={{ rotate: -90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: 90, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Moon className="w-5 h-5" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="sun"
                    initial={{ rotate: 90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: -90, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Sun className="w-5 h-5" />
                  </motion.div>
                )}
              </AnimatePresence>
              <span>
                {resolvedTheme === 'dark' ? 'Dark Mode' : 'Light Mode'}
              </span>
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="pl-64">
        <div className="min-h-screen p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}
