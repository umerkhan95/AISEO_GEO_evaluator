import { Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Layout from './components/Layout'
import Optimize from './pages/Optimize'
import Dashboard from './pages/Dashboard'
import Search from './pages/Search'
import Upload from './pages/Upload'
import Collections from './pages/Collections'
import Chat from './pages/Chat'
import { ThemeProvider } from './hooks/useTheme'

function App() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-background">
        <Toaster
          position="top-right"
          toastOptions={{
            className: 'glass-dark',
            style: {
              background: 'hsl(var(--card))',
              color: 'hsl(var(--foreground))',
              border: '1px solid hsl(var(--border))',
            },
          }}
        />
        <Routes>
          {/* Optimize page is now the main landing page */}
          <Route path="/" element={<Optimize />} />

          {/* Other pages with layout */}
          <Route path="/app" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="search" element={<Search />} />
            <Route path="upload" element={<Upload />} />
            <Route path="collections" element={<Collections />} />
            <Route path="chat" element={<Chat />} />
          </Route>
        </Routes>
      </div>
    </ThemeProvider>
  )
}

export default App
