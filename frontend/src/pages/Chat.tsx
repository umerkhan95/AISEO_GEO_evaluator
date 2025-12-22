import { useState, useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import {
  Send,
  Sparkles,
  User,
  Loader2,
  Lightbulb,
  Search,
  FileText,
  BarChart3,
} from 'lucide-react'
import { chatWithAgent } from '@/lib/api'
import { cn } from '@/lib/utils'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const suggestions = [
  {
    icon: Search,
    text: 'Find papers about citation optimization for B2B SaaS',
    color: 'text-purple-500',
  },
  {
    icon: Lightbulb,
    text: 'What are the best practices for healthcare GEO?',
    color: 'text-amber-500',
  },
  {
    icon: FileText,
    text: 'Show me technical implementation guidelines',
    color: 'text-blue-500',
  },
  {
    icon: BarChart3,
    text: 'How do I measure GEO success metrics?',
    color: 'text-green-500',
  },
]

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState<string>()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const mutation = useMutation({
    mutationFn: (message: string) => chatWithAgent(message, sessionId),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response },
      ])
      if (data.session_id) {
        setSessionId(data.session_id)
      }
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
        },
      ])
    },
  })

  const sendMessage = (message: string) => {
    if (!message.trim() || mutation.isPending) return

    setMessages((prev) => [...prev, { role: 'user', content: message }])
    setInput('')
    mutation.mutate(message)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  return (
    <div className="max-w-4xl mx-auto h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">AI Research Agent</h1>
        <p className="text-muted-foreground text-lg">
          Ask questions about GEO/SEO guidelines and research
        </p>
      </motion.div>

      {/* Messages Area */}
      <div className="flex-1 overflow-hidden rounded-2xl glass border border-white/10">
        <div className="h-full overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="h-full flex flex-col items-center justify-center"
            >
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full blur-xl opacity-50" />
                <div className="relative flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500">
                  <Sparkles className="w-10 h-10 text-white" />
                </div>
              </div>
              <h2 className="text-xl font-semibold text-foreground mb-2">
                Start a conversation
              </h2>
              <p className="text-muted-foreground text-center mb-8 max-w-md">
                I can help you search for papers, find guidelines, and provide
                insights about GEO and SEO optimization strategies.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
                {suggestions.map((suggestion, index) => (
                  <motion.button
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 * index }}
                    onClick={() => sendMessage(suggestion.text)}
                    className="flex items-center gap-3 p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all text-left group"
                  >
                    <suggestion.icon
                      className={cn('w-5 h-5', suggestion.color)}
                    />
                    <span className="text-sm text-foreground/80 group-hover:text-foreground transition-colors">
                      {suggestion.text}
                    </span>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          ) : (
            <AnimatePresence initial={false}>
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={cn(
                    'flex gap-4',
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  )}
                >
                  {message.role === 'assistant' && (
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-r from-purple-500 to-indigo-500 flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-white" />
                    </div>
                  )}
                  <div
                    className={cn(
                      'max-w-[80%] rounded-2xl px-5 py-3',
                      message.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-white/5 text-foreground border border-white/10'
                    )}
                  >
                    {message.role === 'assistant' ? (
                      <ReactMarkdown className="prose prose-sm prose-invert max-w-none">
                        {message.content}
                      </ReactMarkdown>
                    ) : (
                      <p>{message.content}</p>
                    )}
                  </div>
                  {message.role === 'user' && (
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                      <User className="w-4 h-4 text-foreground" />
                    </div>
                  )}
                </motion.div>
              ))}
              {mutation.isPending && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-4 justify-start"
                >
                  <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-r from-purple-500 to-indigo-500 flex items-center justify-center">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <div className="bg-white/5 text-foreground border border-white/10 rounded-2xl px-5 py-3">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">
                        Thinking...
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        onSubmit={handleSubmit}
        className="mt-4"
      >
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about GEO/SEO guidelines..."
            className="w-full h-14 pl-5 pr-14 rounded-2xl glass border border-white/10 focus:border-primary/50 focus:ring-2 focus:ring-primary/20 bg-transparent text-foreground placeholder:text-muted-foreground outline-none transition-all"
          />
          <button
            type="submit"
            disabled={!input.trim() || mutation.isPending}
            className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-xl bg-primary text-primary-foreground flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary/90 transition-all"
          >
            {mutation.isPending ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </motion.form>
    </div>
  )
}
