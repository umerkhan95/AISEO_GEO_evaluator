import { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface GlassCardProps {
  children: ReactNode
  className?: string
  hover?: boolean
  gradient?: boolean
  onClick?: () => void
}

export function GlassCard({
  children,
  className,
  hover = true,
  gradient = false,
  onClick,
}: GlassCardProps) {
  return (
    <motion.div
      whileHover={hover ? { y: -4, scale: 1.01 } : undefined}
      whileTap={onClick ? { scale: 0.98 } : undefined}
      onClick={onClick}
      className={cn(
        'relative overflow-hidden rounded-2xl backdrop-blur-xl transition-all duration-500',
        'bg-white/5 border border-white/10',
        'shadow-[0_8px_32px_rgba(0,0,0,0.12)]',
        hover && 'hover:bg-white/10 hover:border-white/20 hover:shadow-[0_16px_48px_rgba(0,0,0,0.2)]',
        gradient && 'before:absolute before:inset-0 before:-z-10 before:bg-gradient-to-br before:from-purple-500/10 before:via-transparent before:to-indigo-500/10',
        onClick && 'cursor-pointer',
        className
      )}
    >
      {/* Noise texture overlay */}
      <div className="absolute inset-0 opacity-[0.02] pointer-events-none noise" />

      {/* Inner glow */}
      <div className="absolute inset-px rounded-2xl bg-gradient-to-b from-white/10 to-transparent pointer-events-none" />

      {/* Content */}
      <div className="relative z-10">{children}</div>
    </motion.div>
  )
}

interface GlassButtonProps {
  children: ReactNode
  className?: string
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  onClick?: () => void
  type?: 'button' | 'submit'
}

export function GlassButton({
  children,
  className,
  variant = 'primary',
  size = 'md',
  disabled,
  onClick,
  type = 'button',
}: GlassButtonProps) {
  const variants = {
    primary: 'bg-primary text-primary-foreground hover:bg-primary/90',
    secondary: 'bg-white/10 text-foreground hover:bg-white/20 border border-white/10',
    ghost: 'bg-transparent text-foreground hover:bg-white/5',
  }

  const sizes = {
    sm: 'px-3 py-1.5 text-sm rounded-lg',
    md: 'px-4 py-2.5 text-sm rounded-xl',
    lg: 'px-6 py-3 text-base rounded-xl',
  }

  return (
    <motion.button
      type={type}
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        'font-medium transition-all duration-200',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        'focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 focus:ring-offset-background',
        variants[variant],
        sizes[size],
        className
      )}
    >
      {children}
    </motion.button>
  )
}

interface GlassInputProps {
  placeholder?: string
  value?: string
  onChange?: (value: string) => void
  type?: string
  className?: string
  icon?: ReactNode
}

export function GlassInput({
  placeholder,
  value,
  onChange,
  type = 'text',
  className,
  icon,
}: GlassInputProps) {
  return (
    <div className={cn('relative', className)}>
      {icon && (
        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground">
          {icon}
        </div>
      )}
      <input
        type={type}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        placeholder={placeholder}
        className={cn(
          'w-full h-12 rounded-xl backdrop-blur-xl transition-all duration-200',
          'bg-white/5 border border-white/10',
          'text-foreground placeholder:text-muted-foreground',
          'focus:bg-white/10 focus:border-primary/50 focus:ring-2 focus:ring-primary/20',
          'outline-none',
          icon ? 'pl-12 pr-4' : 'px-4'
        )}
      />
    </div>
  )
}

// Liquid Glass Filter for SVG effects
export function LiquidGlassFilter() {
  return (
    <svg style={{ display: 'none' }}>
      <filter
        id="glass-distortion"
        x="0%"
        y="0%"
        width="100%"
        height="100%"
        filterUnits="objectBoundingBox"
      >
        <feTurbulence
          type="fractalNoise"
          baseFrequency="0.001 0.005"
          numOctaves="1"
          seed="17"
          result="turbulence"
        />
        <feComponentTransfer in="turbulence" result="mapped">
          <feFuncR type="gamma" amplitude="1" exponent="10" offset="0.5" />
          <feFuncG type="gamma" amplitude="0" exponent="1" offset="0" />
          <feFuncB type="gamma" amplitude="0" exponent="1" offset="0.5" />
        </feComponentTransfer>
        <feGaussianBlur in="turbulence" stdDeviation="3" result="softMap" />
        <feSpecularLighting
          in="softMap"
          surfaceScale="5"
          specularConstant="1"
          specularExponent="100"
          lightingColor="white"
          result="specLight"
        >
          <fePointLight x="-200" y="-200" z="300" />
        </feSpecularLighting>
        <feComposite
          in="specLight"
          operator="arithmetic"
          k1="0"
          k2="1"
          k3="1"
          k4="0"
          result="litImage"
        />
        <feDisplacementMap
          in="SourceGraphic"
          in2="softMap"
          scale="200"
          xChannelSelector="R"
          yChannelSelector="G"
        />
      </filter>
    </svg>
  )
}
