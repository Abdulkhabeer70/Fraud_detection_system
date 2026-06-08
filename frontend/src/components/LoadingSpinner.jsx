import { Loader2 } from 'lucide-react'

function LoadingSpinner({ text = 'Loading...', size = 'default' }) {
  const sizeClasses = {
    small: 'w-5 h-5',
    default: 'w-8 h-8',
    large: 'w-12 h-12',
  }

  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4 animate-fade-in">
      <div className="relative">
        <div className={`${sizeClasses[size]} rounded-full border-2 border-slate-700 border-t-emerald-500 animate-spin`} />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
        </div>
      </div>
      <p className="text-sm text-slate-400 font-medium">{text}</p>
    </div>
  )
}

export function LoadingSkeleton({ rows = 3 }) {
  return (
    <div className="space-y-3 animate-fade-in">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="loading-shimmer rounded-xl h-12" />
      ))}
    </div>
  )
}

export default LoadingSpinner
