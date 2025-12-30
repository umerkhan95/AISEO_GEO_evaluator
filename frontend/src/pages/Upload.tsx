import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload as UploadIcon,
  File,
  Link,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
  ExternalLink,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { uploadPDF, processURL, getBatchStatus, type BatchStatus } from '@/lib/api'
import { cn } from '@/lib/utils'

interface FileUpload {
  file: File
  status: 'pending' | 'uploading' | 'success' | 'error'
  batchId?: string
  error?: string
  progress?: BatchStatus
}

export default function Upload() {
  const [files, setFiles] = useState<FileUpload[]>([])
  const [urlInput, setUrlInput] = useState('')
  const [urlProcessing, setUrlProcessing] = useState(false)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map((file) => ({
      file,
      status: 'pending' as const,
    }))
    setFiles((prev) => [...prev, ...newFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
  })

  const uploadFile = async (index: number) => {
    const upload = files[index]
    setFiles((prev) =>
      prev.map((f, i) => (i === index ? { ...f, status: 'uploading' } : f))
    )

    try {
      const result = await uploadPDF(upload.file)
      setFiles((prev) =>
        prev.map((f, i) =>
          i === index
            ? { ...f, status: 'success', batchId: result.batch_id }
            : f
        )
      )
      toast.success(`${upload.file.name} uploaded successfully!`)

      // Poll for status
      pollBatchStatus(result.batch_id, index)
    } catch (error) {
      setFiles((prev) =>
        prev.map((f, i) =>
          i === index
            ? {
                ...f,
                status: 'error',
                error: error instanceof Error ? error.message : 'Upload failed',
              }
            : f
        )
      )
      toast.error(`Failed to upload ${upload.file.name}`)
    }
  }

  const pollBatchStatus = async (batchId: string, index: number) => {
    const poll = async () => {
      try {
        const status = await getBatchStatus(batchId)
        setFiles((prev) =>
          prev.map((f, i) => (i === index ? { ...f, progress: status } : f))
        )

        if (status.status !== 'completed' && status.status !== 'failed') {
          setTimeout(poll, 2000)
        }
      } catch {
        // Ignore polling errors
      }
    }
    poll()
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const uploadAll = () => {
    files.forEach((file, index) => {
      if (file.status === 'pending') {
        uploadFile(index)
      }
    })
  }

  const handleURLSubmit = async () => {
    if (!urlInput.trim()) return

    setUrlProcessing(true)
    try {
      await processURL(urlInput)
      toast.success('URL processing started!')
      setUrlInput('')
    } catch {
      toast.error('Failed to process URL')
    } finally {
      setUrlProcessing(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">Upload Papers</h1>
        <p className="text-muted-foreground text-lg">
          Add GEO/SEO research papers to the knowledge base
        </p>
      </motion.div>

      {/* Dropzone */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-8"
      >
        <div
          {...getRootProps()}
          className={cn(
            'relative flex flex-col items-center justify-center h-64 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer',
            isDragActive
              ? 'border-primary bg-primary/5'
              : 'border-white/20 hover:border-white/40 bg-white/5'
          )}
        >
          <input {...getInputProps()} />
          <motion.div
            animate={{ scale: isDragActive ? 1.1 : 1 }}
            className="flex flex-col items-center"
          >
            <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-4">
              <UploadIcon className="w-8 h-8 text-primary" />
            </div>
            <p className="text-lg font-medium text-foreground mb-2">
              {isDragActive ? 'Drop your PDFs here' : 'Drag & drop PDF files'}
            </p>
            <p className="text-sm text-muted-foreground">
              or click to browse (max 50MB per file)
            </p>
          </motion.div>
        </div>
      </motion.div>

      {/* URL Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-8"
      >
        <div className="glass rounded-2xl p-6 border border-white/10">
          <div className="flex items-center gap-3 mb-4">
            <Link className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">
              Or process from URL
            </h2>
          </div>
          <div className="flex gap-3">
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="https://arxiv.org/pdf/..."
              className="flex-1 h-12 px-4 rounded-xl glass border border-white/10 focus:border-primary/50 bg-transparent text-foreground placeholder:text-muted-foreground outline-none"
            />
            <button
              onClick={handleURLSubmit}
              disabled={!urlInput.trim() || urlProcessing}
              className="h-12 px-6 rounded-xl bg-primary text-primary-foreground font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
            >
              {urlProcessing ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <ExternalLink className="w-5 h-5" />
              )}
              Process
            </button>
          </div>
        </div>
      </motion.div>

      {/* File List */}
      <AnimatePresence>
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-foreground">
                Files ({files.length})
              </h2>
              {files.some((f) => f.status === 'pending') && (
                <button
                  onClick={uploadAll}
                  className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-all"
                >
                  Upload All
                </button>
              )}
            </div>

            {files.map((upload, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="glass rounded-xl p-4 border border-white/10"
              >
                <div className="flex items-center gap-4">
                  <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                    <File className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-foreground truncate">
                      {upload.file.name}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {(upload.file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    {upload.status === 'pending' && (
                      <button
                        onClick={() => uploadFile(index)}
                        className="px-3 py-1.5 rounded-lg bg-primary/10 text-primary text-sm font-medium hover:bg-primary/20 transition-all"
                      >
                        Upload
                      </button>
                    )}
                    {upload.status === 'uploading' && (
                      <div className="flex items-center gap-2 text-primary">
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span className="text-sm">Uploading...</span>
                      </div>
                    )}
                    {upload.status === 'success' && (
                      <div className="flex items-center gap-2 text-green-500">
                        <CheckCircle className="w-5 h-5" />
                        <span className="text-sm">
                          {upload.progress?.status === 'completed'
                            ? `${upload.progress.guidelines_extracted} guidelines`
                            : upload.progress?.status || 'Processing...'}
                        </span>
                      </div>
                    )}
                    {upload.status === 'error' && (
                      <div className="flex items-center gap-2 text-red-500">
                        <AlertCircle className="w-5 h-5" />
                        <span className="text-sm">{upload.error}</span>
                      </div>
                    )}
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1.5 rounded-lg hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                {/* Progress Details */}
                {upload.progress && (
                  <div className="mt-3 pt-3 border-t border-white/10">
                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Status</p>
                        <p className="font-medium text-foreground capitalize">
                          {upload.progress.status}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Processed</p>
                        <p className="font-medium text-foreground">
                          {upload.progress.documents_processed}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Extracted</p>
                        <p className="font-medium text-foreground">
                          {upload.progress.guidelines_extracted}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Stored</p>
                        <p className="font-medium text-foreground">
                          {upload.progress.guidelines_stored}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
