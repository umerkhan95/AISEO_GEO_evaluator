import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface Guideline {
  guideline_id: string
  content: string
  category: string
  source_section: string
  page_numbers: number[]
  industries: string[]
  confidence_score: number
  priority: string
  implementation_complexity: string
  related_guideline_ids: string[]
  similarity_score: number
}

export interface SearchResponse {
  query: string
  results: Guideline[]
  total_found: number
}

export interface CollectionStats {
  category: string
  collection_name: string
  vectors_count?: number
  points_count?: number
  status?: string
}

export interface BatchStatus {
  batch_id: string
  status: string
  documents_processed: number
  guidelines_extracted: number
  guidelines_stored: number
  errors: number
  started_at: string
  completed_at?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

// Search endpoints
export async function searchGuidelines(
  query: string,
  options?: {
    category?: string
    industries?: string[]
    priority?: string
    complexity?: string
    limit?: number
  }
): Promise<SearchResponse> {
  const response = await api.post('/search', {
    query,
    ...options,
  })
  return response.data
}

// Research endpoints
export async function researchPapers(
  topic: string,
  industry?: string,
  maxResults = 10
): Promise<unknown> {
  const response = await api.post('/research', {
    topic,
    industry,
    max_results: maxResults,
  })
  return response.data
}

// Chat endpoints
export async function chatWithAgent(
  message: string,
  sessionId?: string
): Promise<{ response: string; session_id?: string }> {
  const response = await api.post('/chat', {
    message,
    session_id: sessionId,
  })
  return response.data
}

// Collection endpoints
export async function getCollections(): Promise<{ collections: CollectionStats[] }> {
  const response = await api.get('/collections')
  return response.data
}

export async function getStats(): Promise<unknown> {
  const response = await api.get('/stats')
  return response.data
}

// Upload endpoints
export async function uploadPDF(file: File): Promise<{
  batch_id: string
  filename: string
  file_size_mb: number
  status: string
  message: string
}> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await axios.post(
    `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/upload`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

export async function getBatchStatus(batchId: string): Promise<BatchStatus> {
  const response = await api.get(`/batch/${batchId}`)
  return response.data
}

export async function processURL(url: string, filename?: string): Promise<{
  batch_id: string
  status: string
  url: string
  message: string
}> {
  const response = await api.post('/process-url', {
    url,
    filename,
  })
  return response.data
}

// Health check
export async function healthCheck(): Promise<unknown> {
  const response = await api.get('/health')
  return response.data
}

export default api
