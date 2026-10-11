import { type ChangeEvent, useCallback, useEffect, useState } from 'react'
import './App.css'

type DocumentItem = {
  id: string
  filename: string
  page_count: number
  processing_status: string
  extraction_method: string
  created_at: string
  file_size: number
  chunk_count: number
  ocr_page_count: number
  indexing_status: string
}

type PageRecord = {
  document_id: string
  page_number: number
  text: string
  extraction_method: string
  ocr_confidence: number | null
  extraction_error: string | null
  processing_status: string
}

type ChatResult = {
  answer: string
  citations: { document_id: string; document: string; page: number; chunk_id: number; passage: string }[]
  passages: { content: string; document_id: string; document: string; page: number; chunk_id: number; dense_score: number; keyword_score: number; final_score: number }[]
  source_documents: string[]
  generation_mode: string
  provider: string | null
  model: string | null
  insufficient_evidence: boolean
  provider_error: string | null
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options?.headers || {}),
    },
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || 'Request failed')
  }
  return (await response.json()) as T
}

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [selectedDocumentId, setSelectedDocumentId] = useState<string>('')
  const [page, setPage] = useState<PageRecord | null>(null)
  const [pageNumber, setPageNumber] = useState(1)
  const [query, setQuery] = useState('What are the main findings in this document?')
  const [chatResult, setChatResult] = useState<ChatResult | null>(null)
  const [searchQuery, setSearchQuery] = useState('document intelligence')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [compareQuestion, setCompareQuestion] = useState('Compare the main differences between these documents.')
  const [compareResult, setCompareResult] = useState<any>(null)
  const [compareDocumentIds, setCompareDocumentIds] = useState<string[]>([])
  const [visualQuestion, setVisualQuestion] = useState('Describe the chart, diagram, or visual structure on this page.')
  const [visualResult, setVisualResult] = useState<any>(null)
  const [analyzingVisual, setAnalyzingVisual] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [searching, setSearching] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [deletingDocumentId, setDeletingDocumentId] = useState('')
  const [error, setError] = useState('')

  const selectedDocument = documents.find((doc) => doc.id === selectedDocumentId) ?? null

  const stats = {
    totalDocuments: documents.length,
    pagesProcessed: documents.reduce((sum, doc) => sum + doc.page_count, 0),
    ocrPages: documents.reduce((sum, doc) => sum + doc.ocr_page_count, 0),
    indexedChunks: documents.reduce((sum, doc) => sum + doc.chunk_count, 0),
  }

  const refreshDocuments = useCallback(async () => {
    const data = await apiFetch<{ items: DocumentItem[] }>('/api/documents')
    setDocuments(data.items)
    setCompareDocumentIds((current) => {
      const existing = current.filter((id) => data.items.some((doc) => doc.id === id)).slice(0, 5)
      return existing.length ? existing : data.items.slice(0, 2).map((doc) => doc.id)
    })
    if (!selectedDocumentId && data.items.length) {
      setSelectedDocumentId(data.items[0].id)
    }
  }, [selectedDocumentId])

  useEffect(() => {
    const load = async () => {
      try {
        await refreshDocuments()
      } catch (err) {
        setError((err as Error).message)
      }
    }
    void load()
  }, [refreshDocuments])

  useEffect(() => {
    if (!selectedDocumentId) return
    const loadPage = async () => {
      try {
        const data = await apiFetch<PageRecord>(`/api/documents/${selectedDocumentId}/pages/${pageNumber}`)
        setPage(data)
        setVisualResult(null)
      } catch (err) {
        setError((err as Error).message)
      }
    }
    void loadPage()
  }, [selectedDocumentId, pageNumber])

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError('')
    const form = new FormData()
    form.append('file', file)

    try {
      const uploaded = await apiFetch<{ document_id: string }>('/api/documents/upload', {
        method: 'POST',
        body: form,
      })
      setSelectedDocumentId(uploaded.document_id)
      setPageNumber(1)
      await refreshDocuments()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  const handleQuery = async () => {
    if (asking) return
    if (!selectedDocument) {
      setError('Upload a document before asking a question.')
      return
    }
    setAsking(true)
    try {
      const response = await apiFetch<ChatResult>('/api/chat/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, document_ids: [selectedDocument.id], top_k: 5 }),
      })
      setChatResult(response)
      setError('')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setAsking(false)
    }
  }

  const handleSearch = async () => {
    if (searching) return
    setSearching(true)
    try {
      const selectedFilter = selectedDocumentId ? `&document_ids=${encodeURIComponent(selectedDocumentId)}` : ''
      const response = await apiFetch<{ results: any[] }>(
        '/api/search?q=' + encodeURIComponent(searchQuery) + selectedFilter,
      )
      setSearchResults(response.results)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSearching(false)
    }
  }

  const handleVisualAnalysis = async () => {
    if (!selectedDocument || !page) return
    setAnalyzingVisual(true)
    try {
      const response = await apiFetch<any>(
        `/api/documents/${selectedDocument.id}/pages/${page.page_number}/analyze-visual`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: visualQuestion }),
        },
      )
      setVisualResult(response)
      setError('')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setAnalyzingVisual(false)
    }
  }

  const openCitation = (documentId: string, citedPage: number) => {
    setSelectedDocumentId(documentId)
    setPageNumber(citedPage)
  }

  const handleCompare = async () => {
    if (comparing) return
    if (compareDocumentIds.length < 2) {
      setError('Upload at least two documents to compare.')
      return
    }
    setComparing(true)
    try {
      const response = await apiFetch<any>('/api/chat/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: compareQuestion, document_ids: compareDocumentIds, top_k: 5 }),
      })
      setCompareResult(response)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setComparing(false)
    }
  }

  const handleDelete = async (docId: string) => {
    if (deletingDocumentId) return
    setDeletingDocumentId(docId)
    try {
      await apiFetch(`/api/documents/${docId}`, { method: 'DELETE' })
      await refreshDocuments()
      if (selectedDocumentId === docId) {
        setSelectedDocumentId('')
        setPageNumber(1)
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setDeletingDocumentId('')
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Document intelligence workspace</p>
          <h1>DocuMind AI</h1>
        </div>
        <label className={`upload-button${uploading ? ' loading' : ''}`}>
          <input type="file" accept="application/pdf" onChange={handleUpload} disabled={uploading} />
          {uploading && <span className="button-spinner" aria-hidden="true" />}
          {uploading ? 'Uploading…' : 'Upload PDF'}
        </label>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="stats-grid">
        <article className="stat-card">
          <span>Total documents</span>
          <strong>{stats.totalDocuments}</strong>
        </article>
        <article className="stat-card">
          <span>Pages processed</span>
          <strong>{stats.pagesProcessed}</strong>
        </article>
        <article className="stat-card">
          <span>OCR processed</span>
          <strong>{stats.ocrPages}</strong>
        </article>
        <article className="stat-card">
          <span>Indexed chunks</span>
          <strong>{stats.indexedChunks}</strong>
        </article>
      </section>

      <main className="content-grid">
        <aside className="panel library-panel">
          <div className="panel-header">
            <h2>Document library</h2>
          </div>
          <ul className="document-list">
            {documents.length === 0 && <li className="empty-state">No documents uploaded yet.</li>}
            {documents.map((doc) => (
              <li key={doc.id} className={doc.id === selectedDocumentId ? 'selected' : ''}>
                <button type="button" className="doc-select" onClick={() => {
                  setSelectedDocumentId(doc.id)
                  setPageNumber(1)
                }}>
                  <div>
                    <strong>{doc.filename}</strong>
                    <small>{new Date(doc.created_at).toLocaleDateString()}</small>
                    <small>{doc.page_count} pages · {doc.extraction_method} · {doc.indexing_status} ({doc.chunk_count} chunks)</small>
                  </div>
                  <span>{doc.processing_status}</span>
                </button>
                <button type="button" className="ghost" onClick={() => handleDelete(doc.id)} disabled={Boolean(deletingDocumentId)}>
                  {deletingDocumentId === doc.id && <span className="button-spinner" aria-hidden="true" />}
                  {deletingDocumentId === doc.id ? 'Deleting…' : 'Delete'}
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <section className="panel viewer-panel">
          <div className="panel-header">
            <h2>Document viewer</h2>
            {selectedDocument && <span>{selectedDocument.filename}</span>}
          </div>
          {page ? (
            <>
              <div className="page-navigation">
                <button type="button" onClick={() => setPageNumber((current) => Math.max(1, current - 1))} disabled={pageNumber <= 1}>Previous page</button>
                <span>Page {page.page_number} of {selectedDocument?.page_count ?? page.page_number}</span>
                <button type="button" onClick={() => setPageNumber((current) => Math.min(selectedDocument?.page_count ?? current, current + 1))} disabled={pageNumber >= (selectedDocument?.page_count ?? pageNumber)}>Next page</button>
              </div>
              <div className="page-preview">
                <img
                  className="rendered-page"
                  src={`${API_BASE}/api/documents/${page.document_id}/pages/${page.page_number}/image`}
                  alt={`${selectedDocument?.filename ?? 'Document'} page ${page.page_number}`}
                />
                <div className="page-card">
                  <span>Page {page.page_number}</span>
                  <p>{page.extraction_method === 'ocr' ? 'OCR mode' : 'Native extraction'}</p>
                  {page.ocr_confidence !== null && <p>OCR confidence: {page.ocr_confidence.toFixed(1)}%</p>}
                </div>
              </div>
              <div className="page-text">
                <h3>Extracted text</h3>
                <p>{page.text || 'No text available for this page.'}</p>
                {page.extraction_error && <p className="status-note">Extraction note: {page.extraction_error}</p>}
              </div>
              <div className="visual-analysis">
                <h3>Visual page analysis (optional)</h3>
                <textarea value={visualQuestion} onChange={(event) => setVisualQuestion(event.target.value)} rows={2} />
                <button type="button" onClick={handleVisualAnalysis} disabled={analyzingVisual}>
                  {analyzingVisual ? 'Analyzing page…' : 'Analyze page image'}
                </button>
                {visualResult && (
                  <div className="answer-box">
                    <p>{visualResult.answer ?? visualResult.message}</p>
                    {visualResult.image_url && (
                      <small>{visualResult.status === 'complete' ? `${visualResult.provider} · ${visualResult.model}` : 'No visual model was run.'}</small>
                    )}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="empty-state">Select a document to preview its pages.</div>
          )}
        </section>
      </main>

      <section className="panel chat-panel">
        <div className="panel-header">
          <h2>AI chat</h2>
        </div>
        <div className="chat-controls">
          <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3} />
          <button type="button" onClick={handleQuery} disabled={asking}>
            {asking && <span className="button-spinner" aria-hidden="true" />}
            {asking ? 'Asking…' : 'Ask question'}
          </button>
        </div>
        {chatResult ? (
          <div className="answer-box">
            <h3>Answer</h3>
            <p className="status-note">
              {chatResult.generation_mode === 'llm'
                ? `Generated by ${chatResult.provider} · ${chatResult.model}`
                : chatResult.generation_mode === 'extractive_fallback'
                  ? 'Extractive fallback; no LLM answer was returned.'
                  : chatResult.insufficient_evidence
                    ? 'Insufficient evidence in the selected documents.'
                    : 'Extractive response from retrieved evidence.'}
            </p>
            {chatResult.provider_error && <p className="status-note">{chatResult.provider_error}</p>}
            <p>{chatResult.answer}</p>
            <div className="citation-list">
              {chatResult.citations.map((citation, index) => (
                <button type="button" key={`${citation.chunk_id}-${index}`} onClick={() => openCitation(citation.document_id, citation.page)}>
                  {citation.document} · p.{citation.page} · chunk {citation.chunk_id}
                </button>
              ))}
            </div>
            {chatResult.passages.length > 0 && (
              <details>
                <summary>Evidence passages</summary>
                {chatResult.passages.map((passage, index) => (
                  <div key={`${passage.document}-${passage.page}-${index}`} className="passage-item">
                    <strong>{passage.document}</strong>
                    <span>Page {passage.page}</span>
                    <p>{passage.content}</p>
                    <small>Dense similarity: {passage.dense_score.toFixed(3)} · BM25: {passage.keyword_score.toFixed(3)} · fused rank score: {passage.final_score.toFixed(4)}</small>
                    <button type="button" className="citation-open" onClick={() => openCitation(passage.document_id, passage.page)}>Open cited page</button>
                  </div>
                ))}
              </details>
            )}
          </div>
        ) : <div className="empty-state">Ask a question based on a selected document.</div>}
      </section>

      <section className="two-column-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>Search</h2>
          </div>
          <div className="search-controls">
            <input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder="Search documents..." />
            <button type="button" onClick={handleSearch} disabled={searching}>
              {searching && <span className="button-spinner" aria-hidden="true" />}
              {searching ? 'Searching…' : 'Search'}
            </button>
          </div>
          <ul className="result-list">
            {searchResults.length === 0 && <li className="empty-state">No search results yet.</li>}
            {searchResults.map((result, index) => (
              <li key={`${result.document_id}-${result.page_number}-${index}`}>
                <strong>{result.document_name}</strong>
                <span>Page {result.page_number}</span>
                <p>{result.content}</p>
                <small>Dense similarity: {result.dense_score.toFixed(3)} · BM25: {result.keyword_score.toFixed(3)} · fused rank score: {result.final_score.toFixed(4)}</small>
              </li>
            ))}
          </ul>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>Cross-document comparison</h2>
          </div>
          <fieldset className="comparison-selection">
            <legend>Select 2–5 documents</legend>
            {documents.map((doc) => (
              <label key={doc.id}>
                <input
                  type="checkbox"
                  checked={compareDocumentIds.includes(doc.id)}
                  disabled={!compareDocumentIds.includes(doc.id) && compareDocumentIds.length >= 5}
                  onChange={(event) => setCompareDocumentIds((current) => (
                    event.target.checked
                      ? [...current, doc.id].slice(0, 5)
                      : current.filter((id) => id !== doc.id)
                  ))}
                />
                {doc.filename}
              </label>
            ))}
          </fieldset>
          <textarea value={compareQuestion} onChange={(e) => setCompareQuestion(e.target.value)} rows={3} />
          <button type="button" onClick={handleCompare} disabled={comparing || compareDocumentIds.length < 2}>
            {comparing && <span className="button-spinner" aria-hidden="true" />}
            {comparing ? 'Comparing…' : 'Compare'}
          </button>
          {compareResult ? (
            <div className="answer-box">
              <p className="status-note">
                {compareResult.generation_mode === 'llm' ? 'LLM-synthesized from retrieved evidence.' : 'Extractive comparison; no LLM synthesis was used.'}
              </p>
              <p>{compareResult.summary}</p>
              {compareResult.common_points?.length > 0 && (
                <div className="comparison-item">
                  <strong>Common points supported across documents</strong>
                  {compareResult.common_points.map((point: any, index: number) => (
                    <p key={`common-${index}`}>
                      {point.text}{' '}
                      {point.citations.map((citation: any, citationIndex: number) => (
                        <button
                          type="button"
                          className="citation-open"
                          key={`${citation.chunk_id}-${citationIndex}`}
                          onClick={() => openCitation(citation.document_id, citation.page)}
                        >
                          {citation.filename} · p.{citation.page}
                        </button>
                      ))}
                    </p>
                  ))}
                </div>
              )}
              {compareResult.citations?.length > 0 && (
                <div className="citation-list">
                  {compareResult.citations.map((citation: any, index: number) => (
                    <button type="button" key={`${citation.chunk_id}-${index}`} onClick={() => openCitation(citation.document_id, citation.page)}>
                      {citation.filename} · p.{citation.page} · chunk {citation.chunk_id}
                    </button>
                  ))}
                </div>
              )}
              <strong>Evidence by document</strong>
              {compareResult.documents?.map((item: any) => (
                <div key={item.name} className="comparison-item">
                  <strong>{item.name}</strong>
                  {item.document_specific_points?.length > 0 && (
                    <div>
                      <small>Document-specific passages (not necessarily contradictions)</small>
                      {item.document_specific_points.map((point: any, index: number) => (
                        <p key={`${item.document_id}-specific-${index}`}>
                          {point.text}{' '}
                          <button type="button" className="citation-open" onClick={() => openCitation(point.citation.document_id, point.citation.page)}>
                            Source page {point.citation.page}
                          </button>
                        </p>
                      ))}
                    </div>
                  )}
                  {item.has_evidence
                    ? item.passages.map((passage: any, idx: number) => (
                      <p key={`${item.name}-${idx}`}>
                        {passage.content}{' '}
                        <button type="button" className="citation-open" onClick={() => openCitation(item.document_id, passage.page)}>
                          Open page {passage.page}
                        </button>
                      </p>
                    ))
                    : <p>No supporting passages were retrieved for this document.</p>}
                </div>
              ))}
            </div>
          ) : <div className="empty-state">Select more than one document to compare.</div>}
        </div>
      </section>
    </div>
  )
}

export default App
