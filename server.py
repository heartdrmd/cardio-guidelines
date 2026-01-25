#!/usr/bin/env python3
"""
Cardiology Guidelines AI Server

Flask API that:
1. Accepts PDF uploads (single or multiple)
2. Extracts text and chunks it
3. Generates embeddings and stores in Postgres (pgvector)
4. Answers questions using Claude + relevant guideline context
5. Tracks all PDFs and allows deletion/replacement
"""

import os
import re
import hashlib
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import psycopg
from psycopg.rows import dict_row
import pdfplumber
import anthropic

# =============================================================================
# CONFIGURATION
# =============================================================================

app = Flask(__name__)
CORS(app)

# Database
DATABASE_URL = os.environ.get('DATABASE_URL')

# Anthropic
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Upload folder
UPLOAD_FOLDER = '/tmp/guidelines'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chunking config
CHUNK_SIZE = 1500  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks

# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db():
    """Get database connection."""
    conn = psycopg.connect(DATABASE_URL)
    return conn

def init_db():
    """Initialize database tables if they don't exist."""
    conn = get_db()
    cur = conn.cursor()
    
    # Guidelines table - tracks each uploaded PDF
    cur.execute("""
        CREATE TABLE IF NOT EXISTS guidelines (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            year INT,
            source TEXT,
            filename TEXT,
            file_hash TEXT UNIQUE,
            page_count INT,
            extracted_title TEXT,
            extracted_headers TEXT,
            file_size_bytes INT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Chunks table - stores extracted text with embeddings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            guideline_id INT REFERENCES guidelines(id) ON DELETE CASCADE,
            section TEXT,
            page_number INT,
            content TEXT NOT NULL,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Create index if not exists
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
        ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """)
    
    # Index for fast guideline lookups
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_guideline_idx ON chunks(guideline_id);
    """)
    
    conn.commit()
    cur.close()
    conn.close()

# =============================================================================
# PDF PROCESSING
# =============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with page numbers and headers."""
    pages = []
    all_headers = []
    title = None
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    'page_number': i + 1,
                    'text': text
                })
                
                # Try to extract title from first page
                if i == 0 and not title:
                    lines = text.strip().split('\n')
                    if lines:
                        # First non-empty line is often the title
                        title = lines[0][:200]  # Limit length
                
                # Extract potential headers (lines that look like headings)
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    # Heuristic: short lines in caps or with numbers might be headers
                    if line and len(line) < 100:
                        if line.isupper() or re.match(r'^\d+\.?\s+[A-Z]', line):
                            all_headers.append(line)
    
    return pages, title, all_headers[:50]  # Limit headers

def chunk_text(text, page_number, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        if chunk.strip():
            chunks.append({
                'content': chunk.strip(),
                'page_number': page_number
            })
        
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    
    return chunks

def get_embedding(text):
    """Generate embedding for a single text using OpenAI API."""
    import requests
    
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        return None
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/embeddings',
            headers={
                'Authorization': f'Bearer {openai_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'text-embedding-3-small',
                'input': text[:8000]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
    
    return None

def get_embeddings_batch(texts):
    """Generate embeddings for multiple texts in one API call (up to 2048)."""
    import requests
    
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        return [None] * len(texts)
    
    # Truncate each text to 8000 chars
    truncated = [t[:8000] for t in texts]
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/embeddings',
            headers={
                'Authorization': f'Bearer {openai_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'text-embedding-3-small',
                'input': truncated
            },
            timeout=120  # Longer timeout for batch
        )
        
        if response.status_code == 200:
            data = response.json()['data']
            # Sort by index to maintain order
            sorted_data = sorted(data, key=lambda x: x['index'])
            return [item['embedding'] for item in sorted_data]
        else:
            print(f"Batch embedding error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Batch embedding error: {e}")
    
    return [None] * len(texts)

def ingest_pdf(pdf_path, name, year=None, source=None):
    """Process PDF and store chunks in database."""
    conn = get_db()
    cur = conn.cursor()
    
    # Get file info
    file_size = os.path.getsize(pdf_path)
    
    # Calculate file hash to prevent duplicates
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    # Check if already ingested
    cur.execute("SELECT id, name FROM guidelines WHERE file_hash = %s", (file_hash,))
    existing = cur.fetchone()
    if existing:
        cur.close()
        conn.close()
        return {'status': 'exists', 'guideline_id': existing[0], 'name': existing[1]}
    
    # Extract text and metadata
    pages, extracted_title, headers = extract_text_from_pdf(pdf_path)
    if not pages:
        cur.close()
        conn.close()
        return {'status': 'error', 'message': 'No text extracted from PDF'}
    
    # Use extracted title if no name provided
    if not name or name == os.path.basename(pdf_path).replace('.pdf', ''):
        if extracted_title:
            name = extracted_title
    
    # Insert guideline record
    cur.execute("""
        INSERT INTO guidelines (name, year, source, filename, file_hash, page_count, 
                                extracted_title, extracted_headers, file_size_bytes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (name, year, source, os.path.basename(pdf_path), file_hash, 
          len(pages), extracted_title, '\n'.join(headers), file_size))
    guideline_id = cur.fetchone()[0]
    
    # Collect all chunks first
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page['text'], page['page_number'])
        all_chunks.extend(chunks)
    
    # Get embeddings in batches of 500
    BATCH_SIZE = 500
    all_embeddings = []
    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c['content'] for c in batch]
        embeddings = get_embeddings_batch(texts)
        all_embeddings.extend(embeddings)
    
    # Insert all chunks with embeddings
    total_chunks = 0
    for chunk, embedding in zip(all_chunks, all_embeddings):
        if embedding:
            cur.execute("""
                INSERT INTO chunks (guideline_id, page_number, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (guideline_id, chunk['page_number'], chunk['content'], str(embedding)))
        else:
            cur.execute("""
                INSERT INTO chunks (guideline_id, page_number, content)
                VALUES (%s, %s, %s)
            """, (guideline_id, chunk['page_number'], chunk['content']))
        
        total_chunks += 1
    
    conn.commit()
    cur.close()
    conn.close()
    
    return {
        'status': 'success',
        'guideline_id': guideline_id,
        'name': name,
        'pages': len(pages),
        'chunks': total_chunks,
        'extracted_title': extracted_title
    }

# =============================================================================
# SEARCH
# =============================================================================

def search_guidelines(query, limit=15):
    """Search for relevant guideline chunks using hybrid search."""
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    # Try vector search first
    embedding = get_embedding(query)
    
    results = []
    
    if embedding:
        # Vector similarity search - get more candidates
        cur.execute("""
            SELECT c.id, c.content, c.page_number, c.section,
                   g.id as guideline_id, g.name as guideline_name, g.year, g.source,
                   1 - (c.embedding <=> %s::vector) as similarity
            FROM chunks c
            JOIN guidelines g ON c.guideline_id = g.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, (str(embedding), str(embedding), limit))
        results = cur.fetchall()
        
        # Also do keyword search for important terms in the query
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        if keywords:
            keyword_pattern = '%' + '%'.join(keywords[:3]) + '%'
            cur.execute("""
                SELECT c.id, c.content, c.page_number, c.section,
                       g.id as guideline_id, g.name as guideline_name, g.year, g.source,
                       0.7 as similarity
                FROM chunks c
                JOIN guidelines g ON c.guideline_id = g.id
                WHERE LOWER(c.content) LIKE %s
                LIMIT %s
            """, (keyword_pattern, limit))
            keyword_results = cur.fetchall()
            
            # Merge results, avoiding duplicates
            seen_ids = {r['id'] for r in results}
            for r in keyword_results:
                if r['id'] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r['id'])
    else:
        # Fallback to keyword search only
        search_terms = query.lower().split()
        search_pattern = '%' + '%'.join(search_terms) + '%'
        
        cur.execute("""
            SELECT c.id, c.content, c.page_number, c.section,
                   g.id as guideline_id, g.name as guideline_name, g.year, g.source,
                   0.5 as similarity
            FROM chunks c
            JOIN guidelines g ON c.guideline_id = g.id
            WHERE LOWER(c.content) LIKE %s
            LIMIT %s
        """, (search_pattern, limit))
        results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return results[:limit]

# =============================================================================
# CLAUDE INTEGRATION
# =============================================================================

def ask_claude(question, context_chunks):
    """Ask Claude a question with guideline context."""
    if not claude:
        return {'error': 'Anthropic API key not configured'}
    
    # Build context from chunks
    context_parts = []
    for chunk in context_chunks:
        source = f"{chunk['guideline_name']}"
        if chunk.get('year'):
            source += f" ({chunk['year']})"
        if chunk.get('source'):
            source += f" - {chunk['source']}"
        if chunk.get('page_number'):
            source += f", p.{chunk['page_number']}"
        
        context_parts.append(f"[Source: {source}]\n{chunk['content']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Build prompt
    system_prompt = """You are a cardiology expert assistant. Answer questions based on the provided guideline excerpts.

Rules:
1. Base your answers on the provided guideline content
2. Cite specific guidelines when making recommendations (e.g., "Per the 2023 ACC/AHA AFib Guideline...")
3. Include recommendation classes when available (Class I, IIa, IIb, III)
4. Include level of evidence when available (A, B, C)
5. If the guidelines don't address the question, say so clearly
6. Be concise but thorough
7. If there's conflicting information between guidelines, note this"""

    user_prompt = f"""Based on the following guideline excerpts, please answer this question:

QUESTION: {question}

GUIDELINE EXCERPTS:
{context}

Please provide a clear, evidence-based answer with citations to the specific guidelines."""

    # Call Claude
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    
    return {
        'answer': response.content[0].text,
        'sources': [
            {
                'guideline': c['guideline_name'],
                'guideline_id': c['guideline_id'],
                'year': c.get('year'),
                'source': c.get('source'),
                'page': c.get('page_number'),
                'excerpt': c['content'][:200] + '...' if len(c['content']) > 200 else c['content']
            }
            for c in context_chunks
        ]
    }

# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory('.', 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': bool(DATABASE_URL),
        'anthropic': bool(ANTHROPIC_API_KEY),
        'openai': bool(os.environ.get('OPENAI_API_KEY'))
    })

@app.route('/api/guidelines', methods=['GET'])
def list_guidelines():
    """List all ingested guidelines with full metadata."""
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    cur.execute("""
        SELECT g.id, g.name, g.year, g.source, g.filename, g.file_hash,
               g.page_count, g.extracted_title, g.extracted_headers,
               g.file_size_bytes, g.created_at, g.updated_at,
               COUNT(c.id) as chunk_count
        FROM guidelines g
        LEFT JOIN chunks c ON g.id = c.guideline_id
        GROUP BY g.id
        ORDER BY g.created_at DESC
    """)
    
    guidelines = cur.fetchall()
    cur.close()
    conn.close()
    
    # Format for JSON
    for g in guidelines:
        if g['created_at']:
            g['created_at'] = g['created_at'].isoformat()
        if g['updated_at']:
            g['updated_at'] = g['updated_at'].isoformat()
        if g['file_size_bytes']:
            g['file_size_mb'] = round(g['file_size_bytes'] / (1024 * 1024), 2)
    
    return jsonify({'guidelines': guidelines})

@app.route('/api/guidelines/<int:guideline_id>', methods=['GET'])
def get_guideline(guideline_id):
    """Get detailed info about a specific guideline."""
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    cur.execute("""
        SELECT g.*, COUNT(c.id) as chunk_count
        FROM guidelines g
        LEFT JOIN chunks c ON g.id = c.guideline_id
        WHERE g.id = %s
        GROUP BY g.id
    """, (guideline_id,))
    
    guideline = cur.fetchone()
    cur.close()
    conn.close()
    
    if not guideline:
        return jsonify({'error': 'Guideline not found'}), 404
    
    if guideline['created_at']:
        guideline['created_at'] = guideline['created_at'].isoformat()
    if guideline['updated_at']:
        guideline['updated_at'] = guideline['updated_at'].isoformat()
    
    return jsonify(guideline)

@app.route('/api/upload', methods=['POST'])
def upload_pdfs():
    """Upload and ingest one or more PDF guidelines."""
    if 'files' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    # Handle both single file and multiple files
    files = request.files.getlist('files') or [request.files.get('file')]
    files = [f for f in files if f and f.filename]
    
    if not files:
        return jsonify({'error': 'No valid files provided'}), 400
    
    results = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            results.append({
                'filename': file.filename,
                'status': 'error',
                'message': 'File must be a PDF'
            })
            continue
        
        # Get metadata (use form data or defaults)
        # For multiple files, we auto-detect names
        name = request.form.get('name', '') if len(files) == 1 else ''
        if not name:
            name = file.filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        
        year = request.form.get('year')
        source = request.form.get('source', '')
        
        # Try to extract year from filename
        if not year:
            year_match = re.search(r'20\d{2}', file.filename)
            if year_match:
                year = int(year_match.group())
        
        if year:
            try:
                year = int(year)
            except ValueError:
                year = None
        
        # Save file temporarily
        filename = f"{hashlib.md5(file.filename.encode()).hexdigest()[:8]}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            file.save(filepath)
            result = ingest_pdf(filepath, name, year, source)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'message': str(e)
            })
        finally:
            # Clean up temp file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    # Summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    existing = sum(1 for r in results if r.get('status') == 'exists')
    failed = sum(1 for r in results if r.get('status') == 'error')
    
    return jsonify({
        'results': results,
        'summary': {
            'total': len(results),
            'successful': successful,
            'existing': existing,
            'failed': failed
        }
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question about the guidelines."""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    
    # Search for relevant chunks
    chunks = search_guidelines(question, limit=15)
    
    # DEBUG: Log what we found
    print(f"=== SEARCH DEBUG ===")
    print(f"Question: {question}")
    print(f"Chunks found: {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"Chunk {i+1} (page {c.get('page_number')}): {c['content'][:100]}...")
    print(f"====================")
    
    if not chunks:
        return jsonify({
            'answer': "I couldn't find any relevant information in the guidelines database. Please make sure guidelines have been uploaded and try rephrasing your question.",
            'sources': []
        })
    
    # Get answer from Claude
    result = ask_claude(question, chunks)
    
    return jsonify(result)

@app.route('/api/delete/<int:guideline_id>', methods=['DELETE'])
def delete_guideline(guideline_id):
    """Delete a guideline and all its chunks."""
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    # Get guideline info first
    cur.execute("SELECT name, filename FROM guidelines WHERE id = %s", (guideline_id,))
    guideline = cur.fetchone()
    
    if not guideline:
        cur.close()
        conn.close()
        return jsonify({'error': 'Guideline not found'}), 404
    
    # Count chunks that will be deleted
    cur.execute("SELECT COUNT(*) as count FROM chunks WHERE guideline_id = %s", (guideline_id,))
    chunk_count = cur.fetchone()['count']
    
    # Delete (chunks will cascade)
    cur.execute("DELETE FROM guidelines WHERE id = %s", (guideline_id,))
    
    conn.commit()
    cur.close()
    conn.close()
    
    return jsonify({
        'status': 'deleted',
        'name': guideline['name'],
        'filename': guideline['filename'],
        'chunks_deleted': chunk_count
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics."""
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    cur.execute("""
        SELECT 
            COUNT(DISTINCT g.id) as total_guidelines,
            COUNT(c.id) as total_chunks,
            SUM(g.page_count) as total_pages,
            SUM(g.file_size_bytes) as total_size_bytes
        FROM guidelines g
        LEFT JOIN chunks c ON g.id = c.guideline_id
    """)
    
    stats = cur.fetchone()
    cur.close()
    conn.close()
    
    if stats['total_size_bytes']:
        stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
    
    return jsonify(stats)

# =============================================================================
# MAIN
# =============================================================================

# Initialize database on startup
try:
    init_db()
    print("Database initialized successfully")
except Exception as e:
    print(f"Database initialization error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
