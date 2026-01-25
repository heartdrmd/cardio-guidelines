#!/usr/bin/env python3
"""
Cardiology Guidelines AI Server

Flask API that:
1. Accepts PDF uploads
2. Extracts text and chunks it
3. Generates embeddings and stores in Postgres (pgvector)
4. Answers questions using Claude + relevant guideline context
"""

import os
import re
import hashlib
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
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
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    """Initialize database tables if they don't exist."""
    conn = get_db()
    cur = conn.cursor()
    
    # Create tables (pgvector extension should already be enabled)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS guidelines (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            year INT,
            source TEXT,
            filename TEXT,
            file_hash TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
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
    
    conn.commit()
    cur.close()
    conn.close()

# =============================================================================
# PDF PROCESSING
# =============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with page numbers."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    'page_number': i + 1,
                    'text': text
                })
    return pages

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
    """Generate embedding using Claude's voyage or OpenAI."""
    # For now, we'll use a simple approach - call Claude to get a semantic representation
    # In production, you'd use OpenAI embeddings or Voyage AI
    
    # Using OpenAI embeddings (most common for pgvector)
    import requests
    
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        # Return None if no OpenAI key - we'll do keyword search instead
        return None
    
    response = requests.post(
        'https://api.openai.com/v1/embeddings',
        headers={
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'text-embedding-3-small',
            'input': text[:8000]  # Limit input size
        }
    )
    
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    return None

def ingest_pdf(pdf_path, name, year=None, source=None):
    """Process PDF and store chunks in database."""
    conn = get_db()
    cur = conn.cursor()
    
    # Calculate file hash to prevent duplicates
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    # Check if already ingested
    cur.execute("SELECT id FROM guidelines WHERE file_hash = %s", (file_hash,))
    existing = cur.fetchone()
    if existing:
        cur.close()
        conn.close()
        return {'status': 'exists', 'guideline_id': existing[0]}
    
    # Extract text
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        cur.close()
        conn.close()
        return {'status': 'error', 'message': 'No text extracted from PDF'}
    
    # Insert guideline record
    cur.execute("""
        INSERT INTO guidelines (name, year, source, filename, file_hash)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """, (name, year, source, os.path.basename(pdf_path), file_hash))
    guideline_id = cur.fetchone()[0]
    
    # Process each page
    total_chunks = 0
    for page in pages:
        chunks = chunk_text(page['text'], page['page_number'])
        
        for chunk in chunks:
            # Get embedding
            embedding = get_embedding(chunk['content'])
            
            # Insert chunk
            if embedding:
                cur.execute("""
                    INSERT INTO chunks (guideline_id, page_number, content, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (guideline_id, chunk['page_number'], chunk['content'], embedding))
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
        'pages': len(pages),
        'chunks': total_chunks
    }

# =============================================================================
# SEARCH
# =============================================================================

def search_guidelines(query, limit=10):
    """Search for relevant guideline chunks."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Try vector search first
    embedding = get_embedding(query)
    
    if embedding:
        # Vector similarity search
        cur.execute("""
            SELECT c.id, c.content, c.page_number, c.section,
                   g.name as guideline_name, g.year,
                   1 - (c.embedding <=> %s::vector) as similarity
            FROM chunks c
            JOIN guidelines g ON c.guideline_id = g.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, (embedding, embedding, limit))
    else:
        # Fallback to keyword search
        search_terms = query.lower().split()
        search_pattern = '%' + '%'.join(search_terms) + '%'
        
        cur.execute("""
            SELECT c.id, c.content, c.page_number, c.section,
                   g.name as guideline_name, g.year,
                   0.5 as similarity
            FROM chunks c
            JOIN guidelines g ON c.guideline_id = g.id
            WHERE LOWER(c.content) LIKE %s
            LIMIT %s
        """, (search_pattern, limit))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return results

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
                'year': c.get('year'),
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
        'anthropic': bool(ANTHROPIC_API_KEY)
    })

@app.route('/api/guidelines', methods=['GET'])
def list_guidelines():
    """List all ingested guidelines."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT g.id, g.name, g.year, g.source, g.filename, g.created_at,
               COUNT(c.id) as chunk_count
        FROM guidelines g
        LEFT JOIN chunks c ON g.id = c.guideline_id
        GROUP BY g.id
        ORDER BY g.year DESC NULLS LAST, g.name
    """)
    
    guidelines = cur.fetchall()
    cur.close()
    conn.close()
    
    return jsonify({'guidelines': guidelines})

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Upload and ingest a PDF guideline."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400
    
    # Get metadata
    name = request.form.get('name', file.filename.replace('.pdf', ''))
    year = request.form.get('year')
    source = request.form.get('source', '')
    
    if year:
        try:
            year = int(year)
        except ValueError:
            year = None
    
    # Save file
    filename = f"{hashlib.md5(file.filename.encode()).hexdigest()[:8]}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Ingest
    try:
        result = ingest_pdf(filepath, name, year, source)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question about the guidelines."""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    
    # Search for relevant chunks
    chunks = search_guidelines(question, limit=8)
    
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
    """Delete a guideline and its chunks."""
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("DELETE FROM guidelines WHERE id = %s RETURNING name", (guideline_id,))
    deleted = cur.fetchone()
    
    conn.commit()
    cur.close()
    conn.close()
    
    if deleted:
        return jsonify({'status': 'deleted', 'name': deleted[0]})
    return jsonify({'error': 'Guideline not found'}), 404

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Run server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
