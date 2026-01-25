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
# PRICING (per 1M tokens)
# =============================================================================

MODEL_PRICING = {
    # Claude models (input, output)
    'claude-opus-4-20250514': {'input': 15.00, 'output': 75.00},
    'claude-sonnet-4-20250514': {'input': 3.00, 'output': 15.00},
    'claude-haiku-3-5-20241022': {'input': 0.80, 'output': 4.00},
    # OpenAI models
    'gpt-5.2': {'input': 1.75, 'output': 14.00},
}

def calculate_cost(model, input_tokens, output_tokens):
    """Calculate cost in dollars for API call."""
    pricing = MODEL_PRICING.get(model.replace('-thinking', ''), {'input': 3.00, 'output': 15.00})
    input_cost = (input_tokens / 1_000_000) * pricing['input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']
    return round(input_cost + output_cost, 6)

# =============================================================================
# CLAUDE INTEGRATION
# =============================================================================

def ask_claude(question, context_chunks, model="claude-sonnet-4-20250514"):
    """Ask Claude or GPT a question with guideline context."""
    
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

    # Check if using OpenAI
    if model.startswith('gpt-'):
        return ask_openai(user_prompt, system_prompt, model, context_chunks)
    
    # Check if using extended thinking
    if model.endswith('-thinking'):
        actual_model = model.replace('-thinking', '')
        return ask_claude_thinking(user_prompt, system_prompt, actual_model, context_chunks)
    
    # Standard Claude call
    if not claude:
        return {'error': 'Anthropic API key not configured'}
    
    response = claude.messages.create(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    
    # Calculate cost
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    
    return {
        'answer': response.content[0].text,
        'model': model,
        'usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_usd': cost
        },
        'sources': [
            {
                'guideline': c['guideline_name'],
                'guideline_id': c['guideline_id'],
                'year': c.get('year'),
                'source': c.get('source'),
                'page': c.get('page_number'),
                'excerpt': c['content'][:200] + '...' if len(c['content']) > 200 else c['content'],
                'full_text': c['content']
            }
            for c in context_chunks
        ]
    }

def ask_claude_thinking(user_prompt, system_prompt, model, context_chunks):
    """Ask Claude with extended thinking enabled."""
    if not claude:
        return {'error': 'Anthropic API key not configured'}
    
    try:
        # Extended thinking requires combining system + user prompt
        # because system parameter is not supported with thinking
        combined_prompt = f"""{system_prompt}

{user_prompt}"""
        
        response = claude.messages.create(
            model=model,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            messages=[{"role": "user", "content": combined_prompt}]
        )
        
        # Extract the text response (skip thinking blocks)
        answer_text = ""
        for block in response.content:
            if block.type == "text":
                answer_text = block.text
                break
        
        if not answer_text:
            answer_text = "No response generated"
        
        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(model, input_tokens, output_tokens)
        
        return {
            'answer': answer_text,
            'model': f"{model} (thinking)",
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cost_usd': cost
            },
            'sources': [
                {
                    'guideline': c['guideline_name'],
                    'guideline_id': c['guideline_id'],
                    'year': c.get('year'),
                    'source': c.get('source'),
                    'page': c.get('page_number'),
                    'excerpt': c['content'][:200] + '...' if len(c['content']) > 200 else c['content'],
                    'full_text': c['content']
                }
                for c in context_chunks
            ]
        }
    except Exception as e:
        import traceback
        print(f"Thinking mode error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {'error': f'Extended thinking failed: {str(e)}'}

def ask_openai(user_prompt, system_prompt, model, context_chunks):
    """Ask OpenAI GPT a question."""
    import requests
    
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        return {'error': 'OpenAI API key not configured'}
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {openai_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                'max_tokens': 2000
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            cost = calculate_cost(model, input_tokens, output_tokens)
            
            return {
                'answer': data['choices'][0]['message']['content'],
                'model': model,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost_usd': cost
                },
                'sources': [
                    {
                        'guideline': c['guideline_name'],
                        'guideline_id': c['guideline_id'],
                        'year': c.get('year'),
                        'source': c.get('source'),
                        'page': c.get('page_number'),
                        'excerpt': c['content'][:200] + '...' if len(c['content']) > 200 else c['content'],
                        'full_text': c['content']
                    }
                    for c in context_chunks
                ]
            }
        else:
            return {'error': f'OpenAI API error: {response.status_code} - {response.text}'}
    except Exception as e:
        return {'error': f'OpenAI API error: {str(e)}'}

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
    model = data.get('model', 'claude-sonnet-4-20250514')
    
    # Search for relevant chunks
    chunks = search_guidelines(question, limit=15)
    
    # DEBUG: Log what we found
    print(f"=== SEARCH DEBUG ===")
    print(f"Question: {question}")
    print(f"Model: {model}")
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
    result = ask_claude(question, chunks, model)
    
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
# PPTX GENERATION
# =============================================================================

@app.route('/api/generate-pptx/<int:guideline_id>', methods=['POST'])
def generate_pptx(guideline_id):
    """Generate a summary PPTX for a guideline."""
    data = request.get_json() or {}
    topic = data.get('topic')  # Optional - if None, generate full summary
    
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    # Get guideline info
    cur.execute("SELECT * FROM guidelines WHERE id = %s", (guideline_id,))
    guideline = cur.fetchone()
    
    if not guideline:
        cur.close()
        conn.close()
        return jsonify({'error': 'Guideline not found'}), 404
    
    # Get relevant chunks
    if topic:
        # Topic-specific - search for chunks related to topic
        cur.execute("""
            SELECT content, page_number FROM chunks 
            WHERE guideline_id = %s AND LOWER(content) LIKE %s
            ORDER BY page_number
            LIMIT 30
        """, (guideline_id, f'%{topic.lower()}%'))
    else:
        # Full summary - get chunks spread across the document
        cur.execute("""
            SELECT content, page_number FROM chunks 
            WHERE guideline_id = %s
            ORDER BY page_number
        """, (guideline_id,))
    
    chunks = cur.fetchall()
    cur.close()
    conn.close()
    
    if not chunks:
        return jsonify({'error': 'No content found for this guideline'}), 404
    
    # Build context for Claude
    context = "\n\n---\n\n".join([f"[Page {c['page_number']}]\n{c['content']}" for c in chunks[:25]])
    
    # Generate summary with Claude
    if not claude:
        return jsonify({'error': 'Anthropic API key not configured'}), 500
    
    topic_text = f" focusing on {topic}" if topic else ""
    
    summary_prompt = f"""Based on the following guideline content{topic_text}, create a structured summary for a PowerPoint presentation.

GUIDELINE: {guideline['name']} ({guideline.get('year', 'N/A')})

CONTENT:
{context}

Please provide a summary in this exact JSON format:
{{
    "title": "Presentation title",
    "subtitle": "Brief subtitle",
    "slides": [
        {{
            "title": "Slide title",
            "points": ["Point 1", "Point 2", "Point 3"],
            "class": "I/IIa/IIb/III if applicable, or null",
            "loe": "A/B/C if applicable, or null"
        }}
    ]
}}

Create 5-8 slides covering the key recommendations. For each recommendation, include the Class (I, IIa, IIb, III) and Level of Evidence (A, B-R, B-NR, C) when available.
Keep points concise (under 15 words each).
Return ONLY valid JSON, no other text."""

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.1,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        summary_text = response.content[0].text
        
        # Parse JSON from response
        import json
        # Clean up response - find JSON
        start = summary_text.find('{')
        end = summary_text.rfind('}') + 1
        if start >= 0 and end > start:
            summary_json = json.loads(summary_text[start:end])
        else:
            return jsonify({'error': 'Failed to parse summary'}), 500
        
        # Generate PPTX
        pptx_path = create_pptx(summary_json, guideline, topic)
        
        if pptx_path:
            # Return the file
            from flask import send_file
            return send_file(
                pptx_path,
                as_attachment=True,
                download_name=f"{guideline['name'].replace(' ', '_')}{'_' + topic if topic else ''}_Summary.pptx",
                mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation'
            )
        else:
            return jsonify({'error': 'Failed to create PPTX'}), 500
            
    except Exception as e:
        print(f"PPTX generation error: {e}")
        return jsonify({'error': str(e)}), 500

def create_pptx(summary, guideline, topic=None):
    """Create a PowerPoint file from summary data."""
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RgbColor
        from pptx.enum.text import PP_ALIGN
        
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        
        # Color scheme
        DARK_BLUE = RgbColor(30, 58, 95)
        ACCENT_BLUE = RgbColor(59, 130, 246)
        GREEN = RgbColor(16, 185, 129)
        YELLOW = RgbColor(245, 158, 11)
        RED = RgbColor(239, 68, 68)
        
        # Title slide
        title_slide_layout = prs.slide_layouts[6]  # Blank
        slide = prs.slides.add_slide(title_slide_layout)
        
        # Title
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.333), Inches(1.5))
        title_frame = title_box.text_frame
        title_para = title_frame.paragraphs[0]
        title_para.text = summary.get('title', guideline['name'])
        title_para.font.size = Pt(44)
        title_para.font.bold = True
        title_para.alignment = PP_ALIGN.CENTER
        
        # Subtitle
        subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.333), Inches(1))
        subtitle_frame = subtitle_box.text_frame
        subtitle_para = subtitle_frame.paragraphs[0]
        subtitle_para.text = summary.get('subtitle', f"Summary {'- ' + topic if topic else ''}")
        subtitle_para.font.size = Pt(24)
        subtitle_para.alignment = PP_ALIGN.CENTER
        
        # Source info
        source_box = slide.shapes.add_textbox(Inches(0.5), Inches(6), Inches(12.333), Inches(0.5))
        source_frame = source_box.text_frame
        source_para = source_frame.paragraphs[0]
        source_para.text = f"Source: {guideline['name']} ({guideline.get('year', 'N/A')})"
        source_para.font.size = Pt(14)
        source_para.alignment = PP_ALIGN.CENTER
        
        # Content slides
        for slide_data in summary.get('slides', []):
            content_slide = prs.slides.add_slide(prs.slide_layouts[6])
            
            # Slide title
            title_box = content_slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.333), Inches(1))
            title_frame = title_box.text_frame
            title_para = title_frame.paragraphs[0]
            title_para.text = slide_data.get('title', 'Recommendations')
            title_para.font.size = Pt(32)
            title_para.font.bold = True
            
            # Class badge if present
            rec_class = slide_data.get('class')
            if rec_class:
                badge_left = Inches(11)
                badge_box = content_slide.shapes.add_textbox(badge_left, Inches(0.5), Inches(1.5), Inches(0.5))
                badge_frame = badge_box.text_frame
                badge_para = badge_frame.paragraphs[0]
                badge_para.text = f"Class {rec_class}"
                badge_para.font.size = Pt(14)
                badge_para.font.bold = True
                badge_para.alignment = PP_ALIGN.CENTER
            
            # Content points
            content_top = Inches(1.5)
            for i, point in enumerate(slide_data.get('points', [])):
                point_box = content_slide.shapes.add_textbox(Inches(0.7), content_top + Inches(i * 0.8), Inches(11.5), Inches(0.7))
                point_frame = point_box.text_frame
                point_para = point_frame.paragraphs[0]
                point_para.text = f"• {point}"
                point_para.font.size = Pt(20)
            
            # Level of evidence
            loe = slide_data.get('loe')
            if loe:
                loe_box = content_slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(3), Inches(0.4))
                loe_frame = loe_box.text_frame
                loe_para = loe_frame.paragraphs[0]
                loe_para.text = f"Level of Evidence: {loe}"
                loe_para.font.size = Pt(12)
        
        # Save
        output_path = f'/tmp/pptx_{guideline["id"]}_{topic or "full"}.pptx'
        prs.save(output_path)
        return output_path
        
    except ImportError:
        print("python-pptx not installed")
        return None
    except Exception as e:
        print(f"PPTX creation error: {e}")
        return None

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
