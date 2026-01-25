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

# Test pptx import at startup
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.util import Emu
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor  # Note: RGBColor not RgbColor
    PPTX_AVAILABLE = True
    print("✓ python-pptx loaded successfully")
except ImportError as e:
    PPTX_AVAILABLE = False
    print(f"✗ python-pptx import failed: {e}")

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
            pdf_data BYTEA,
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
            is_reference BOOLEAN DEFAULT FALSE,
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
    
    # Migration: Add new columns if they don't exist
    # Add pdf_data column to guidelines
    try:
        cur.execute("ALTER TABLE guidelines ADD COLUMN IF NOT EXISTS pdf_data BYTEA")
    except:
        pass  # Column may already exist
    
    # Add is_reference column to chunks
    try:
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_reference BOOLEAN DEFAULT FALSE")
    except:
        pass  # Column may already exist
    
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
    
    # Read PDF binary data
    with open(pdf_path, 'rb') as f:
        pdf_binary = f.read()
    
    # Calculate file hash to prevent duplicates
    file_hash = hashlib.md5(pdf_binary).hexdigest()
    
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
    
    # Insert guideline record WITH PDF data
    cur.execute("""
        INSERT INTO guidelines (name, year, source, filename, file_hash, page_count, 
                                extracted_title, extracted_headers, file_size_bytes, pdf_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (name, year, source, os.path.basename(pdf_path), file_hash, 
          len(pages), extracted_title, '\n'.join(headers), file_size, pdf_binary))
    guideline_id = cur.fetchone()[0]
    
    # Detect reference section start page
    reference_start_page = None
    for page in pages:
        text_lower = page['text'].lower()
        # Common reference section headers
        if any(marker in text_lower for marker in [
            '\nreferences\n', '\nreference\n', '\nbibliography\n',
            'references\n1.', 'references\n 1.', '\n\nreferences',
            'author contributions', 'conflict of interest', 'acknowledgments'
        ]):
            reference_start_page = page['page_number']
            print(f"Detected reference section starting at page {reference_start_page}")
            break
    
    # Collect all chunks first
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page['text'], page['page_number'])
        for chunk in chunks:
            # Mark as reference if at or after reference section
            is_ref = reference_start_page and page['page_number'] >= reference_start_page
            chunk['is_reference'] = is_ref
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
    ref_chunks = 0
    for chunk, embedding in zip(all_chunks, all_embeddings):
        is_ref = chunk.get('is_reference', False)
        if is_ref:
            ref_chunks += 1
            
        if embedding:
            cur.execute("""
                INSERT INTO chunks (guideline_id, page_number, content, embedding, is_reference)
                VALUES (%s, %s, %s, %s, %s)
            """, (guideline_id, chunk['page_number'], chunk['content'], str(embedding), is_ref))
        else:
            cur.execute("""
                INSERT INTO chunks (guideline_id, page_number, content, is_reference)
                VALUES (%s, %s, %s, %s)
            """, (guideline_id, chunk['page_number'], chunk['content'], is_ref))
        
        total_chunks += 1
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"Ingested {total_chunks} chunks ({ref_chunks} marked as references)")
    
    return {
        'status': 'success',
        'guideline_id': guideline_id,
        'name': name,
        'pages': len(pages),
        'chunks': total_chunks,
        'reference_chunks': ref_chunks,
        'extracted_title': extracted_title
    }

# =============================================================================
# SEARCH
# =============================================================================

# Medical abbreviation expansions
MEDICAL_EXPANSIONS = {
    'afib': ['atrial fibrillation', 'af', 'afib'],
    'a fib': ['atrial fibrillation', 'af', 'afib'],
    'ac': ['anticoagulation', 'anticoagulant'],
    'oral ac': ['oral anticoagulation', 'anticoagulant', 'doac', 'noac', 'warfarin'],
    'doac': ['direct oral anticoagulant', 'doac', 'noac', 'apixaban', 'rivaroxaban', 'dabigatran', 'edoxaban'],
    'noac': ['novel oral anticoagulant', 'doac', 'noac'],
    'asa': ['aspirin', 'acetylsalicylic acid'],
    'htn': ['hypertension', 'blood pressure'],
    'dm': ['diabetes', 'diabetic'],
    'cad': ['coronary artery disease', 'coronary'],
    'hf': ['heart failure'],
    'chf': ['congestive heart failure', 'heart failure'],
    'mi': ['myocardial infarction', 'heart attack'],
    'cv': ['cardiovascular'],
    'tia': ['transient ischemic attack'],
    'cva': ['cerebrovascular accident', 'stroke'],
    'lvef': ['left ventricular ejection fraction', 'ejection fraction'],
    'ef': ['ejection fraction'],
    'ablation': ['catheter ablation', 'ablation', 'pulmonary vein isolation', 'pvi'],
    'cardioversion': ['cardioversion', 'cardiovert'],
    'rate control': ['rate control', 'heart rate'],
    'rhythm control': ['rhythm control', 'sinus rhythm'],
}

def understand_query(query):
    """Use Claude to understand the clinical question and extract search terms."""
    if not claude:
        return None
    
    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",  # Sonnet for better understanding
            max_tokens=300,
            temperature=0,
            messages=[{
                "role": "user", 
                "content": f"""Extract the key medical concepts from this cardiology question. Return ONLY a comma-separated list of search terms.

Question: "{query}"

Think about:
- What condition is being discussed? (expand abbreviations: afib=atrial fibrillation, HTN=hypertension, etc.)
- What treatment/intervention is being asked about?
- What clinical decision needs to be made?
- What guideline topics would be relevant?

Return ONLY the search terms, nothing else. Example: atrial fibrillation, anticoagulation, stroke risk, CHA2DS2-VASc"""
            }]
        )
        
        terms = response.content[0].text.strip()
        print(f"AI understood query as: {terms}")
        return [t.strip().lower() for t in terms.split(',') if t.strip()]
    except Exception as e:
        print(f"Query understanding failed: {e}")
        return None

def expand_query_terms(query):
    """Expand medical abbreviations in query to improve search."""
    query_lower = query.lower()
    expanded_terms = set()
    
    # Check for abbreviations and expand them
    for abbrev, expansions in MEDICAL_EXPANSIONS.items():
        if abbrev in query_lower:
            expanded_terms.update(expansions)
    
    # Also add original significant words (4+ chars)
    for word in query.split():
        word_clean = word.lower().strip('?.,/')
        if len(word_clean) >= 4:
            expanded_terms.add(word_clean)
    
    return list(expanded_terms)

def search_guidelines(query, limit=50):
    """Search for relevant guideline chunks using AI-powered hybrid search."""
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    # Step 1: Use AI to understand the query
    ai_terms = understand_query(query)
    
    # Step 2: Also expand with our abbreviation dictionary
    expanded_terms = expand_query_terms(query)
    
    # Combine AI terms with expanded terms
    all_search_terms = set(expanded_terms)
    if ai_terms:
        all_search_terms.update(ai_terms)
    
    print(f"Final search terms: {list(all_search_terms)}")
    
    # Step 3: Vector search with original query
    embedding = get_embedding(query)
    
    print(f"=== EMBEDDING DEBUG ===")
    print(f"Query: {query}")
    print(f"Embedding received: {embedding is not None}")
    if embedding:
        print(f"Embedding length: {len(embedding)}")
    
    results = []
    seen_ids = set()
    
    if embedding:
        # Vector similarity search - EXCLUDE reference sections
        # Format embedding as string for pgvector
        emb_str = '[' + ','.join(map(str, embedding)) + ']'
        cur.execute("""
            SELECT c.id, c.content, c.page_number, c.section,
                   g.id as guideline_id, g.name as guideline_name, g.year, g.source,
                   1 - (c.embedding <=> %s::vector) as similarity
            FROM chunks c
            JOIN guidelines g ON c.guideline_id = g.id
            WHERE c.embedding IS NOT NULL
            AND (c.is_reference IS NULL OR c.is_reference = FALSE)
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, (emb_str, emb_str, limit))
        results = cur.fetchall()
        print(f"Vector search returned: {len(results)} results")
        seen_ids = {r['id'] for r in results}
    else:
        print("No embedding - skipping vector search")
    
    # Step 4: Keyword search for each term (AI + expanded) - EXCLUDE references
    for term in list(all_search_terms)[:10]:  # Limit to top 10 terms
        if len(term) >= 3:
            cur.execute("""
                SELECT c.id, c.content, c.page_number, c.section,
                       g.id as guideline_id, g.name as guideline_name, g.year, g.source,
                       0.7 as similarity
                FROM chunks c
                JOIN guidelines g ON c.guideline_id = g.id
                WHERE LOWER(c.content) LIKE %s
                AND (c.is_reference IS NULL OR c.is_reference = FALSE)
                LIMIT %s
            """, (f'%{term}%', limit // 2))
            keyword_results = cur.fetchall()
            
            for r in keyword_results:
                if r['id'] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r['id'])
    
    cur.close()
    conn.close()
    
    # Sort by similarity and return top results
    results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
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
        # GPT-5.2 uses max_completion_tokens instead of max_tokens
        request_body = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'temperature': 0.1
        }
        
        # Use correct token parameter based on model
        if model.startswith('gpt-5'):
            request_body['max_completion_tokens'] = 2000
        else:
            request_body['max_tokens'] = 2000
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {openai_key}',
                'Content-Type': 'application/json'
            },
            json=request_body,
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
    
    # Search for relevant chunks - get more for better coverage
    chunks = search_guidelines(question, limit=50)
    
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
# TOPIC EXTRACTION
# =============================================================================

@app.route('/api/topics/<int:guideline_id>')
def get_guideline_topics(guideline_id):
    """Extract main topics from a guideline using AI analysis of headers and content."""
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    # Get guideline info including extracted headers
    cur.execute("SELECT id, name, extracted_headers FROM guidelines WHERE id = %s", (guideline_id,))
    guideline = cur.fetchone()
    
    if not guideline:
        cur.close()
        conn.close()
        return jsonify({'error': 'Guideline not found'}), 404
    
    # Get sample content from different parts of the document (excluding references)
    cur.execute("""
        SELECT DISTINCT content, page_number 
        FROM chunks 
        WHERE guideline_id = %s 
        AND (is_reference IS NULL OR is_reference = FALSE)
        ORDER BY page_number
        LIMIT 50
    """, (guideline_id,))
    chunks = cur.fetchall()
    cur.close()
    conn.close()
    
    # Build content sample for AI
    headers = guideline.get('extracted_headers', '') or ''
    content_sample = '\n'.join([c['content'][:500] for c in chunks[:30]])
    
    # Use Claude to identify main topics
    if not claude:
        # Fallback to headers-based extraction
        return jsonify({'topics': extract_topics_from_headers(headers)})
    
    try:
        prompt = f"""Analyze this medical guideline and identify the 5-10 main clinical TOPICS covered.

GUIDELINE: {guideline['name']}

EXTRACTED HEADERS:
{headers[:2000]}

SAMPLE CONTENT:
{content_sample[:4000]}

Return ONLY a JSON array of topic objects. Each topic should be a distinct clinical area.
DO NOT include: References, Authors, Disclosures, Appendix, Introduction, Methods, Acknowledgments

Format:
[
    {{"topic": "Topic Name", "icon": "emoji"}},
    {{"topic": "Another Topic", "icon": "emoji"}}
]

Use relevant medical emojis like: 💊 💉 🫀 ❤️ 🧠 ⚡ 🔥 🩺 📊 🎯 ⚠️ 🛡️ 💓 🩸

Return ONLY the JSON array, no other text."""

        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text.strip()
        
        # Parse JSON
        import json
        start = result_text.find('[')
        end = result_text.rfind(']') + 1
        if start >= 0 and end > start:
            topics = json.loads(result_text[start:end])
            return jsonify({'topics': topics, 'guideline': guideline['name']})
        else:
            return jsonify({'topics': extract_topics_from_headers(headers)})
            
    except Exception as e:
        print(f"Topic extraction error: {e}")
        return jsonify({'topics': extract_topics_from_headers(headers)})

def extract_topics_from_headers(headers):
    """Fallback: extract topics from headers if AI fails."""
    if not headers:
        return []
    
    # Common clinical topic patterns
    topic_keywords = [
        'anticoagulation', 'ablation', 'rate control', 'rhythm control',
        'stroke', 'heart failure', 'prevention', 'diagnosis', 'treatment',
        'management', 'therapy', 'screening', 'risk', 'assessment'
    ]
    
    topics = []
    seen = set()
    
    for line in headers.split('\n'):
        line_lower = line.lower().strip()
        if len(line_lower) < 5 or len(line_lower) > 60:
            continue
        if any(skip in line_lower for skip in ['reference', 'author', 'disclosure', 'appendix', 'acknowledge']):
            continue
        if any(kw in line_lower for kw in topic_keywords):
            topic_name = line.strip()
            if topic_name.lower() not in seen:
                seen.add(topic_name.lower())
                topics.append({'topic': topic_name, 'icon': '📋'})
    
    return topics[:10]

# =============================================================================
# PDF PAGE IMAGE EXTRACTION
# =============================================================================

@app.route('/api/page-image/<int:guideline_id>/<int:page_number>')
def get_page_image(guideline_id, page_number):
    """Extract and return a specific page from a stored PDF as an image."""
    conn = get_db()
    cur = conn.cursor(row_factory=dict_row)
    
    # Get PDF data
    cur.execute("SELECT pdf_data, page_count, name FROM guidelines WHERE id = %s", (guideline_id,))
    guideline = cur.fetchone()
    cur.close()
    conn.close()
    
    if not guideline:
        return jsonify({'error': 'Guideline not found'}), 404
    
    if not guideline['pdf_data']:
        return jsonify({'error': 'PDF not stored for this guideline'}), 404
    
    if page_number < 1 or page_number > guideline['page_count']:
        return jsonify({'error': f'Invalid page number. PDF has {guideline["page_count"]} pages'}), 400
    
    try:
        # Try to use pdf2image (requires poppler)
        try:
            from pdf2image import convert_from_bytes
            import io
            
            # Convert just the requested page (page_number is 1-indexed)
            images = convert_from_bytes(
                bytes(guideline['pdf_data']),
                first_page=page_number,
                last_page=page_number,
                dpi=150  # Good balance of quality and size
            )
            
            if images:
                # Convert to PNG
                img_buffer = io.BytesIO()
                images[0].save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                from flask import send_file
                return send_file(
                    img_buffer,
                    mimetype='image/png',
                    download_name=f'{guideline["name"]}_page_{page_number}.png'
                )
        except ImportError:
            # pdf2image not available, try pymupdf
            try:
                import fitz  # PyMuPDF
                import io
                
                pdf_doc = fitz.open(stream=bytes(guideline['pdf_data']), filetype="pdf")
                page = pdf_doc[page_number - 1]  # 0-indexed
                
                # Render at 2x for good quality
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                
                img_buffer = io.BytesIO(pix.tobytes("png"))
                img_buffer.seek(0)
                
                from flask import send_file
                return send_file(
                    img_buffer,
                    mimetype='image/png',
                    download_name=f'{guideline["name"]}_page_{page_number}.png'
                )
            except ImportError:
                return jsonify({'error': 'PDF image extraction not available. Install pdf2image or PyMuPDF'}), 500
                
    except Exception as e:
        print(f"Page image extraction error: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# =============================================================================
# PPTX GENERATION
# =============================================================================

@app.route('/api/generate-pptx/<int:guideline_id>', methods=['POST'])
def generate_pptx(guideline_id):
    """Generate a summary PPTX for a guideline."""
    data = request.get_json() or {}
    topic = data.get('topic')  # Optional - if None, generate full summary
    model = data.get('model', 'claude-sonnet-4-20250514')  # Default to Sonnet
    
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
    
    summary_prompt = f"""Extract ALL clinical recommendations from the following guideline content{topic_text} and organize them by recommendation Class.

GUIDELINE: {guideline['name']} ({guideline.get('year', 'N/A')})

CONTENT:
{context}

IMPORTANT: Do NOT summarize. Extract EVERY recommendation exactly as stated, organized by Class.

Return JSON in this exact format:
{{
    "title": "{guideline['name']}",
    "subtitle": "{topic or 'Complete Recommendations'}",
    "class_1": [
        {{
            "recommendation": "The exact recommendation text",
            "loe": "A/B-R/B-NR/C-LD/C-EO",
            "page": "page number if known"
        }}
    ],
    "class_2a": [
        {{
            "recommendation": "The exact recommendation text",
            "loe": "Level of Evidence",
            "page": "page number if known"
        }}
    ],
    "class_2b": [
        {{
            "recommendation": "The exact recommendation text", 
            "loe": "Level of Evidence",
            "page": "page number if known"
        }}
    ],
    "class_3": [
        {{
            "recommendation": "The exact recommendation text (Harm or No Benefit)",
            "loe": "Level of Evidence",
            "page": "page number if known"
        }}
    ]
}}

Rules:
1. Include EVERY recommendation found - do not summarize or shorten
2. Class I = Recommended (benefit >>> risk)
3. Class IIa = Reasonable (benefit >> risk)  
4. Class IIb = May be considered (benefit >= risk)
5. Class III = Not recommended (no benefit or harmful)
6. Keep the exact clinical language from the guideline
7. Return ONLY valid JSON, no other text."""

    try:
        # Handle different model types
        input_tokens = 0
        output_tokens = 0
        
        if model.startswith('gpt-'):
            # Use OpenAI
            import requests as req
            openai_key = os.environ.get('OPENAI_API_KEY')
            if not openai_key:
                return jsonify({'error': 'OpenAI API key not configured'}), 500
            
            request_body = {
                'model': model,
                'messages': [{'role': 'user', 'content': summary_prompt}],
                'temperature': 0.1
            }
            if model.startswith('gpt-5'):
                request_body['max_completion_tokens'] = 4000
            else:
                request_body['max_tokens'] = 4000
                
            resp = req.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {openai_key}', 'Content-Type': 'application/json'},
                json=request_body,
                timeout=120
            )
            if resp.status_code == 200:
                resp_data = resp.json()
                summary_text = resp_data['choices'][0]['message']['content']
                usage = resp_data.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
            else:
                return jsonify({'error': f'OpenAI error: {resp.text}'}), 500
                
        elif model.endswith('-thinking'):
            # Claude with extended thinking
            base_model = model.replace('-thinking', '')
            response = claude.messages.create(
                model=base_model,
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 10000},
                messages=[{"role": "user", "content": summary_prompt}]
            )
            # Extract text from thinking response
            summary_text = ""
            for block in response.content:
                if block.type == "text":
                    summary_text = block.text
                    break
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        else:
            # Standard Claude
            response = claude.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.1,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            summary_text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        
        # Calculate cost
        cost = calculate_cost(model.replace('-thinking', ''), input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens
        
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
            # Return the file with cost headers
            from flask import send_file, make_response
            response = make_response(send_file(
                pptx_path,
                as_attachment=True,
                download_name=f"{guideline['name'].replace(' ', '_')}{'_' + topic if topic else ''}_Summary.pptx",
                mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation'
            ))
            response.headers['X-Generation-Cost'] = f"{cost:.4f}"
            response.headers['X-Generation-Tokens'] = str(total_tokens)
            response.headers['Access-Control-Expose-Headers'] = 'X-Generation-Cost, X-Generation-Tokens'
            return response
        else:
            return jsonify({'error': 'Failed to create PPTX'}), 500
            
    except Exception as e:
        print(f"PPTX generation error: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def create_pptx(summary, guideline, topic=None):
    """Create a beautiful PowerPoint file from summary data."""
    if not PPTX_AVAILABLE:
        print("PPTX generation skipped - python-pptx not available")
        return None
    
    try:
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.oxml.ns import nsmap
        from pptx.oxml import parse_xml
        
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        
        # Beautiful color palette - Medical/Professional theme
        NAVY = RGBColor(15, 23, 42)           # Dark navy background
        SLATE = RGBColor(30, 41, 59)          # Slightly lighter navy
        WHITE = RGBColor(255, 255, 255)
        LIGHT_GRAY = RGBColor(226, 232, 240)
        
        # Class colors - vibrant but professional
        CLASS_COLORS = {
            'class_1': RGBColor(34, 197, 94),    # Green - strong
            'class_2a': RGBColor(59, 130, 246),   # Blue 
            'class_2b': RGBColor(251, 191, 36),   # Amber/Yellow
            'class_3': RGBColor(239, 68, 68),     # Red
        }
        
        # =====================
        # TITLE SLIDE
        # =====================
        slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
        
        # Dark gradient background (solid navy since gradients are complex)
        bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg_shape.fill.solid()
        bg_shape.fill.fore_color.rgb = NAVY
        bg_shape.line.fill.background()
        
        # Accent stripe at top
        top_stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.15))
        top_stripe.fill.solid()
        top_stripe.fill.fore_color.rgb = CLASS_COLORS['class_1']
        top_stripe.line.fill.background()
        
        # Heart/Medical icon area (decorative bar)
        accent_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.5), Inches(2.2), Inches(2.333), Inches(0.08))
        accent_bar.fill.solid()
        accent_bar.fill.fore_color.rgb = CLASS_COLORS['class_2a']
        accent_bar.line.fill.background()
        
        # Main title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(2.5), Inches(11.733), Inches(1.8))
        title_frame = title_box.text_frame
        title_frame.word_wrap = True
        title_para = title_frame.paragraphs[0]
        title_para.text = summary.get('title', guideline['name'])
        title_para.font.size = Pt(48)
        title_para.font.bold = True
        title_para.font.color.rgb = WHITE
        title_para.alignment = PP_ALIGN.CENTER
        
        # Subtitle
        subtitle_box = slide.shapes.add_textbox(Inches(0.8), Inches(4.5), Inches(11.733), Inches(0.8))
        subtitle_frame = subtitle_box.text_frame
        subtitle_para = subtitle_frame.paragraphs[0]
        subtitle_para.text = summary.get('subtitle', topic or 'Clinical Recommendations')
        subtitle_para.font.size = Pt(28)
        subtitle_para.font.color.rgb = LIGHT_GRAY
        subtitle_para.alignment = PP_ALIGN.CENTER
        
        # Source line at bottom
        source_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.733), Inches(0.5))
        source_frame = source_box.text_frame
        source_para = source_frame.paragraphs[0]
        source_para.text = f"📋 {guideline.get('source', 'ACC/AHA')} {guideline.get('year', '')} Guidelines"
        source_para.font.size = Pt(16)
        source_para.font.color.rgb = RGBColor(148, 163, 184)
        source_para.alignment = PP_ALIGN.CENTER
        
        # =====================
        # CLASS DEFINITION SLIDE  
        # =====================
        def_slide = prs.slides.add_slide(prs.slide_layouts[6])
        
        # Background
        bg2 = def_slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg2.fill.solid()
        bg2.fill.fore_color.rgb = NAVY
        bg2.line.fill.background()
        
        # Header
        header_box = def_slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.333), Inches(0.8))
        header_frame = header_box.text_frame
        header_para = header_frame.paragraphs[0]
        header_para.text = "Recommendation Classification"
        header_para.font.size = Pt(36)
        header_para.font.bold = True
        header_para.font.color.rgb = WHITE
        
        # Class boxes
        class_defs = [
            ('CLASS I', 'Recommended', 'Benefit >>> Risk', CLASS_COLORS['class_1']),
            ('CLASS IIa', 'Reasonable', 'Benefit >> Risk', CLASS_COLORS['class_2a']),
            ('CLASS IIb', 'May Be Considered', 'Benefit ≥ Risk', CLASS_COLORS['class_2b']),
            ('CLASS III', 'Not Recommended', 'No Benefit or Harm', CLASS_COLORS['class_3']),
        ]
        
        box_top = 1.5
        for class_name, label, desc, color in class_defs:
            # Colored indicator bar
            bar = def_slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(box_top), Inches(0.15), Inches(1.0))
            bar.fill.solid()
            bar.fill.fore_color.rgb = color
            bar.line.fill.background()
            
            # Class name
            name_box = def_slide.shapes.add_textbox(Inches(1.2), Inches(box_top), Inches(2.5), Inches(0.5))
            name_frame = name_box.text_frame
            name_para = name_frame.paragraphs[0]
            name_para.text = class_name
            name_para.font.size = Pt(24)
            name_para.font.bold = True
            name_para.font.color.rgb = color
            
            # Label
            label_box = def_slide.shapes.add_textbox(Inches(3.8), Inches(box_top), Inches(3), Inches(0.5))
            label_frame = label_box.text_frame
            label_para = label_frame.paragraphs[0]
            label_para.text = label
            label_para.font.size = Pt(22)
            label_para.font.color.rgb = WHITE
            
            # Description
            desc_box = def_slide.shapes.add_textbox(Inches(7.5), Inches(box_top + 0.1), Inches(5), Inches(0.5))
            desc_frame = desc_box.text_frame
            desc_para = desc_frame.paragraphs[0]
            desc_para.text = desc
            desc_para.font.size = Pt(18)
            desc_para.font.italic = True
            desc_para.font.color.rgb = LIGHT_GRAY
            
            box_top += 1.4
        
        # =====================
        # CONTENT SLIDES
        # =====================
        class_info = {
            'class_1': {'name': 'CLASS I', 'subtitle': 'Recommended', 'color': CLASS_COLORS['class_1']},
            'class_2a': {'name': 'CLASS IIa', 'subtitle': 'Reasonable', 'color': CLASS_COLORS['class_2a']},
            'class_2b': {'name': 'CLASS IIb', 'subtitle': 'May Be Considered', 'color': CLASS_COLORS['class_2b']},
            'class_3': {'name': 'CLASS III', 'subtitle': 'Not Recommended', 'color': CLASS_COLORS['class_3']},
        }
        
        for class_key, info in class_info.items():
            recommendations = summary.get(class_key, [])
            if not recommendations:
                continue
            
            recs_per_slide = 3  # Fewer per slide = more readable
            total_slides = -(-len(recommendations) // recs_per_slide)
            
            for slide_num, start_idx in enumerate(range(0, len(recommendations), recs_per_slide)):
                slide_recs = recommendations[start_idx:start_idx + recs_per_slide]
                
                content_slide = prs.slides.add_slide(prs.slide_layouts[6])
                
                # Dark background
                bg = content_slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
                bg.fill.solid()
                bg.fill.fore_color.rgb = NAVY
                bg.line.fill.background()
                
                # Colored top bar for this class
                top_bar = content_slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
                top_bar.fill.solid()
                top_bar.fill.fore_color.rgb = info['color']
                top_bar.line.fill.background()
                
                # Left accent bar
                left_bar = content_slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(0.5), Inches(0.12), Inches(1.0))
                left_bar.fill.solid()
                left_bar.fill.fore_color.rgb = info['color']
                left_bar.line.fill.background()
                
                # Class header
                header_box = content_slide.shapes.add_textbox(Inches(0.7), Inches(0.4), Inches(8), Inches(0.6))
                header_frame = header_box.text_frame
                header_para = header_frame.paragraphs[0]
                header_para.text = info['name']
                header_para.font.size = Pt(36)
                header_para.font.bold = True
                header_para.font.color.rgb = info['color']
                
                # Subtitle
                sub_box = content_slide.shapes.add_textbox(Inches(0.7), Inches(1.0), Inches(8), Inches(0.4))
                sub_frame = sub_box.text_frame
                sub_para = sub_frame.paragraphs[0]
                sub_para.text = info['subtitle']
                sub_para.font.size = Pt(20)
                sub_para.font.color.rgb = LIGHT_GRAY
                
                # Page indicator
                if total_slides > 1:
                    page_box = content_slide.shapes.add_textbox(Inches(11), Inches(0.5), Inches(2), Inches(0.4))
                    page_frame = page_box.text_frame
                    page_para = page_frame.paragraphs[0]
                    page_para.text = f"{slide_num + 1}/{total_slides}"
                    page_para.font.size = Pt(16)
                    page_para.font.color.rgb = RGBColor(100, 116, 139)
                    page_para.alignment = PP_ALIGN.RIGHT
                
                # Recommendations
                current_top = 1.7
                for i, rec in enumerate(slide_recs):
                    rec_text = rec.get('recommendation', '') if isinstance(rec, dict) else str(rec)
                    loe = rec.get('loe', '') if isinstance(rec, dict) else ''
                    
                    # Number circle/bullet
                    num_box = content_slide.shapes.add_textbox(Inches(0.5), Inches(current_top), Inches(0.5), Inches(0.5))
                    num_frame = num_box.text_frame
                    num_para = num_frame.paragraphs[0]
                    num_para.text = f"{start_idx + i + 1}"
                    num_para.font.size = Pt(18)
                    num_para.font.bold = True
                    num_para.font.color.rgb = info['color']
                    
                    # Recommendation text
                    rec_box = content_slide.shapes.add_textbox(Inches(1.0), Inches(current_top), Inches(11.5), Inches(1.5))
                    rec_frame = rec_box.text_frame
                    rec_frame.word_wrap = True
                    rec_para = rec_frame.paragraphs[0]
                    rec_para.text = rec_text
                    rec_para.font.size = Pt(20)
                    rec_para.font.color.rgb = WHITE
                    rec_para.line_spacing = 1.2
                    
                    # LOE badge
                    if loe:
                        loe_box = content_slide.shapes.add_textbox(Inches(1.0), Inches(current_top + 1.1), Inches(4), Inches(0.35))
                        loe_frame = loe_box.text_frame
                        loe_para = loe_frame.paragraphs[0]
                        loe_para.text = f"Level of Evidence: {loe}"
                        loe_para.font.size = Pt(14)
                        loe_para.font.italic = True
                        loe_para.font.color.rgb = RGBColor(148, 163, 184)
                    
                    current_top += 1.8
        
        # Save
        output_path = f'/tmp/pptx_{guideline["id"]}_{topic or "full"}.pptx'
        prs.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"PPTX creation error: {e}")
        import traceback
        print(traceback.format_exc())
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
