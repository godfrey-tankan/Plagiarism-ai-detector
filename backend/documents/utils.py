import PyPDF2
import docx
import re
import textstat
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from nltk.tokenize import sent_tokenize # New import for sentence tokenization
import numpy as np # New import for numerical operations

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab') 
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    nltk.download('punkt_tab') 

logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Normalizes whitespace and attempts to re-flow text to improve readability from PDFs.
    - Replaces non-breaking spaces and zero-width spaces.
    - Removes common problematic characters (e.g., control characters, invisible unicode).
    - Attempts to join hyphens at line breaks.
    - Aggressively handles newlines:
        - Replaces a newline not clearly indicating a paragraph break with a space.
        - Reduces excessive blank lines to at most two.
    - Replaces any remaining multiple spaces/tabs with single space.
    - Trims leading/trailing whitespace.
    """
    text = text.replace('\xa0', ' ').replace('\u200b', '') 
    text = re.sub(r'-\n', '', text) 
    text = re.sub(r'(?<!\n)\n(?!\n|\.)', ' ', text)
    text = re.sub(r'(\s*\n\s*){2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

def extract_text_from_file(file):
    """Extract text with better error handling and post-processing."""
    text = ""
    logger.info(f"Starting extraction for {file.name}")
    file_extension = file.name.lower()

    if file_extension.endswith('.pdf'):
        try:
            file.seek(0)
            reader = PyPDF2.PdfReader(file)
            if not reader.pages:
                raise ValueError("PDF has no readable pages")
            for i, page in enumerate(reader.pages):
                chunk = page.extract_text() or ''
                text += chunk + "\n" 
                if i >= 4 and len(text.strip()) < 200 and i < len(reader.pages) - 1:
                    logger.warning(f"PDF {file.name} looks image-based or contains insufficient text after {i+1} pages.")
            if len(text.strip()) < 100: 
                raise ValueError("PDF contains insufficient text for analysis or is image-based.")
        except Exception as e:
            logger.error(f"PDF extraction error: {e}", exc_info=True)
            raise ValueError(f"Failed to extract PDF text: {e}. It might be an image-based PDF or corrupted.")

    elif file_extension.endswith('.docx'):
        try:
            doc = docx.Document(file)
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}", exc_info=True)
            raise ValueError(f"Failed to extract DOCX text: {e}")

    elif file_extension.endswith('.txt'):
        text = file.read().decode('utf-8', errors='ignore')

    else:
        raise ValueError("Unsupported format. Only PDF, DOCX, TXT allowed.")

    cleaned_text = clean_text(text)
    logger.info(f"Extracted and cleaned {len(cleaned_text)} characters from {file.name}")
    return cleaned_text

def analyze_text(content_hash, text):
    """
    Enhanced plagiarism detection against other uploaded documents.
    Combines character n-gram cosine similarity and sentence-level Jaccard similarity.
    Includes confidence score for highlights.
    """
    others = list(
        Document.objects
        .exclude(content_hash=content_hash)
        .values_list('content', flat=True)
    )
    if not others:
        logger.info("No other documents for plagiarism check. Returning 0.0% score.")
        return {'score': 0.0, 'highlights': []}

    all_highlights = []
    total_matched_chars = set()
    total_text_len = len(text)
    
    logger.info("Starting character n-gram plagiarism check...")
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(5, 9))
    char_corpus = [text] + others
    char_mat = char_vec.fit_transform(char_corpus)

    char_window = 150
    char_step = 50
    char_plagiarism_threshold = 0.4

    for start in range(0, total_text_len - char_window + 1, char_step):
        snippet = text[start:start + char_window]
        if not snippet.strip():
            continue
        
        snippet_vec = char_vec.transform([snippet])
        sim = cosine_similarity(snippet_vec, char_mat[1:])[0]
        max_sim = sim.max() if sim.size > 0 else 0.0

        if max_sim > char_plagiarism_threshold:
            end = start + char_window
            total_matched_chars.update(range(start, end))
            all_highlights.append({
                'type': 'plagiarism',
                'position': calculate_position(text, start, end),
                'confidence': round(max_sim * 100, 1)
            })
    logger.info(f"Character n-gram check found {len(total_matched_chars)} matched characters.")

    logger.info("Starting sentence-level Jaccard similarity check...")
    text_sentences = sent_tokenize(text)
    
    def preprocess_sentence(s):
        return set(re.findall(r'\w+', s.lower()))

    processed_text_sentences = [preprocess_sentence(s) for s in text_sentences]
    processed_other_docs_sentences = []
    for other_doc_text in others:
        processed_other_docs_sentences.extend([preprocess_sentence(s) for s in sent_tokenize(other_doc_text)])

    sentence_plagiarism_threshold = 0.6
    
    sentence_highlights_found = []
    
    for i, current_sentence_words in enumerate(processed_text_sentences):
        if not current_sentence_words: continue
        
        original_sentence_start = text.find(text_sentences[i])
        original_sentence_end = original_sentence_start + len(text_sentences[i])

        for other_sentence_words in processed_other_docs_sentences:
            if not other_sentence_words: continue

            intersection = len(current_sentence_words.intersection(other_sentence_words))
            union = len(current_sentence_words.union(other_sentence_words))
            jaccard_similarity = intersection / union if union > 0 else 0.0

            if jaccard_similarity > sentence_plagiarism_threshold:
                confidence = round(jaccard_similarity * 100, 1)
                
                is_already_covered = False
                for char_hl in all_highlights:
                    char_hl_start = int(char_hl['position']['x'] / 100 * total_text_len)
                    char_hl_end = int((char_hl['position']['x'] + char_hl['position']['width']) / 100 * total_text_len)
                    if (char_hl_start <= original_sentence_start < char_hl_end) or \
                        (char_hl_start < original_sentence_end <= char_hl_end):
                        is_already_covered = True
                        break 

                if not is_already_covered:
                    sentence_highlights_found.append({
                        'type': 'plagiarism',
                        'position': calculate_position(text, original_sentence_start, original_sentence_end),
                        'confidence': confidence
                    })
                break

    all_highlights.extend(sentence_highlights_found)
    logger.info(f"Sentence-level check found {len(sentence_highlights_found)} sentence matches.")
    matched_chars_from_all_highlights = set()
    for hl in all_highlights:
        hl_start = int(hl['position']['x'] / 100 * total_text_len)
        hl_end = int((hl['position']['x'] + hl['position']['width']) / 100 * total_text_len)
        matched_chars_from_all_highlights.update(range(hl_start, hl_end))

    final_score = round(len(matched_chars_from_all_highlights) / total_text_len * 100, 1) if total_text_len else 0.0

    return {
        'score': min(final_score, 100.0),
        'highlights': all_highlights
    }


AI_DETECTOR = None
AI_DETECTOR_MODEL_NAME = 'roberta-base-openai-detector' # OR 'distilbert-base-uncased-finetuned-sst-2-english' or others
AI_DETECTOR_LOADED = False

def load_ai_detector():
    global AI_DETECTOR, AI_DETECTOR_LOADED
    if not AI_DETECTOR_LOADED:
        logger.info(f"Initializing AI detector model: {AI_DETECTOR_MODEL_NAME}...")
        try:
            # Explicitly load model and tokenizer for better control
            tokenizer = AutoTokenizer.from_pretrained(AI_DETECTOR_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(AI_DETECTOR_MODEL_NAME)
            
            # Use GPU if available
            device = 0 if torch.cuda.is_available() else -1
            
            AI_DETECTOR = pipeline(
                'text-classification',
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                max_length=512,
                device=device
            )
            AI_DETECTOR_LOADED = True
            logger.info(f"AI detector '{AI_DETECTOR_MODEL_NAME}' initialized successfully on device: {device}.")
        except Exception as e:
            logger.critical(f"Failed to load AI detector model '{AI_DETECTOR_MODEL_NAME}': {e}. AI detection will be disabled.", exc_info=True)
            AI_DETECTOR = None # Disable detector if it fails to load
            AI_DETECTOR_LOADED = True # Mark as attempted to load


def check_ai_probability(text, plagiarism_highlights=None, plagiarism_score=0):
    """
    AI detection: Uses a robust pre-trained Hugging Face model with improved chunking and scoring.
    Includes confidence score for highlights.
    """
    plagiarism_score = plagiarism_score or 0
    min_text_length_for_ai = 300  # Minimum length for AI analysis
    min_chunk_length = 50      # Minimum length for a chunk to be analyzed

    if len(text) < min_text_length_for_ai:
        logger.info(f"Text too short for AI analysis (min {min_text_length_for_ai} chars): {len(text)} chars.")
        return {'score': 0.0, 'highlights': []}

    load_ai_detector() # Ensure the detector is loaded
    if AI_DETECTOR is None:
        return {'score': 0.0, 'highlights': []}

    chunk_size = 512
    chunk_overlap_ratio = 0.25 # 25% overlap
    chunk_overlap = int(chunk_size * chunk_overlap_ratio)

    all_chunk_probabilities = []
    all_highlights = []
    
    # Keep track of highlighted character ranges to avoid redundant highlights for same content
    highlighted_ranges = [] 

    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        
        # Skip chunks that are too short or just whitespace after stripping
        if len(chunk.strip()) < min_chunk_length:
            logger.debug(f"Skipping chunk at index {i} due to insufficient length ({len(chunk.strip())} chars) or being whitespace-only.")
            continue

        try:
            prediction = AI_DETECTOR(chunk)[0]
            label = prediction['label'].lower()
            score = prediction['score'] 

            ai_probability = 0.0
            if label == 'machine' or label == 'ai': # If model predicts AI/Machine
                ai_probability = score * 100
            elif label == 'human': 
                ai_probability = (1 - score) * 100

            all_chunk_probabilities.append(ai_probability)
            logger.debug(f"Chunk from {i} to {i+len(chunk)} (len {len(chunk)}): Label='{label}', RawScore={prediction['score']:.4f}, AI_Prob={ai_probability:.1f}%")

            highlight_threshold_for_ai = 10 
            if ai_probability >= highlight_threshold_for_ai:
                current_start = i
                current_end = min(i + chunk_size, len(text))
                
                is_covered = False
                for existing_start, existing_end in highlighted_ranges:
                    if not (current_end < existing_start or current_start > existing_end):
                        is_covered = True
                        break
                
                if not is_covered:
                    all_highlights.append({
                        'type': 'ai',
                        'position': calculate_position(text, current_start, current_end),
                        'confidence': round(ai_probability, 1)
                    })
                    highlighted_ranges.append((current_start, current_end))

        except Exception as e:
            logger.error(f"Error during AI detection for chunk at index {i}: {e}", exc_info=True)
            continue

    avg_ai_probability = np.mean(all_chunk_probabilities) if all_chunk_probabilities else 0.0
    final_ai_score = round(avg_ai_probability, 1)
    cap = max(0.0, 100.0 - plagiarism_score)
    final_ai_score_capped = min(final_ai_score, cap)

    logger.info(f"AI detection total chunks analyzed: {len(all_chunk_probabilities)}. Average AI probability: {final_ai_score:.1f}%. Final capped AI score: {final_ai_score_capped:.1f}%")
    return {
        'score': final_ai_score_capped,
        'highlights': all_highlights
    }

def calculate_position(full_text, start, end):
    """Return percentage-based box for front-end. Ensures robustness."""
    total = len(full_text)
    if total == 0 or start < 0 or end < 0 or start >= total:
        return {'page': 1, 'x': 0, 'y': 0, 'width': 0, 'height': 2}
    end = min(end, total)
    if start > end:
        start = end 

    width_percent = round((end - start) / total * 100, 2)
    x_percent = round(start / total * 100, 2)

    return {
        'page': 1, 
        'x': x_percent,
        'y': 0,  
        'width': width_percent,
        'height': 2 
    }


def calculate_document_stats(text):
    """Word count, char count, pages, reading time (mins)."""
    words = len(text.split())
    chars = len(text)
    pages = max(1, (chars // 1800) + 1)
    try:
        read_time_seconds = textstat.reading_time(text, wpm=200) 
        read = max(1, round(read_time_seconds / 60)) 
    except Exception:
        read = max(1, words // 200) 
    return {
        'word_count': words,
        'character_count': chars,
        'page_count': pages,
        'reading_time': read
    }
    
# def check_ai_probability(text):
#     """Return AI detection with sentence positions"""
#     if not AI_MODEL:
#         return {'score': 0, 'highlights': []}
    
#     text = (text or "").strip()
#     if not text:
#         return {'score': 0, 'highlights': []}

#     try:
#         sentences = re.split(r'(?<=[.!?]) +', text)
#         highlights = []
#         total_score = 0
#         detected = 0
        
#         for sentence in sentences:
#             if not sentence.strip():
#                 continue
                
#             result = AI_MODEL(sentence[:512])[0]
            
#             if result['label'] == 'AI' and result['score'] > 0.7:
#                 start = text.find(sentence)
#                 if start != -1:
#                     highlights.append({
#                         'type': 'ai',
#                         'position': calculate_position(text, start, start + len(sentence))
#                     })
#                     total_score += result['score']
#                     detected += 1

#         avg_score = (total_score / detected * 100) if detected > 0 else 0
        
#         return {
#             'score': round(avg_score, 2),
#             'highlights': highlights
#         }

#     except Exception as e:
#         logger.error(f"AI detection error: {str(e)}")
#         return {'score': 0, 'highlights': []}
# def check_ai_probability(text):

# def analyze_text(content_hash,text):
#     """Analyze text for plagiarism using TF-IDF and cosine similarity"""
#     existing_docs = Document.objects.exclude(content_hash=content_hash).values_list('content', flat=True)
    
#     # Handle empty document database
#     if not existing_docs:
#         return {
#             'score': 0.0,
#             'matches': [],
#             'highlighted': text
#         }

#     vectorizer = TfidfVectorizer(stop_words='english')
#     vectors = vectorizer.fit_transform([text] + list(existing_docs))
    
#     similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:])
#     similarities = similarity_matrix[0]
    
#     matches = []
#     for idx, score in enumerate(similarities):
#         if score > 0.2:  
#             doc = Document.objects.all()[idx]
#             matches.append({
#                 'source': doc.file.name,
#                 'similarity': round(score * 100, 2),
#                 'url': f"/documents/{doc.id}/"  
#             })
    
#     return {
#         'score': round(max(similarities, default=0) * 100, 2),
#         'matches': sorted(matches, key=lambda x: x['similarity'], reverse=True),
#         'highlighted': highlight_matches(text, existing_docs)
#     }

# # def check_ai_probability(text):
# #     """Lightweight AI detection using a faster model"""
# #     try:
#     print("AI detection started")
#     """Lightweight AI detection using a faster model"""
#     # return {'score': 0, 'highlights': []}
#     try:
#         # Enhanced validation
#         if not text or len(re.findall(r'\w+', text)) < 3:  # At least 3 words
#             return {'score': 0, 'highlights': []}
#         if not hasattr(check_ai_probability, 'ai_detector'):
#             check_ai_probability.ai_detector = pipeline(
#                 'text-classification', 
#                 model='Hello-SimpleAI/chatgpt-detector-roberta',
#                 truncation=True,
#                 max_length=512,
#                 device=0 if torch.cuda.is_available() else -1
#             )

#         processed_text = text[:1024].strip()
        
#         # Final safety check
#         if not processed_text or len(processed_text) < 10:
#             return {'score': 0, 'highlights': []}

#         result = check_ai_probability.ai_detector(processed_text)
        
#         # Handle empty results
#         if not result:
#             return {'score': 0, 'highlights': []}
            
#         return {
#             'score': round(result[0]['score'] * 100, 2),
#             'highlights': []  # Add your highlight logic here
#         }
        
#     except ValueError as ve:
#         if "0 sample(s)" in str(ve):
#             return {'score': 0, 'highlights': []}
#         raise
#     except Exception as e:
#         print(f"AI Detection Error: {str(e)}")
#         return {'score': 0, 'highlights': []}
# #         # Use a smaller distilled model
# #         ai_detector = pipeline(
# #             'text-classification', 
# #             model='Hello-SimpleAI/chatgpt-detector-roberta',
# #             truncation=True,
# #             max_length=512,
# #             device=0 if torch.cuda.is_available() else -1  # Use GPU if available
# #         )
        
# #         # Process first 1024 characters only for speed
# #         result = ai_detector(text[:1024])
# #         return round(result[0]['score'] * 100, 2)
# #     except Exception as e:
# #         print(f"AI Detection Error: {str(e)}")
# #         return 0.0
    
    
# def check_ai_probability(text):
#     """AI detection with proper resource management"""
#     try:
#         if not torch.cuda.is_available():
#             print("Warning: Using CPU for AI detection - this will be slow!")

#         # Use smaller model for better performance
#         ai_detector = pipeline(
#             'text-classification',
#             model='distilbert-base-uncased',  # Lighter model
#             device=0 if torch.cuda.is_available() else -1,
#             truncation=True,
#             max_length=512
#         )

#         # Process first 512 characters only
#         result = ai_detector(text[:512])
#         return round(result[0]['score'] * 100, 2)
        
#     except Exception as e:
#         print(f"AI Detection Failed: {str(e)}")
#         return 0.0  # Return safe default    
    