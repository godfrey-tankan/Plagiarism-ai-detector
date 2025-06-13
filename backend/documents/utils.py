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
from nltk.tokenize import sent_tokenize
import numpy as np
import os

import google.generativeai as genai
from django.conf import settings

# Attempt NLTK download if not present
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
GEMINI_MAX_WORDS = 1000 # Strict word limit for all Gemini API calls
HIGH_CONFIDENCE_AI_THRESHOLD = 75.0 # Min probability for AI detection to be considered "high confidence"
MIN_TEXT_LENGTH_FOR_AI_ANALYSIS = 300 # Minimum characters required to run AI detection
ROBERTA_CHUNK_SIZE = 512
ROBERTA_CHUNK_OVERLAP_RATIO = 0.25
ROBERTA_MIN_CHUNK_LENGTH = 50
# --- Global Model Loaders ---
AI_DETECTOR_ROBERTA = None
AI_DETECTOR_ROBERTA_MODEL_NAME = 'roberta-base-openai-detector'
AI_DETECTOR_ROBERTA_LOADED = False

GEMINI_MODEL = None
AI_DETECTOR_GEMINI_MODEL_NAME = 'gemini-2.0-flash'
GEMINI_LOADED = False


def analyze_text_for_plagiarism_and_ai(text_content, content_hash=None):
    """
    Consolidates the plagiarism and AI analysis process for a given text.
    Returns combined scores, highlights, and document statistics.
    
    Args:
        text_content (str): The text content of the document to analyze.
        content_hash (str, optional): The hash of the content, used for internal plagiarism
        to exclude the document itself. If None, it's assumed
        the document is new or temporary.

    Returns:
        tuple: (plagiarism_score, ai_score, originality_score, highlights, stats)
    """
    logger.info(f"Starting combined analysis for text of length: {len(text_content)} characters.")

    stats = calculate_document_stats(text_content)

    # Internal Plagiarism Detection (analyze_text)
    plagiarism_results_internal = analyze_text(content_hash, text_content)
    final_plagiarism_score = plagiarism_results_internal['score']
    all_plagiarism_highlights = plagiarism_results_internal['highlights']

    # AI Detection (RoBERTa & Gemini for overall)
    ai_detection_results = check_ai_probability(text_content)
    ai_score_raw = ai_detection_results['score']
    ai_highlights = ai_detection_results['highlights']

    # Cap AI score based on remaining originality after plagiarism
    remaining_original_percentage = max(0.0, 100.0 - final_plagiarism_score)
    final_ai_score = min(ai_score_raw, remaining_original_percentage)
    final_ai_score = round(final_ai_score, 1)

    # Calculate final originality score
    final_originality_score = round(max(0.0, 100.0 - (final_plagiarism_score + final_ai_score)), 1)

    # Combine all raw highlights and resolve overlaps/priorities
    all_combined_highlights_raw = all_plagiarism_highlights + ai_highlights
    final_resolved_highlights = resolve_overlapping_highlights(all_combined_highlights_raw)

    logger.info(f"Combined analysis finished. Plag: {final_plagiarism_score}%, AI: {final_ai_score}%, Orig: {final_originality_score}%. Resolved {len(final_resolved_highlights)} highlights.")

    return final_plagiarism_score, final_ai_score, final_originality_score, final_resolved_highlights, stats


def load_roberta_detector():
    global AI_DETECTOR_ROBERTA, AI_DETECTOR_ROBERTA_LOADED
    if not AI_DETECTOR_ROBERTA_LOADED:
        logger.info(f"Initializing RoBERTa AI detector model: {AI_DETECTOR_ROBERTA_MODEL_NAME}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(AI_DETECTOR_ROBERTA_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(AI_DETECTOR_ROBERTA_MODEL_NAME)
            device = 0 if torch.cuda.is_available() else -1 # Use GPU if available
            AI_DETECTOR_ROBERTA = pipeline(
                'text-classification',
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                max_length=ROBERTA_CHUNK_SIZE,
                device=device
            )
            AI_DETECTOR_ROBERTA_LOADED = True
            logger.info(f"RoBERTa detector '{AI_DETECTOR_ROBERTA_MODEL_NAME}' initialized successfully on device: {device}.")
        except Exception as e:
            logger.critical(f"Failed to load RoBERTa detector model '{AI_DETECTOR_ROBERTA_MODEL_NAME}': {e}. RoBERTa detection will be disabled.", exc_info=True)
            AI_DETECTOR_ROBERTA = None
            AI_DETECTOR_ROBERTA_LOADED = True

def load_gemini_model():
    global GEMINI_MODEL, GEMINI_LOADED
    if not GEMINI_LOADED:
        logger.info("Initializing Gemini model for AI detection and semantic plagiarism...")
        try:
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                logger.error("GEMINI_API_KEY is not set in Django settings. Gemini features will be disabled.")
                GEMINI_MODEL = None
                GEMINI_LOADED = True
                return

            genai.configure(api_key=api_key)
            GEMINI_MODEL = genai.GenerativeModel(AI_DETECTOR_GEMINI_MODEL_NAME)
            GEMINI_LOADED = True
            logger.info(f"Gemini model '{AI_DETECTOR_GEMINI_MODEL_NAME}' initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Gemini model '{AI_DETECTOR_GEMINI_MODEL_NAME}': {e}. Gemini features will be disabled.", exc_info=True)
            GEMINI_MODEL = None
            GEMINI_LOADED = True

def truncate_text_by_words(text, num_words):
    """Truncates text to the first `num_words` and appends '...' if truncated."""
    words = text.split()
    if len(words) > num_words:
        return " ".join(words[:num_words]) + "..."
    return text

def clean_text(text):
    """
    Normalizes whitespace and cleans up common formatting issues from extracted text.
    """
    text = text.replace('\xa0', ' ').replace('\u200b', '') # Non-breaking space, zero-width space
    text = re.sub(r'-\s*\n', '', text) # Join hyphenated words broken by newline
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) # Replace single newlines (likely soft wraps) with space
    text = re.sub(r'(\s*\n\s*){2,}', '\n\n', text) # Reduce excessive blank lines
    text = re.sub(r'[ \t]+', ' ', text) # Replace multiple spaces/tabs with single space
    return text.strip()

def extract_text_from_file(file):
    """Extract text from PDF, DOCX, or TXT files with robust error handling."""
    text = ""
    logger.info(f"Attempting text extraction for file: {file.name}")
    file_extension = file.name.lower()

    if file_extension.endswith('.pdf'):
        try:
            file.seek(0) # Ensure file pointer is at the beginning
            reader = PyPDF2.PdfReader(file)
            if not reader.pages:
                raise ValueError("PDF has no readable pages or is empty.")
            
            # Extract text page by page
            for i, page in enumerate(reader.pages):
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += extracted_page_text + "\n"
                
                # Heuristic for image-based PDFs: check early if text is insufficient
                if i >= 4 and len(text.strip()) < MIN_TEXT_LENGTH_FOR_AI_ANALYSIS and i < len(reader.pages) - 1:
                    logger.warning(f"PDF {file.name} might be image-based or lacks sufficient text after {i+1} pages. Continuing extraction...")
            
            if len(text.strip()) < 100: # Final check for very short or empty PDFs
                raise ValueError("PDF contains insufficient text for analysis or is entirely image-based.")

        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Failed to read PDF file (corrupted?): {e}", exc_info=True)
            raise ValueError(f"Failed to read PDF file: {e}. It might be corrupted or malformed.")
        except Exception as e:
            logger.error(f"General PDF extraction error: {e}", exc_info=True)
            raise ValueError(f"Failed to extract PDF text: {e}. It might be an image-based PDF or have an unsupported structure.")

    elif file_extension.endswith('.docx'):
        try:
            doc = docx.Document(file)
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}", exc_info=True)
            raise ValueError(f"Failed to extract DOCX text: {e}. The file might be corrupted.")

    elif file_extension.endswith('.txt'):
        try:
            text = file.read().decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"TXT extraction error: {e}", exc_info=True)
            raise ValueError(f"Failed to extract TXT text: {e}. Encoding issue or corrupted file.")

    else:
        raise ValueError("Unsupported file format. Only PDF, DOCX, TXT allowed.")

    cleaned_text = clean_text(text)
    logger.info(f"Successfully extracted and cleaned {len(cleaned_text)} characters from {file.name}")
    return cleaned_text


def calculate_document_stats(text):
    """Calculates word count, character count, estimated pages, and reading time."""
    words = len(text.split())
    chars = len(text)
    pages = max(1, (chars // 1800) + 1)
    
    try:
        read_time_seconds = textstat.reading_time(text)
        read_minutes = max(1, round(read_time_seconds / 60))
    except Exception as e:
        logger.warning(f"Could not calculate reading time accurately: {e}. Falling back to words/200.", exc_info=True)
        read_minutes = max(1, words // 200)
    
    return {
        'word_count': words,
        'character_count': chars,
        'page_count': pages,
        'reading_time': read_minutes
    }

# --- Plagiarism Detection (Internal docs) ---
def analyze_text(content_hash, text):
    """
    Performs internal plagiarism detection using character n-gram cosine similarity and
    sentence-level Jaccard similarity against other documents in the system.
    """
    others = list(
        Document.objects
        .exclude(content_hash=content_hash)
        .values_list('content', flat=True)
    )
    if not others:
        logger.info("No other documents found for internal plagiarism check. Returning 0.0% score.")
        return {'score': 0.0, 'highlights': []}

    all_highlights = []
    total_text_len = len(text)

    # 1. Character N-gram based check (for exact or near-exact phrases)
    logger.info("Starting character n-gram plagiarism check...")
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(5, 9))
    char_corpus = [text] + others # Document being checked + all other documents
    try:
        char_mat = char_vec.fit_transform(char_corpus)
    except ValueError:
        logger.warning("N-gram vectorizer could not fit vocabulary. Text might be too short or unusual.")
        return {'score': 0.0, 'highlights': []}


    char_window = 150
    char_step = 50
    char_plagiarism_threshold = 0.4

    for start in range(0, total_text_len - char_window + 1, char_step):
        snippet = text[start:start + char_window]
        if not snippet.strip(): continue # Skip empty snippets

        snippet_vec = char_vec.transform([snippet])
        # Compare snippet against all other documents (char_mat[1:])
        sim = cosine_similarity(snippet_vec, char_mat[1:])[0]
        max_sim = sim.max() if sim.size > 0 else 0.0

        if max_sim > char_plagiarism_threshold:
            end = start + char_window
            all_highlights.append({
                'type': 'plagiarism',
                'start': start, 
                'end': end,     
                'confidence': round(max_sim * 100, 1),
                'source': 'internal_ngram'
            })
    logger.info(f"Character n-gram check found {len([h for h in all_highlights if h['source'] == 'internal_ngram'])} potential highlights.")

    # 2. Sentence-level Jaccard Similarity (for rephrased sentences)
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
                for hl in all_highlights: 
                    if (hl['start'] <= original_sentence_start < hl['end']) or \
                        (hl['start'] < original_sentence_end <= hl['end']) or \
                        (original_sentence_start <= hl['start'] and original_sentence_end >= hl['end']):
                        is_already_covered = True
                        break
                if not is_already_covered:
                    for hl in sentence_highlights_found:
                        if (hl['start'] <= original_sentence_start < hl['end']) or \
                            (hl['start'] < original_sentence_end <= hl['end']) or \
                            (original_sentence_start <= hl['start'] and original_sentence_end >= hl['end']):
                            is_already_covered = True
                            break

                if not is_already_covered:
                    sentence_highlights_found.append({
                        'type': 'plagiarism',
                        'start': original_sentence_start, 
                        'end': original_sentence_end,     
                        'confidence': confidence,
                        'source': 'internal_jaccard'
                    })
                break 

    all_highlights.extend(sentence_highlights_found)
    logger.info(f"Sentence-level check found {len(sentence_highlights_found)} additional matches.")

    matched_chars_from_all_highlights = set()
    for hl in all_highlights:
        if hl['source'].startswith('internal'): 
            matched_chars_from_all_highlights.update(range(hl['start'], hl['end']))

    final_score = round(len(matched_chars_from_all_highlights) / total_text_len * 100, 1) if total_text_len else 0.0

    return {
        'score': min(final_score, 100.0),
        'highlights': all_highlights
    }

# --- AI Detection Logic ---
def check_ai_probability(text):
    """
    Detects AI-generated content using RoBERTa and Gemini.
    Only high-confidence detections contribute to the overall score and highlights.
    The final AI score is an average of these high-confidence probabilities.
    """
    if len(text) < MIN_TEXT_LENGTH_FOR_AI_ANALYSIS:
        logger.info(f"Text too short for AI analysis (min {MIN_TEXT_LENGTH_FOR_AI_ANALYSIS} chars): {len(text)} chars.")
        return {'score': 0.0, 'highlights': []}

    load_roberta_detector()
    load_gemini_model()

    high_confidence_ai_probabilities = [] # Stores probabilities >= HIGH_CONFIDENCE_AI_THRESHOLD
    all_highlights = []

    total_text_len = len(text)

    # 1. RoBERTa-based AI Detection (chunk-by-chunk)
    if AI_DETECTOR_ROBERTA:
        logger.info("Running RoBERTa AI detection...")
        chunk_overlap = int(ROBERTA_CHUNK_SIZE * ROBERTA_CHUNK_OVERLAP_RATIO)

        for i in range(0, len(text), ROBERTA_CHUNK_SIZE - chunk_overlap):
            chunk = text[i:i + ROBERTA_CHUNK_SIZE]
            if len(chunk.strip()) < ROBERTA_MIN_CHUNK_LENGTH:
                continue

            try:
                prediction = AI_DETECTOR_ROBERTA(chunk)[0]
                label = prediction['label'].lower()
                score = prediction['score']

                ai_probability = 0.0
                if label == 'machine' or label == 'ai':
                    ai_probability = score * 100
                elif label == 'human':
                    ai_probability = (1 - score) * 100
                
                logger.debug(f"RoBERTa Chunk from {i} (len {len(chunk)}): Label='{label}', RawScore={prediction['score']:.4f}, AI_Prob={ai_probability:.1f}%")

                # Only consider for score and highlighting if above high confidence threshold
                if ai_probability >= HIGH_CONFIDENCE_AI_THRESHOLD:
                    high_confidence_ai_probabilities.append(ai_probability)
                    current_start = i
                    current_end = min(i + ROBERTA_CHUNK_SIZE, len(text))
                    
                    # Remove internal 'is_covered' logic here. Global resolver handles it.
                    all_highlights.append({
                        'type': 'ai',
                        'start': current_start, 
                        'end': current_end,     
                        'confidence': round(ai_probability, 1),
                        'source': 'roberta'
                    })

            except Exception as e:
                logger.error(f"Error during RoBERTa AI detection for chunk at index {i}: {e}", exc_info=True)
                continue
    else:
        logger.warning("RoBERTa detector not loaded, skipping RoBERTa AI detection.")

    # 2. Gemini-based AI Detection (overall document assessment)
    if GEMINI_MODEL:
        logger.info("Running Gemini AI detection (overall document assessment)...")
        truncated_text_for_gemini = truncate_text_by_words(text, GEMINI_MAX_WORDS)

        gemini_prompt = (
            "Analyze the following text and determine the likelihood that it was generated by an AI. "
            "Respond with a single number representing the percentage probability (0-100) that it is AI-generated, "
            "followed by a short, concise reasoning. Example: '85% - Highly consistent phrasing.'\n\nText:\n"
            f"{truncated_text_for_gemini}"
        )
        try:
            response = GEMINI_MODEL.generate_content(gemini_prompt)
            gemini_response_text = response.text.strip()
            match = re.match(r'(\d+)%', gemini_response_text)
            if match:
                gemini_ai_prob = float(match.group(1))
                logger.debug(f"Gemini AI Probability: {gemini_ai_prob:.1f}% - Response: {gemini_response_text}")

                # Only consider for score and highlighting if above high confidence threshold
                if gemini_ai_prob >= HIGH_CONFIDENCE_AI_THRESHOLD:
                    high_confidence_ai_probabilities.append(gemini_ai_prob)
                    
                    all_highlights.append({
                        'type': 'ai',
                        'start': 0,         
                        'end': len(text),  
                        'confidence': round(gemini_ai_prob, 1),
                        'source': 'gemini_overall'
                    })
            else:
                logger.warning(f"Gemini AI detection response format not recognized: {gemini_response_text}")
        except Exception as e:
            logger.error(f"Error during Gemini AI detection: {e}", exc_info=True)
    else:
        logger.warning("Gemini model not loaded, skipping Gemini AI detection.")

    # --- Final AI Score Calculation ---
    final_ai_score = round(np.mean(high_confidence_ai_probabilities) if high_confidence_ai_probabilities else 0.0, 1)

    logger.info(f"AI detection (RoBERTa & Gemini) collected {len(high_confidence_ai_probabilities)} high-confidence probabilities. Average AI probability: {final_ai_score:.1f}%")
    return {
        'score': final_ai_score, 
        'highlights': all_highlights
    }


def resolve_overlapping_highlights(highlights: list):
    """
    Resolves overlapping highlights, prioritizing plagiarism over AI,
    and merging same-type overlaps.

    Args:
        highlights (list): A list of highlight dictionaries, each with
        'start', 'end', 'type', 'confidence', 'source'.

    Returns:
        list: A new list of non-overlapping, prioritized highlight dictionaries.
    """
    if not highlights:
        return []

    # 1. Separate highlights by type
    plagiarism_highlights = [h for h in highlights if h['type'] == 'plagiarism']
    ai_highlights = [h for h in highlights if h['type'] == 'ai']

    # 2. Merge overlapping highlights within the same type
    def merge_same_type(hls):
        if not hls: return []
        hls.sort(key=lambda x: x['start']) # Sort by start position
        merged = []
        
        # Iterate and merge
        for current_hl in hls:
            if not merged:
                merged.append(current_hl)
            else:
                last_merged = merged[-1]
                # Check for overlap or immediate succession
                if current_hl['start'] <= last_merged['end']: 
                    # Merge: extend end, take max confidence
                    last_merged['end'] = max(last_merged['end'], current_hl['end'])
                    last_merged['confidence'] = max(last_merged['confidence'], current_hl['confidence'])
                else:
                    # No overlap, add as a new highlight
                    merged.append(current_hl)
        return merged

    merged_plagiarism = merge_same_type(plagiarism_highlights)
    merged_ai = merge_same_type(ai_highlights)

    final_resolved_highlights = []

    # 3. Prioritize plagiarism over AI
    # Iterate through AI highlights. If an AI highlight overlaps with *any* plagiarism highlight,
    # it is either excluded or its non-overlapping parts are kept.
    # For simplicity and strong prioritization: if an AI highlight is covered by *any*
    # plagiarism highlight, it is completely discarded.
    for ai_hl in merged_ai:
        is_covered_by_plag = False
        for plag_hl in merged_plagiarism:
            # Check for overlap
            if ai_hl['start'] < plag_hl['end'] and plag_hl['start'] < ai_hl['end']:
                is_covered_by_plag = True
                break # This AI highlight is covered, discard it
        if not is_covered_by_plag:
            final_resolved_highlights.append(ai_hl)

    # Add all resolved plagiarism highlights
    final_resolved_highlights.extend(merged_plagiarism)

    # Sort the final list by start position for correct rendering order
    final_resolved_highlights.sort(key=lambda x: x['start'])

    return final_resolved_highlights
