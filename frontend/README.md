
---

# DocuVerify Core: AI & Plagiarism Detection Engine

## ✨ **Overview**

DocuVerify Core is a powerful backend system engineered to meticulously analyze uploaded documents for originality and potential AI-generated content. By integrating sophisticated text processing, adaptable machine learning models, and a growing internal document comparison database, it offers unparalleled insights into a document's authenticity. This system is ideal for maintaining academic integrity, verifying content uniqueness, and flagging AI-generated material.

## 🚀 **Key Features**

* **Multi-Format Document Extraction:** Seamlessly supports `.pdf`, `.docx`, and `.txt` file formats for broad compatibility.
* **Intelligent Text Cleaning:** Our advanced `clean_text` process normalizes and refines extracted text. It intelligently handles complex formatting issues common in PDFs (like fragmented words or single line breaks), ensuring high accuracy in subsequent analysis.
* **Enhanced Plagiarism Detection (Local Corpus):**
    * Compares the uploaded document against **every other document previously stored within your DocuVerify system**.
    * Employs a robust dual-strategy approach:
        * **Character N-gram Cosine Similarity:** Excels at identifying exact or near-exact phrases and short sentence matches.
        * **Sentence-Level Jaccard Similarity:** Targets conceptual plagiarism by comparing the word sets within sentences, effectively catching rephrased content.
    * **Self-Improving Accuracy:** The more documents uploaded to the system, the larger and more robust the internal corpus becomes. This directly enhances the effectiveness of the TF-IDF vectorizer and Jaccard similarity, leading to more accurate and comprehensive internal plagiarism detection over time.
    * Provides a precise **plagiarism score** and highlights matched sections with confidence levels.
* **External Plagiarism Detection (Web - Free Tier Integration):**
    * **_New Feature:_** Integrates with a free-tier web plagiarism API to extend checks beyond the internal corpus, scanning publicly available content on the internet.
    * **Combined Results:** The findings from the web scan are intelligently merged with the internal plagiarism detection results, providing a holistic view of originality.
    * **Important Note on Free Tiers:** While seeking truly free and robust web plagiarism APIs, users should be aware that such services often come with limitations (e.g., query limits, document size restrictions, or slower response times). Our system is designed to seamlessly incorporate results from a suitable free API, acknowledging these potential limitations.
* **Advanced AI Content Detection:**
    * Leverages a highly capable pre-trained Hugging Face transformer model (`roberta-base-openai-detector`) to accurately assess the likelihood of text being AI-generated.
    * Analyzes document content in intelligently sized, overlapping chunks to maintain contextual understanding.
    * Generates a clear **AI score** and highlights sections exhibiting high AI probability.
* **Originality Score:** A clear metric calculated by subtracting combined plagiarism and AI scores from 100%, indicating the unique human-authored portion of the content.
* **Comprehensive Document Statistics:** Delivers essential document metrics including word count, character count, estimated page count, and approximate reading time.
* **Secure History Tracking:** Maintains a detailed analysis history for each document, accessible only by the owning user, ensuring traceability and accountability.
* **User Authentication & Authorization:** Robust JWT-based authentication ensures secure access and user-specific data isolation.

## ⚙️ **How It Works: A Detailed Breakdown**

The DocuVerify Core processes each uploaded document through a sophisticated, multi-stage pipeline:

### **1. Document Upload & Initial Validation**

Upon document submission, the system initiates critical preliminary checks:
* **File Size Limit:** A 10MB maximum file size is enforced to prevent resource exhaustion.
* **File Format Validation:** Only `.pdf`, `.docx`, and `.txt` formats are accepted, ensuring compatibility for text extraction.

### **2. Text Extraction & Intelligent Cleaning**

The raw textual content is extracted from the uploaded file:
* **PDFs (.pdf):** Utilizes `PyPDF2` with enhanced error handling to manage corrupted or image-only PDFs, ensuring sufficient text is available for analysis.
* **DOCX (.docx):** Employs `python-docx` for reliable text extraction from word documents.
* **TXT (.txt):** Directly reads the file content, handling various encodings.

Post-extraction, the text undergoes a rigorous `clean_text` process:
* Special characters like non-breaking spaces and zero-width spaces are removed.
* Complex newline sequences are intelligently handled: single newlines within sentences are converted to spaces (re-flowing text), while deliberate paragraph breaks (multiple newlines) are preserved.
* Excessive whitespace is normalized, and leading/trailing spaces are trimmed, ensuring consistent and analyzable text.

### **3. Basic Content Validation & Deduplication**

Before in-depth analysis, the system verifies and optimizes:
* **Minimum Length Check:** Ensures the cleaned text meets a minimum threshold of words (10) and characters (200) for meaningful analysis.
* **Content Hashing:** An MD5 hash of the cleaned text is generated, serving as a unique digital fingerprint.
* **Smart Deduplication:**
    * If a document with the exact same content hash already exists in the system, the system efficiently retrieves its previously computed analysis results (scores, highlights, stats).
    * For documents owned by a different user, the existing analysis is returned, preserving computational resources and data privacy for non-admin users.

### **4. Comprehensive Plagiarism Detection**

This crucial phase involves two distinct but complementary checks:

#### **4a. Internal Plagiarism Detection (Local Corpus)**

* **Mechanism:** The input document is compared against every other document residing within the DocuVerify system. This forms a constantly growing, self-improving corpus of knowledge, meaning the plagiarism detection gets better everytime a new document is uploaded for analysis.
* **Character N-gram Cosine Similarity:**
    * `TfidfVectorizer` transforms documents into numerical representations based on overlapping sequences of characters (n-grams, from 5 to 9 characters long).
    * Sliding windows of the input text are compared via `cosine_similarity` to n-grams across the local document system. High similarity scores indicate potential direct matches.
* **Sentence-Level Jaccard Similarity:**
    * `nltk.sent_tokenize` breaks down texts into sentences.
    * Word sets within the input document's sentences are compared to word sets in sentences from the entire local corpus using Jaccard similarity. This effectively identifies instances where the core vocabulary and meaning of a sentence have been retained, even if words are rephrased.
* **Benefit of Growth:** As more documents are uploaded, the internal corpus expands, making the TF-IDF vectorizer and Jaccard similarity checks increasingly robust and accurate in identifying internal plagiarism patterns.
* **Highlighting:** Detected matches from both n-gram and sentence-level analysis contribute to the "plagiarism" highlights.

#### **4b. External Plagiarism Detection (Web - Free Tier)**

* **Integration Point:** Following the internal checks, the system reaches out to a configured **free-tier web plagiarism API**. The cleaned text (or relevant snippets) is sent to this external service for a scan against publicly available content on the internet.
* **Result Merging:** The plagiarism results (scores, detected similarities, and corresponding highlights) returned by the external web API are intelligently combined with the findings from the internal plagiarism detection. This ensures a comprehensive final plagiarism score and set of highlights, representing both internal and external sources of detected similarities.

### **5. Advanced AI Content Detection**

The system employs a sophisticated approach to identify AI-generated text:
* **Model Loading:** The `roberta-base-openai-detector` transformer model (or a dynamically chosen alternative) is loaded just once per process to maximize efficiency. It supports both CPU (default fallback) and GPU execution.
* **Intelligent Text Chunking:** The document's text is segmented into overlapping chunks (e.g., 512 characters with 100 characters overlap). This strategy preserves context and allows for granular analysis across the document.
* **Probability Prediction:** Each text chunk is analyzed by the loaded transformer model, which predicts a `label` (e.g., 'real'/'human' or 'machine'/'AI') and a `score` (its confidence in that label).
* **AI Probability Calculation:** Based on the model's output, a precise AI probability is calculated for each chunk. For instance, if the model labels a chunk as 'human' with 90% confidence, the AI probability is 10%.
* **Highlighting & Scoring:** Chunks exceeding a set AI probability threshold (e.g., 70%) are marked as "AI" highlights. The overall AI score for the document is an aggregated average of the individual chunk probabilities, meticulously capped to maintain logical score distribution.

### **6. Originality Score & Document Statistics**

* **Originality Calculation:** The "Original Score" is derived as `100 - (Plagiarism Score + AI Score)`, ensuring that all three scores sum up to 100% and provide a clear overall picture of the document's uniqueness.
* **Document Statistics:** Critical metrics are computed for quick document overview:
    * **Word Count:** Total number of words.
    * **Character Count:** Total characters.
    * **Page Count:** An estimated number of pages based on character density (approx. 1800 characters per page).
    * **Reading Time:** An estimated reading duration using average reading speeds.

### **7. Data Persistence & History**

Finally, all analysis results are securely stored and organized:
* **Document Model:** The extracted content, its unique content hash, calculated plagiarism and AI scores, combined highlights, file reference, and all statistics are saved or updated within the `Document` model.
* **Document History Model:** A detailed, immutable record of each analysis run (including content snapshot, scores, and highlights at that time) is created in the `DocumentHistory` model, providing a traceable audit trail.

## 🛠️ **Installation & Setup**

(This section assumes this is a Python/Django project)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/godfrey-tankan/Plagiarism-ai-detector.git .
    cd backend 
    ```
2.  **Create Virtual Environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```
3.  **Install Core Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Django, djangorestframework, PyPDF2, python-docx, textstat, transformers, torch, scikit-learn, nltk, numpy, Pillow (for file uploads), requests (for web API calls)
    ```
4.  **Download NLTK Data:**
    ```bash
    pip install google-generativeai
    python -c "import nltk; nltk.download('punkt')"
    python -c "import nltk; nltk.download('punkt_tab')" # Essential for advanced sentence tokenization
    ```
5.  **Configure `settings.py`:**
    * Set up your `DATABASE` settings (e.g., PostgreSQL, SQLite).
    * Configure `MEDIA_ROOT` and `MEDIA_URL` for handling file uploads.
    * Configure `LOGGING` to capture `DEBUG` level messages for detailed insights (`documents` and `console` loggers).
    * Add `rest_framework_simplejwt` to `INSTALLED_APPS` and configure its settings.
    * **_New:_** Add a setting for your external plagiarism API key/endpoint (e.g., `EXTERNAL_PLAGIARISM_API_URL`, `EXTERNAL_PLAGIARISM_API_KEY`). **You will need to sign up for a free-tier API and configure this.**
6.  **Run Database Migrations:**
    ```bash
    python manage.py makemigrations documents # If you made model changes
    python manage.py migrate
    ```
7.  **Create Superuser (Optional):**
    ```bash
    python manage.py createsuperuser
    ```
8.  **Start the Development Server:**
    ```bash
    python manage.py runserver
    ```

## 📈 **Usage (API Endpoints)**

The core analysis functionalities are exposed via well-defined REST API endpoints.

### **`POST /api/analyze/`**

Initiates the document analysis process.

* **Authentication:** Requires a valid JWT Bearer Token.
* **Request Body (`multipart/form-data`):**
    * `document`: (File) The document file to be analyzed (PDF, DOCX, TXT).
* **Example Success Response (200 OK):**
    ```json
    {
        "id": 123,
        "fileUrl": "/media/documents/uploaded_doc.pdf",
        "plagiarismScore": 18.5, # Combined internal + external
        "aiScore": 22.0,
        "originalScore": 59.5,
        "documentStats": {
            "wordCount": 750,
            "characterCount": 4200,
            "pageCount": 3,
            "readingTime": 4
        },
        "highlights": [
            {
                "type": "plagiarism",
                "position": {"page": 1, "x": 10.0, "y": 0, "width": 5.2, "height": 2},
                "confidence": 88.5
            },
            {
                "type": "plagiarism",
                "position": {"page": 1, "x": 60.0, "y": 0, "width": 8.0, "height": 2},
                "confidence": 75.0,
                "source": "web" # Indicate source if from external API
            },
            {
                "type": "ai",
                "position": {"page": 1, "x": 40.5, "y": 0, "width": 10.0, "height": 2},
                "confidence": 78.1
            }
        ],
        "content": "This is the full extracted and cleaned content of the document..."
    }
    ```

### **`GET /api/documents/`**

Retrieves a list of all documents uploaded by the authenticated user.

### **`GET /api/documents/{id}/`**

Fetches detailed analysis results for a specific document by its ID.

### **`GET /api/history/`**

Accesses the chronological analysis history for all documents associated with the authenticated user.



**Next Steps for Implementation:**


2.  **Integrate the API in `utils.py`:**
    * Create a new function, e.g., `check_web_plagiarism(text)`.
    * This function will make an HTTP request to your chosen API with the document text.
    * Parse the API's response to extract plagiarism scores and detected matches/highlights.
    * Handle API rate limits, errors, and potential empty responses gracefully.
    * Return a dictionary similar to `analyze_text`'s output, but specifically for web results (e.g., `{'score': X, 'highlights': Y, 'source': 'web'}`).
3.  **Combine Results in `views.py`:**
    * In `AnalyzeDocumentView.post`, after calling `analyze_text` (internal plagiarism), call your new `check_web_plagiarism` function.
    * Merge the highlights from both internal and external plagiarism checks.
    * Decide how to combine the scores. A simple approach is to take the maximum of the two plagiarism scores or a weighted average, but typically, you'd calculate the percentage of unique characters covered by *all* highlights combined, which I already implemented as the scoring method in `analyze_text`. You'd just add the web highlights to the total `all_highlights` before recalculating the `final_score`.
    * You might want to add a `source` field to highlights (e.g., `'internal'`, `'web'`) for better clarity in the frontend.