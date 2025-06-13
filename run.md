
### 🧠 **Project Summary & Technical Overview**

> The system I developed is an **AI-based plagiarism and content authenticity detection platform** that allows users to upload academic documents for analysis. The main goal is to detect not only **plagiarized content** but also **AI-generated text**, using both traditional NLP techniques and machine learning models. It is built with a full-stack architecture using **Django REST Framework (DRF)** for API development, **React** for the frontend, and **PostgreSQL** as the primary database.

---

### 🔍 **Algorithms & Techniques Used**

#### **1. Plagiarism Detection**

> I used two main methods for internal plagiarism detection:

* **Character N-Gram TF-IDF + Cosine Similarity**

  * **Efficiency:** Fast and scalable using sparse matrices.
  * **Strengths:** Excellent for near-exact matches or paraphrased phrases.
  * **Weaknesses:** Cannot handle deeply reworded or semantically altered text.

* **Jaccard Similarity on Tokenized Sentences**

  * **Efficiency:** Linear with sentence count but more memory-intensive.
  * **Strengths:** Good for reworded sentence comparisons.
  * **Weaknesses:** Struggles with short documents and partial overlaps.

---

#### **2. AI-Generated Text Detection**

> For AI detection, I combined two approaches to improve confidence and reduce false positives:

* **RoBERTa (Transformer Model) via Hugging Face Pipeline**

  * **Model Used:** `roberta-base-openai-detector`
  * **Efficiency:** Moderate, especially on CPU. Chunked with overlapping windows.
  * **Strengths:** Capable of detecting AI-style patterns in short chunks.
  * **Weaknesses:** Chunk-level detection might miss broader context.

* **Gemini (Google’s Generative Model) – API-Based**

  * **Usage:** Provides an overall AI probability score and reasoning.
  * **Efficiency:** Fast but depends on external API calls and limits.
  * **Strengths:** Human-like reasoning and contextual understanding.
  * **Weaknesses:** Rate-limited and may fail under connectivity/API issues.

> These models complement each other. Gemini assesses global structure, while RoBERTa captures granular AI indicators. The final AI score is an average of high-confidence results.

---

### 🔐 **Security and Architecture**

* **Authentication:** I implemented **JWT (JSON Web Tokens)** for secure, stateless authentication. Tokens are used to protect private endpoints such as document uploads and user history.
* **Authorization:** Documents are scoped to the authenticated user. Publicly shared reports use a unique `document_code` as access control.
* **Frameworks:**

  * **Django REST Framework (DRF):** For API views, serialization, and permissions.
  * **Celery (planned):** For asynchronous email delivery and processing heavy tasks.
* **Data Validation:** Each file is hashed using SHA-256 for deduplication and integrity checking.

---

### 📊 **Strengths & Limitations**

| **Aspect**           | **Strengths**                                                       | **Limitations / Mitigations**                                      |
| -------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Plagiarism Detection | Good coverage of literal and partially paraphrased content          | Struggles with semantic plagiarism across domains                  |
| AI Detection         | High accuracy when combining RoBERTa & Gemini                       | Gemini reliance introduces dependency on API limits                |
| Performance          | Fast analysis (<10s for medium-length docs), scalable with chunking | GPU recommended for production RoBERTa inference                   |
| Security             | JWT authentication, hashed files, audit logs                        | Email-based sharing can be abused without throttling (future work) |
| Usability            | Clean RESTful APIs, frontend integration, email reports             | No current support for bulk uploads (planned)                      |

---

### ✅ **Conclusion**

> The system meets the objectives of detecting both **plagiarism** and **AI-generated text** with high confidence. It leverages modern ML models (RoBERTa, Gemini), secure backend architecture (DRF, JWT), and robust document parsing logic to deliver a practical, ethical tool for academic use.

---
