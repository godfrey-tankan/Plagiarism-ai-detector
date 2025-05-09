---

# 🧠 How the **DocuVerify** System Works

The **DocuVerify** system is a comprehensive web application designed to detect **plagiarism** and **AI-generated content** in uploaded documents.

Below is a detailed explanation of how the system works—from account creation to document analysis—including logic behind detection and the React-based frontend.

---

## 1. 🔐 User Authentication

### 🧾 Account Creation

**Frontend**

* Users sign up via `/signup` in the React frontend.
* A form collects user info (e.g., email, password) and calls:

  ```ts
  registerUser() // in /frontend/src/lib/api.ts
  ```
* Backend responses (success/error) are shown using **toast** notifications.

**Backend**

* Endpoint: `POST /auth/register/`
* Validates user input and creates a user record in the database.

---

### 🔑 Logging In

**Frontend**

* Users log in at `/login`.
* Credentials are sent via:

  ```ts
  loginUser() // in /frontend/src/lib/api.ts
  ```
* Access & refresh tokens are stored in `localStorage`.

**Backend**

* Endpoint: `POST /auth/login/`
* Verifies credentials, issues **JWT tokens** for authentication.

---

### 🧭 Session Management

* Uses Axios **interceptors** in `/frontend/src/lib/api.ts` to handle expired tokens.
* If a token is invalid → redirect to login.

---

## 2. 📤 Document Upload & Analysis

### Uploading a Document

**Frontend**

* Upload via `DocumentUpload` component:

  ```tsx
  /frontend/src/components/DocumentUpload.tsx
  ```
* Supported formats: `.pdf`, `.docx`, `.doc`, `.txt`
* Sent to backend via:

  ```ts
  analyzeDocument() // in /frontend/src/lib/api.ts
  ```

**Backend**

* Endpoint: `POST /analyze/`
* File content extracted using:

  ```py
  extract_text_from_file() // in /documents/utils.py
  ```

---

## 3. 🕵️ Plagiarism Detection

### 🔬 Logic

Detection is handled by:

```py
analyze_text() // in /documents/utils.py
```

**Steps**:

* **Vectorization**: TF-IDF with 5-gram character analysis.
* **Cosine Similarity**: Compared against other documents.
* **Highlighting**: Texts exceeding threshold (e.g., `0.3`) are marked.

### 💻 Frontend

* Results shown in:

  ```tsx
  PlagiarismSection.tsx
  ```
* Visual score shown using:

  ```tsx
  Progress // component
  SourceItem // for match listing
  ```

---

## 4. 🤖 AI Content Detection

### ⚙️ Logic

Detection via:

```py
check_ai_probability() // in /documents/utils.py
```

**Steps**:

* **Chunking**: Splits text into 512-char chunks.
* **Model**: Uses `Hello-SimpleAI/chatgpt-detector-roberta` (Hugging Face).
* **Scoring**: AI confidence score per chunk.
* **Highlighting**: Flags AI-generated segments.

### 💻 Frontend

* Displayed in:

  ```tsx
  AIAnalysisSection.tsx
  ```
* Shows:

  * AI confidence progress bar
  * Highlighted sections
  * Model details

---

## 5. 📊 Results Display

**Frontend**

* Displayed in:

  ```tsx
  ResultsPanel.tsx
  ```

**Features**:

* 📈 **Charts**:

  ```tsx
  PlagiarismChart.tsx
  ```

  Visualizes original, plagiarized, AI-generated distribution.

* ✍️ **Highlights**:

  * Problematic content is visually marked.

---

## 6. ⚛️ React Implementation

### 🧠 State Management

* Uses `useState` and `useEffect`.
* Example:

  ```tsx
  Index.tsx // manages isAnalyzing, analysisResult
  ```

### 🧭 Routing

* React Router handles navigation (`/login`, `/signup`, `/`).
* Uses:

  ```ts
  useNavigate() // to redirect users
  ```

### 🧩 UI Components

* Reusable components: `Button`, `Progress`, `Toast`
* Styled using **Tailwind CSS**

---

## 7. ❗ Error Handling

**Frontend**

* Shows errors using **toast**.
* Catches crashes via:

  ```tsx
  ErrorBoundary.tsx
  ```

**Backend**

* Logs errors and returns appropriate HTTP status codes.

---

## 8. 🚪 Logout

* Handled by:

  ```ts
  logoutUser() // in /frontend/src/lib/api.ts
  ```
* Clears tokens and redirects to `/login`.

---

## ✅ Summary

The **DocuVerify** system combines:

* A **secure backend** for authentication and analysis.
* A **user-friendly frontend** built in React.
* Advanced techniques:

  * 🔍 **TF-IDF + Cosine Similarity** for plagiarism.
  * 🤖 **Transformer models** for AI content detection.

Together, it provides **actionable, transparent, and detailed insights** for document authenticity.

---
