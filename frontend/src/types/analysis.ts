// src/types/analysis.ts

export interface TextAnalysisResult {
  originalContent: number;       // Percentage of original content
  plagiarizedContent: number;    // Percentage of plagiarized content
  aiGeneratedContent: number;    // Percentage of AI-generated content
}

export interface AIMarker {
  type: string;                  // AI model type (e.g., 'GPT-4', 'Bard', 'Claude-2')
  confidence: number;            // Detection confidence percentage
  sections: string[];            // Sections of text flagged as AI-generated
}

export interface SourceMatch {
  source: string;
  url?: string;
  matchPercentage: number;
  snippets?: string[];
}

export type DocumentFormat = 'pdf' | 'docx' | 'txt' | 'md';

export interface DocumentStats {
  word_count: number;     // Backend sends snake_case
  character_count: number;
  page_count: number;
  reading_time: number;
  // If you map these to camelCase in frontend, keep original snake_case for direct backend mapping
}

// Interface for a single highlight returned from backend
export interface BackendHighlight {
  start: number; // Start character index
  end: number;   // End character index
  type: 'plagiarism' | 'ai';
  confidence: number;
  text?: string; // Optional: The actual text of the highlight
}

// Interface for DocumentHistory records
export interface DocumentHistoryRecord { // <--- ENSURE THIS IS EXPORTED
  id: number;
  document: number; // ID of the parent document
  content: string;
  plagiarism_score: number;
  ai_score: number;
  originality_score: number;
  highlights: BackendHighlight[]; // History records also have highlights
  created_at: string;
  word_count: number;
  character_count: number;
  page_count: number;
  reading_time: number;
}


// Main DocumentAnalysis interface
export interface DocumentAnalysis { // <--- ENSURE THIS IS EXPORTED
  id: number;
  document_code: string; // New field for public lookup
  user: number | null; // User ID or null
  title: string;
  file: string; // URL to the uploaded file
  content: string;
  content_hash: string;

  plagiarism_score: number;
  ai_score: number;
  originality_score: number; // New field
  highlights: BackendHighlight[]; // Changed to BackendHighlight array

  created_at: string;
  word_count: number;
  character_count: number;
  page_count: number;
  reading_time: number;

  recipient_email: string | null; // New field

  history_records?: DocumentHistoryRecord[]; // New field for history

  // Frontend-specific fields, if you add them to the object after API call
  fileName?: string;
  analyzedAt?: string;
  // If you want to transform stats to camelCase:
  documentStats?: {
    wordCount: number;
    characterCount: number;
    pageCount: number;
    readingTime: number;
  };
}

// You can keep these if you still use them elsewhere, but BackendHighlight is the new standard
// export interface HighlightPosition {
//   page: number;
//   x: number;
//   y: number;
//   width: number;
//   height: number;
// }

// export interface Highlight {
//   type: 'plagiarism' | 'ai';
//   position: HighlightPosition;
//   confidence: number;
// }