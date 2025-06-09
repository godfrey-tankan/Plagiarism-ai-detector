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
  source: string
  url?: string
  matchPercentage: number
  snippets?: string[]
}

export type DocumentFormat = 'pdf' | 'docx' | 'txt' | 'md';  // Added markdown support


export interface DocumentStats {
  // Backend response fields (snake_case)
  word_count?: number;
  character_count?: number;
  page_count?: number;
  reading_time?: number;
  // Frontend fields (camelCase)
  wordCount?: number;
  characterCount?: number;
  pageCount?: number;
  readingTime?: number;
}

export interface DocumentAnalysis {
  textAnalysis: {
    originalContent: number;
    plagiarizedContent: number;
    aiGeneratedContent: number;
  };
  plagiarismScore: number;
  originalScore: number;
  aiScore: number;
  documentStats: DocumentStats;
  sourcesDetected: SourceMatch[];
  aiMarkers: AIMarker[];
  fileName?: string;
  analyzedAt?: string;
  highlightedText?: string;
  content: string;
  highlights?: Highlight[];
  fileUrl?: string;
}


export interface HighlightPosition {
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Highlight {
  type: 'plagiarism' | 'ai';
  position: HighlightPosition;
  confidence: number;
}

