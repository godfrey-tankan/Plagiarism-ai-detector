// src/components/ResultsPanel.tsx
import { useState, useEffect, useCallback } from 'react';
// Import BackendHighlight instead of your local Highlight
import { DocumentAnalysis, BackendHighlight } from '@/types/analysis';
import * as mammoth from 'mammoth';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger
} from '@/components/ui/tabs';

// Tooltip component (can be a separate file, but inline for example)
const TextHighlightTooltip = ({
  content,
  isVisible,
  position
}: {
  content: string;
  isVisible: boolean;
  position: { x: number; y: number };
}) => {
  if (!isVisible) return null;

  return (
    <div
      style={{
        position: 'fixed',
        left: position.x + 10,
        top: position.y + 10,
        zIndex: 100,
        pointerEvents: 'none',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '6px 10px',
        borderRadius: '4px',
        fontSize: '0.85rem',
        maxWidth: '200px',
        wordWrap: 'break-word',
      }}
    >
      {content}
    </div>
  );
};


const ResultsPanel = ({ analysis }: { analysis?: DocumentAnalysis }) => {
  const [activeTab, setActiveTab] = useState('stats');
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);
  const [displayContent, setDisplayContent] = useState<string>(''); // State for processed content

  // Tooltip state
  const [tooltipContent, setTooltipContent] = useState('');
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  const safeAnalysis = analysis || {
    plagiarism_score: 0,
    ai_score: 0,
    originality_score: 0,
    word_count: 0,
    character_count: 0,
    page_count: 0,
    reading_time: 0,
    highlights: [],
    file: '', // Backend provides file URL as 'file'
    content: '',
    document_code: '', // Ensure these are present for safety
    user: null, id: 0, title: '', content_hash: '', created_at: '', recipient_email: null
  };

  // Backend properties are snake_case. Map them to camelCase for display if needed.
  const documentStats = {
    wordCount: safeAnalysis.word_count || 0,
    characterCount: safeAnalysis.character_count || 0,
    pageCount: safeAnalysis.page_count || 0,
    readingTime: safeAnalysis.reading_time || 0,
  };

  const fileExtension = safeAnalysis.file?.split('.').pop()?.toLowerCase() || ''; // Use safeAnalysis.file
  const isWordDoc = ['doc', 'docx'].includes(fileExtension);
  const isPdfDoc = fileExtension === 'pdf'; // Add PDF check

  const chartData = [
    {
      name: 'Original',
      value: (100 - (safeAnalysis.plagiarism_score + safeAnalysis.ai_score)).toFixed(1),
      fill: 'oklch(59.6% 0.145 163.225)'
    },
    {
      name: 'Plagiarized',
      value: safeAnalysis.plagiarism_score.toFixed(1),
      fill: '#EF4444'
    },
    {
      name: 'AI Generated',
      value: safeAnalysis.ai_score.toFixed(1),
      fill: 'oklch(60% 0.15 270)'
    }
  ];

  const handleMouseEnterHighlight = useCallback((e: React.MouseEvent, highlight: BackendHighlight) => {
    setTooltipContent(`${highlight.type === 'ai' ? 'AI Generated' : 'Plagiarized'}: ${highlight.confidence.toFixed(1)}% Confidence`);
    setTooltipPosition({ x: e.clientX, y: e.clientY });
    setTooltipVisible(true);
  }, []);

  const handleMouseLeaveHighlight = useCallback(() => {
    setTooltipVisible(false);
  }, []);

  // Function to render text with highlights based on start/end indices
  const renderHighlightedText = useCallback((text: string, highlights: BackendHighlight[]) => {
    if (!text || !highlights || highlights.length === 0) return <span>{text}</span>;

    // Sort highlights by their start position to ensure correct rendering order
    const sortedHighlights = [...highlights].sort((a, b) => a.start - b.start);

    const elements: JSX.Element[] = [];
    let lastIndex = 0;

    sortedHighlights.forEach((highlight, index) => {
      // Add text before the current highlight
      if (highlight.start > lastIndex) {
        elements.push(
          <span key={`text-pre-${index}`}>
            {text.substring(lastIndex, highlight.start)}
          </span>
        );
      }

      // Add the highlighted text with mouse handlers
      const highlightText = text.substring(highlight.start, highlight.end);
      const highlightClass = highlight.type === 'plagiarism'
        ? 'bg-red-200'
        : highlight.type === 'ai'
          ? 'bg-yellow-200'
          : '';

      elements.push(
        <span
          key={`highlight-${index}`}
          className={`${highlightClass} relative cursor-help`}
          onMouseEnter={(e) => handleMouseEnterHighlight(e, highlight)}
          onMouseLeave={handleMouseLeaveHighlight}
        >
          {highlightText}
        </span>
      );

      lastIndex = highlight.end;
    });

    // Add any remaining text after the last highlight
    if (lastIndex < text.length) {
      elements.push(
        <span key={`text-post-${sortedHighlights.length}`}>
          {text.substring(lastIndex)}
        </span>
      );
    }

    return (
      <pre className="whitespace-pre-wrap font-sans text-sm p-4 leading-relaxed">
        {elements}
      </pre>
    );
  }, [handleMouseEnterHighlight, handleMouseLeaveHighlight]);


  const loadDocumentContent = useCallback(async () => {
    // If content is already available (from backend directly), use it
    if (safeAnalysis.content) {
      setDisplayContent(safeAnalysis.content);
      setLoadingDoc(false);
      return;
    }
    // If no content and no file URL, nothing to do
    if (!safeAnalysis.file) { // Use safeAnalysis.file
      setDocError('No document content or file URL available.');
      setLoadingDoc(false);
      return;
    }

    setLoadingDoc(true);
    setDocError(null);

    try {
      if (isWordDoc) {
        // Fetch .docx file and convert to HTML
        const response = await fetch(safeAnalysis.file);
        if (!response.ok) throw new Error('Failed to fetch document');
        const arrayBuffer = await response.arrayBuffer();
        const result = await mammoth.convertToHtml({ arrayBuffer });
        setDisplayContent(result.value); // mammoth returns HTML in .value
      } else if (isPdfDoc) {
        // For PDF, you might link directly or use a PDF viewer library.
        // For now, we'll just show a message. Full PDF text extraction
        // is typically done on the backend.
        setDocError('PDF preview not directly supported here. Please download the file to view.');
        setDisplayContent(''); // Clear any previous content
      } else { // Assume it's .txt or other text-based format
        const response = await fetch(safeAnalysis.file);
        if (!response.ok) throw new Error('Failed to fetch document');
        const text = await response.text();
        setDisplayContent(text);
      }
    } catch (error) {
      console.error('Document load error:', error);
      setDocError(error instanceof Error ? error.message : 'Document load failed');
      setDisplayContent('');
    } finally {
      setLoadingDoc(false);
    }
  }, [safeAnalysis.content, safeAnalysis.file, isWordDoc, isPdfDoc]);


  useEffect(() => {
    if (activeTab === 'text') {
      loadDocumentContent();
    }
  }, [activeTab, loadDocumentContent]);


  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-8">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-2 w-full md:w-[400px]">
          <TabsTrigger value="stats">Statistics View</TabsTrigger>
          <TabsTrigger value="text">Document View</TabsTrigger>
        </TabsList>

        <TabsContent value="stats">
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <StatCard
                title="Original Content"
                value={`${(safeAnalysis.originality_score).toFixed(2)}%`}
                colorClass="text-green-600"
                description="Unique content percentage"
              />
              <StatCard
                title="Plagiarized Content"
                value={`${safeAnalysis.plagiarism_score.toFixed(2)}%`}
                colorClass="text-red-600"
                description="Matched with existing sources"
              />
              <StatCard
                title="AI Generated"
                value={`${safeAnalysis.ai_score.toFixed(2)}%`}
                colorClass="text-yellow-600"
                description="Probability of AI generation"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="h-64">
                <h3 className="text-xl font-semibold mb-4">Content Distribution</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="value">
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="h-64">
                <h3 className="text-xl font-semibold mb-4">Risk Scores</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[
                    { name: 'Plagiarism', score: safeAnalysis.plagiarism_score },
                    { name: 'AI', score: safeAnalysis.ai_score }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="score" fill="#EF4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                title="Word Count"
                value={documentStats.wordCount.toLocaleString()}
                description="Total words in document"
              />
              <StatCard
                title="Characters"
                value={documentStats.characterCount.toLocaleString()}
                description="Including spaces"
              />
              <StatCard
                title="Pages"
                value={documentStats.pageCount.toString()}
                description="Approximate count"
              />
              <StatCard
                title="Reading Time"
                value={`${documentStats.readingTime}m`}
                description="Average reading time"
              />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="text">
          <div className="space-y-4">
            {loadingDoc && <p className="text-gray-500">Loading document content...</p>}
            {docError && (
              <div className="text-red-500 p-4 bg-red-50 rounded-lg">
                {docError}
              </div>
            )}
            {!loadingDoc && !docError && displayContent ? (
              <div
                className="border rounded-lg min-h-[400px] max-h-[80vh] overflow-y-auto"
                style={{ resize: 'vertical' }}
              >
                {renderHighlightedText(displayContent, safeAnalysis.highlights)}
              </div>
            ) : (
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-gray-500">No document content available for display or highlighting.</p>
                <p className="text-gray-500 mt-2">
                  Please ensure the document file is valid and the backend returns its `content` or a `file` URL.
                </p>
                {safeAnalysis.file && (
                  <a href={safeAnalysis.file} target="_blank" rel="noopener noreferrer" className="text-teal-600 hover:underline">
                    Download original file
                  </a>
                )}
              </div>
            )}
          </div>
          {/* Tooltip component */}
          <TextHighlightTooltip
            content={tooltipContent}
            isVisible={tooltipVisible}
            position={tooltipPosition}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
};

const StatCard = ({ title, value, description, colorClass = "text-gray-800" }: {
  title: string;
  value: string;
  description: string;
  colorClass?: string;
}) => (
  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
    <h3 className="text-sm font-medium text-gray-500">{title}</h3>
    <p className={`text-2xl font-bold mt-2 ${colorClass}`}>{value}</p>
    <p className="text-xs text-gray-500 mt-1">{description}</p>
  </div>
);

export default ResultsPanel;