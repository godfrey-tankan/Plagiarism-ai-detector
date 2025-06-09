import { useState, useEffect, useCallback } from 'react';
import { DocumentAnalysis, Highlight } from '@/types/analysis'; // Ensure Highlight is imported
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
        position: 'fixed', // Use fixed to position relative to viewport
        left: position.x + 10, // Offset from cursor
        top: position.y + 10,
        zIndex: 100,
        pointerEvents: 'none', // Allow clicks/hovers to pass through
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

  // Tooltip state
  const [tooltipContent, setTooltipContent] = useState('');
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  const safeAnalysis = analysis || {
    plagiarismScore: 0,
    aiScore: 0,
    originalScore: 0,
    documentStats: {
      wordCount: 0,
      characterCount: 0,
      pageCount: 0,
      readingTime: 0
    },
    highlights: [],
    fileUrl: '',
    content: ''
  };

  const fileExtension = safeAnalysis.fileUrl?.split('.').pop()?.toLowerCase();
  const isWordDoc = ['doc', 'docx'].includes(fileExtension || '');

  const chartData = [
    {
      name: 'Original',
      value: (100 - (safeAnalysis.plagiarismScore + safeAnalysis.aiScore)).toFixed(1),
      fill: 'oklch(59.6% 0.145 163.225)'
    },
    {
      name: 'Plagiarized',
      value: safeAnalysis.plagiarismScore.toFixed(1),
      fill: '#EF4444'
    },
    {
      name: 'AI Generated',
      value: safeAnalysis.aiScore.toFixed(1),
      fill: 'oklch(60% 0.15 270)'
    }
  ];

  const handleMouseEnterHighlight = useCallback((e: React.MouseEvent, highlight: Highlight) => {
    setTooltipContent(`${highlight.type === 'ai' ? 'AI Generated' : 'Plagiarized'}: ${highlight.confidence.toFixed(1)}% Confidence`);
    setTooltipPosition({ x: e.clientX, y: e.clientY });
    setTooltipVisible(true);
  }, []);

  const handleMouseLeaveHighlight = useCallback(() => {
    setTooltipVisible(false);
  }, []);

  // Function to render text with highlights and tooltips
  const renderHighlightedText = useCallback((text: string, highlights: DocumentAnalysis['highlights']) => {
    if (!text) return null;

    const sortedHighlights = highlights
      .map(h => ({
        ...h,
        charStart: Math.floor(h.position.x / 100 * text.length),
        charEnd: Math.floor((h.position.x + h.position.width) / 100 * text.length)
      }))
      .sort((a, b) => a.charStart - b.charStart);

    const elements: JSX.Element[] = [];
    let lastIndex = 0;

    sortedHighlights.forEach((highlight, index) => {
      // Clamp highlight indices to text length
      const charStart = Math.max(0, Math.min(text.length, highlight.charStart));
      const charEnd = Math.max(0, Math.min(text.length, highlight.charEnd));

      // Add text before the current highlight
      if (charStart > lastIndex) {
        elements.push(
          <span key={`text-pre-${index}`}>
            {text.substring(lastIndex, charStart)}
          </span>
        );
      }

      // Add the highlighted text with mouse handlers
      const highlightText = text.substring(charStart, charEnd);
      const highlightClass = highlight.type === 'plagiarism'
        ? 'bg-red-200'
        : highlight.type === 'ai'
          ? 'bg-yellow-200'
          : '';

      elements.push(
        <span
          key={`highlight-${index}`}
          className={`${highlightClass} relative cursor-help`} // Add cursor-help for visual cue
          onMouseEnter={(e) => handleMouseEnterHighlight(e, highlight)}
          onMouseLeave={handleMouseLeaveHighlight}
        >
          {highlightText}
        </span>
      );

      lastIndex = charEnd;
    });

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
  }, [handleMouseEnterHighlight, handleMouseLeaveHighlight]); // Add dependencies for useCallback


  const loadDocumentContent = async () => {
    if (safeAnalysis.content) {
      setLoadingDoc(false);
      return;
    }
    if (!safeAnalysis.fileUrl) return;

    setLoadingDoc(true);
    setDocError(null);

    try {
      if (isWordDoc) {
        const response = await fetch(safeAnalysis.fileUrl);
        if (!response.ok) throw new Error('Failed to fetch document');
        const arrayBuffer = await response.arrayBuffer();
        const result = await mammoth.convertToHtml({ arrayBuffer });
      }
    } catch (error) {
      console.error('Document load error:', error);
      setDocError(error instanceof Error ? error.message : 'Document load failed');
    } finally {
      setLoadingDoc(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    if (activeTab === 'text' && (safeAnalysis.fileUrl || safeAnalysis.content)) {
      loadDocumentContent();
    }
    return () => {
      isMounted = false;
    };
  }, [activeTab, safeAnalysis.fileUrl, safeAnalysis.content, loadDocumentContent]);


  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-8">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-2 w-full md:w-[400px]">
          <TabsTrigger value="stats">Statistics View</TabsTrigger>
          <TabsTrigger value="text">Document View</TabsTrigger>
        </TabsList>

        <TabsContent value="stats">
          {/* Statistics Tab content remains the same */}
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <StatCard
                title="Original Content"
                value={`${(safeAnalysis.originalScore).toFixed(2)}%`}
                colorClass="text-green-600"
                description="Unique content percentage"
              />
              <StatCard
                title="Plagiarized Content"
                value={`${safeAnalysis.plagiarismScore.toFixed(2)}%`}
                colorClass="text-red-600"
                description="Matched with existing sources"
              />
              <StatCard
                title="AI Generated"
                value={`${safeAnalysis.aiScore.toFixed(2)}%`}
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
                    { name: 'Plagiarism', score: safeAnalysis.plagiarismScore },
                    { name: 'AI', score: safeAnalysis.aiScore }
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
                value={safeAnalysis.documentStats.wordCount.toLocaleString()}
                description="Total words in document"
              />
              <StatCard
                title="Characters"
                value={safeAnalysis.documentStats.characterCount.toLocaleString()}
                description="Including spaces"
              />
              <StatCard
                title="Pages"
                value={safeAnalysis.documentStats.pageCount.toString()}
                description="Approximate count"
              />
              <StatCard
                title="Reading Time"
                value={`${safeAnalysis.documentStats.readingTime}m`}
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
            {!loadingDoc && !docError && safeAnalysis.content ? (
              <div
                className="border rounded-lg min-h-[400px] max-h-[80vh] overflow-y-auto" // Added max-h and overflow-y-auto
                style={{ resize: 'vertical' }} // Optional: Allow user to resize height
              >
                {renderHighlightedText(safeAnalysis.content, safeAnalysis.highlights)}
              </div>
            ) : (
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-gray-500">No document content available for display or highlighting.</p>
                <p className="text-gray-500 mt-2">
                  Please ensure the document file is valid and the backend returns its `content`.
                </p>
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