import { useState, useEffect } from 'react';
import { DocumentAnalysis } from '@/types/analysis';
import * as mammoth from 'mammoth';
import { Viewer, SpecialZoomLevel, PdfJs } from '@react-pdf-viewer/core';
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout';
import { GlobalWorkerOptions } from 'pdfjs-dist';
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

// GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`;
GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`;

// Import PDF viewer styles
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';

const ResultsPanel = ({ analysis }: { analysis?: DocumentAnalysis }) => {
  const [activeTab, setActiveTab] = useState('stats');
  const [docContent, setDocContent] = useState<string>('');
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  // Default values for safe analysis
  const safeAnalysis = analysis || {
    plagiarismScore: 0,
    aiScore: 0,
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
  const normalizeUrl = (url: string) => {
    // Handle relative paths from Django
    if (url.startsWith('/media/')) {
      return `${window.location.origin}${url}`;
    }
    return url;
  };

  // File type detection
  const fileExtension = safeAnalysis.fileUrl?.split('.').pop()?.toLowerCase();
  const isPdf = fileExtension === 'pdf';
  const isWordDoc = ['doc', 'docx'].includes(fileExtension || '');
  const chartData = [
    {
      name: 'Original',
      value: 100 - (safeAnalysis.plagiarismScore + safeAnalysis.aiScore),
      fill: 'oklch(59.6% 0.145 163.225)'
    },
    {
      name: 'Plagiarized',
      value: safeAnalysis.plagiarismScore,
      fill: '#EF4444'

    },
    {
      name: 'AI Generated',
      value: safeAnalysis.aiScore,
      fill: 'oklch(60% 0.15 270)'
    }
  ];


  // Document loading handler
  const loadDocumentContent = async () => {
    if (!safeAnalysis.fileUrl) return;

    setLoadingDoc(true);
    setDocError(null);
    setPdfError(null);

    try {
      if (isWordDoc) {
        const response = await fetch(safeAnalysis.fileUrl);
        if (!response.ok) throw new Error('Failed to fetch document');
        const arrayBuffer = await response.arrayBuffer();
        const result = await mammoth.convertToHtml({ arrayBuffer });
        setDocContent(result.value);
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
    if (activeTab === 'text' && safeAnalysis.fileUrl) {
      loadDocumentContent();
    }
    return () => {
      isMounted = false;
    };
  }, [activeTab, safeAnalysis.fileUrl]);

  const handlePdfError = (error: Error) => {
    console.error('PDF Error:', error);
    setPdfError(error.message);
  };

  const validatePdfUrl = (url: string) => {
    try {
      const parsed = new URL(url);
      return parsed.protocol.startsWith('http');
    } catch {
      return false;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-8">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-2 w-full md:w-[400px]">
          <TabsTrigger value="stats">Statistics View</TabsTrigger>
          <TabsTrigger value="text">Document View</TabsTrigger>
        </TabsList>


        {/* Statistics Tab */}
        <TabsContent value="stats">
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <StatCard
                title="Original Content"
                value={`${(100 - (safeAnalysis.plagiarismScore + safeAnalysis.aiScore)).toFixed(2)}%`}
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
                    <YAxis />
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
                    <YAxis />
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

        {/* Document View Tab */}
        {/* Document View Tab */}
        <TabsContent value="text">
          <div className="space-y-4">
            {safeAnalysis.fileUrl ? (
              <>
                {isPdf && (
                  <div className="h-[800px] relative">
                    {validatePdfUrl(safeAnalysis.fileUrl) && (
                      <Viewer
                        fileUrl={normalizeUrl(safeAnalysis.fileUrl)}
                        plugins={[defaultLayoutPluginInstance]}
                        defaultScale={SpecialZoomLevel.PageFit}
                        httpHeaders={{
                          'Accept-Ranges': 'bytes',
                          'Cache-Control': 'no-cache',
                          'Access-Control-Allow-Origin': '*' // Explicit CORS header
                        }}
                        renderLoader={() => (
                          <div className="p-4 bg-blue-50 text-blue-600">
                            Loading PDF document...
                          </div>
                        )}
                        renderError={(error) => {
                          console.error('PDF Render Error:', error);
                          return (
                            <div className="p-4 bg-red-50 text-red-600">
                              PDF Error: {error.message}. Please check:
                              <ul className="list-disc pl-6 mt-2">
                                <li>File URL is accessible</li>
                                <li>No browser extensions blocking PDFs</li>
                                <li>Valid PDF format</li>
                              </ul>
                            </div>
                          );
                        }}
                      />
                    )}
                  </div>
                )}

                {isWordDoc && (
                  <div className="border rounded-lg p-4 min-h-[800px]">
                    {loadingDoc && <p className="text-gray-500">Loading document...</p>}
                    {docError && (
                      <div className="text-red-500 p-4 bg-red-50 rounded-lg">
                        {docError}
                      </div>
                    )}
                    {!loadingDoc && !docError && (
                      <div
                        className="prose max-w-none"
                        dangerouslySetInnerHTML={{ __html: docContent }}
                      />
                    )}
                    {!docContent && !loadingDoc && !docError && (
                      <iframe
                        src={`https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(safeAnalysis.fileUrl)}`}
                        width="100%"
                        height="800px"
                        frameBorder="0"
                        className="border rounded-lg"
                        title="Document preview"
                      />
                    )}
                  </div>
                )}

                {!isPdf && !isWordDoc && (
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-gray-500">
                      Preview not available for .{fileExtension} files
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-gray-500">No document preview available</p>
              </div>
            )}
          </div>
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