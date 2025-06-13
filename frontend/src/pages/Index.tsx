// src/pages/Index.tsx
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import DocumentUpload from '@/components/DocumentUpload';
import ResultsPanel from '@/components/ResultsPanel';
import Footer from '@/components/Footer';
import { DocumentAnalysis } from '@/types/analysis';
import { useToast } from '@/components/ui/use-toast';
import { analyzeDocument } from '@/lib/api';
import ErrorBoundary from '@/components/ErrorBoundary';

const Index = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<DocumentAnalysis | null>(null);
  const { toast } = useToast();
  const navigate = useNavigate();
  const userType = localStorage.getItem('user_type');

  useEffect(() => {
    const isAuthenticated = localStorage.getItem('access_token') !== null;
    if (!isAuthenticated) {
      toast({
        title: "Session Expired",
        description: "Your session has expired. Please log in again.",
        variant: "destructive"
      });
      navigate('/login');
    }
  }, [navigate, toast]);

  const handleAnalyzeDocument = async (file: File, sendReportToOther: boolean, recipientEmail: string | null) => {
    setIsAnalyzing(true);
    setAnalysisResult(null);
    try {
      toast({
        title: "Analysis Started",
        description: `Analyzing ${file.name}...`,
      });

      const result = await analyzeDocument(file, sendReportToOther, recipientEmail);

      setAnalysisResult({
        ...result,
        fileName: file.name,
        analyzedAt: new Date().toISOString()
      });

      toast({
        title: "Analysis Complete",
        description: "Your document has been analyzed successfully. Check your email for a report summary if specified.",
      });
    } catch (error: unknown) {
      console.error("Analysis failed:", error);
      let errorMessage = "Failed to analyze document. Please try again later.";
      if (error && typeof error === "object" && "response" in error && error.response && typeof error.response === "object" && "data" in error.response && error.response.data && typeof error.response.data === "object" && "error" in error.response.data) {
        errorMessage = (error.response.data as { error?: string }).error || errorMessage;
      }
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-teal-50">
      <Navbar />
      <Hero />

      <main className="flex-grow py-12">
        <div className="container mx-auto px-6 space-y-8">
          <div className="max-w-3xl mx-auto">
            {/* Render DocumentUpload only for non-students */}
            {userType !== 'student' && (
              <DocumentUpload
                onAnalyze={handleAnalyzeDocument}
                isAnalyzing={isAnalyzing}
              />
            )}
            {/* Student specific message */}
            {userType === 'student' && (
              <div className="p-8 text-center bg-white rounded-xl shadow-lg">
                <h3 className="text-xl font-medium text-teal-700 mb-3">Document Upload Disabled</h3>
                <p className="text-gray-600">As a student, you cannot upload documents for analysis. You can only view your document metrics. If your
                  supervisor has uploaded a document for you, you can view it here.</p>
                <p className="text-gray-500">Please contact your <a className="text-teal-500" href="/supervisors">supervisor</a> for more information.</p>
                <p className="text-gray-400">If you are a supervisor, please log in with your supervisor account to upload documents.</p>
                <button
                  onClick={() => navigate('/student-dashboard')}
                  className="mt-4 px-6 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors"
                >
                  Go to Student Dashboard
                </button>
              </div>
            )}
          </div>

          {/* Conditional rendering for analysis results or loading spinner */}
          {isAnalyzing && (
            <div className="mt-12 flex justify-center">
              <div className="p-8 text-center">
                <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-teal-500 border-r-transparent mb-4"></div>
                <p className="text-teal-700 text-lg">Analyzing your document...</p>
                <p className="text-gray-500 mt-2">This may take a few moments</p>
              </div>
            </div>
          )}

          {analysisResult && !isAnalyzing && (
            <div className="mt-12 animate-fade-in">
              <ErrorBoundary>
                <ResultsPanel analysis={analysisResult} />
              </ErrorBoundary>
            </div>
          )}

          {!analysisResult && !isAnalyzing && userType !== "student" && (
            <div className="mt-12 p-8 text-center bg-white rounded-xl shadow-lg max-w-3xl mx-auto">
              <h3 className="text-xl font-medium text-teal-700 mb-3">Ready to Analyze Your Document?</h3>
              <p className="text-gray-600">Upload a file above to check for plagiarism and AI-generated content.</p>
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Index;