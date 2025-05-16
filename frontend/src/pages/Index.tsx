import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import DocumentUpload from '@/components/DocumentUpload';
import ResultsPanel from '@/components/ResultsPanel';
import Footer from '@/components/Footer';
import { DocumentAnalysis } from '@/types/analysis';
import { sampleAnalysis } from '@/utils/demoData';
import { useToast } from '@/components/ui/use-toast';
import { analyzeDocument, backendAPI } from '@/lib/api';
import ErrorBoundary from '@/components/ErrorBoundary';

const Index = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<DocumentAnalysis | null>(null);
  const { toast } = useToast();
  const navigate = useNavigate();
  const userType = localStorage.getItem('user_type');

  useEffect(() => {
    const isAuthenticated = localStorage.getItem('access_token') !== null;
    const checkBackendServer = async () => {
      try {
        const response = await backendAPI({
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          }
        }
        );
        if (response.status !== 200) {
          throw new Error('Backend server is down');
        }
      } catch (error) {
        toast({
          title: "Session Expired",
          description: "Your session has expired. Please log in again.",
          variant: "destructive"
        });
        navigate('/login');
      }
    };
    checkBackendServer();

    if (!isAuthenticated) {
      navigate('/login');
    }
  }, [navigate]);

  const handleAnalyzeDocument = async (file: File) => {
    setIsAnalyzing(true);

    try {
      toast({
        title: "Analysis Started",
        description: `Analyzing ${file.name}...`,
      });

      const result = await analyzeDocument(file);
      setAnalysisResult({
        ...result,
        fileName: file.name,
        analyzedAt: new Date().toISOString()
      });

      toast({
        title: "Analysis Complete",
        description: "Your document has been analyzed successfully.",
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: error.response?.data?.error || "Failed to analyze document",
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
            {userType !== 'student' && (

              <DocumentUpload
                onAnalyze={handleAnalyzeDocument}
                isAnalyzing={isAnalyzing}
              />
            )}
            {userType === 'student' && (
              <div className="p-8 text-center bg-white rounded-xl shadow-lg">
                <h3 className="text-xl font-medium text-teal-700 mb-3">Document Upload Disabled</h3>
                <p className="text-gray-600">As a student, you cannot upload documents for analysis, You can only view your document metrics, If your
                  supervisor has uploaded a document for you, you can view it here.</p>
                <p className="text-gray-400">Please contact your supervisor for more information.</p>
                <p className="text-gray-350">If you are a supervisor, please log in with your supervisor account to upload documents.</p>
              </div>
            )}
          </div>

          {analysisResult && !isAnalyzing && (
            <div className="mt-12 animate-fade-in">
              <ErrorBoundary>
                <ResultsPanel analysis={analysisResult} />
              </ErrorBoundary>
            </div>
          )}

          {isAnalyzing && (
            <div className="mt-12 flex justify-center">
              <div className="p-8 text-center">
                <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-teal-500 border-r-transparent mb-4"></div>
                <p className="text-teal-700 text-lg">Analyzing your document...</p>
                <p className="text-gray-500 mt-2">This may take a few moments</p>
              </div>
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
