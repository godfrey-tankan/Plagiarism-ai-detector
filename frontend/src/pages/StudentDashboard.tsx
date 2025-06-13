// src/pages/StudentDashboard.tsx
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useToast } from '@/components/ui/use-toast';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { DocumentAnalysis } from '@/types/analysis'; // Import DocumentAnalysis type
import { checkDocumentHistory } from '@/lib/api'; // Import the new API call
import ResultsPanel from '@/components/ResultsPanel'; // Reuse ResultsPanel
import ErrorBoundary from '@/components/ErrorBoundary'; // Reuse ErrorBoundary

const StudentDashboard = () => {
    const [documentCode, setDocumentCode] = useState('');
    const [fetchingHistory, setFetchingHistory] = useState(false);
    const [historyResult, setHistoryResult] = useState<DocumentAnalysis | null>(null);
    const { toast } = useToast();
    const navigate = useNavigate();

    useEffect(() => {
        // Ensure student is authenticated
        const userType = localStorage.getItem('user_type');
        const accessToken = localStorage.getItem('access_token');

        if (userType !== 'student' || !accessToken) {
            toast({
                title: "Access Denied",
                description: "You must be logged in as a student to access this page.",
                variant: "destructive"
            });
            navigate('/login');
        }
        // Optionally, fetch documents specifically for this student's email here
        // This would require a new API endpoint on the backend (e.g., /api/documents/my-documents/)
        // For now, we'll focus on the document code lookup.
    }, [navigate, toast]);

    const handleFetchDocumentHistory = async () => {
        if (!documentCode) {
            toast({
                title: "Missing Document Code",
                description: "Please enter a document code to view its history.",
                variant: "destructive"
            });
            return;
        }

        setFetchingHistory(true);
        setHistoryResult(null); // Clear previous results
        try {
            toast({
                title: "Fetching History",
                description: `Retrieving document history for code: ${documentCode}...`,
            });

            const result: DocumentAnalysis = await checkDocumentHistory(documentCode);
            setHistoryResult(result);
            toast({
                title: "History Retrieved",
                description: `Document history for code '${documentCode}' loaded.`,
            });
        } catch (error: unknown) {
            console.error("Failed to fetch document history:", error);
            let errorMessage = "Could not find document with that code.";
            if (typeof error === "object" && error !== null && "response" in error) {
                const err = error as { response?: { data?: { error?: string } } };
                errorMessage = err.response?.data?.error || errorMessage;
            }
            toast({
                title: "Error Fetching History",
                description: errorMessage,
                variant: "destructive"
            });
        } finally {
            setFetchingHistory(false);
        }
    };

    return (
        <div className="min-h-screen flex flex-col bg-teal-50">
            <Navbar />
            <main className="flex-grow py-12">
                <div className="container mx-auto px-6 space-y-8">
                    <h2 className="text-3xl font-bold text-teal-800 text-center mb-8">Student Dashboard</h2>

                    {/* Section for checking document history by code */}
                    <div className="bg-white rounded-xl p-8 shadow-lg max-w-3xl mx-auto">
                        <h3 className="text-xl font-bold text-teal-700 mb-4">Check Document History by Code</h3>
                        <p className="text-gray-600 mb-4">
                            Enter the document code provided by your supervisor or from an analysis report to view its details and history.
                        </p>
                        <div className="flex flex-col sm:flex-row gap-4">
                            <Input
                                type="text"
                                placeholder="Enter document code (e.g., 0F825506399A)"
                                value={documentCode}
                                onChange={(e) => setDocumentCode(e.target.value.toUpperCase())} // Convert to uppercase as backend generates uppercase
                                className="flex-grow"
                                disabled={fetchingHistory}
                            />
                            <Button
                                onClick={handleFetchDocumentHistory}
                                disabled={fetchingHistory || !documentCode}
                                className="bg-teal-600 hover:bg-teal-700"
                            >
                                {fetchingHistory ? 'Searching...' : 'View History'}
                            </Button>
                        </div>
                    </div>

                    {/* Display fetched history results */}
                    {fetchingHistory && (
                        <div className="mt-12 flex justify-center">
                            <div className="p-8 text-center">
                                <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-teal-500 border-r-transparent mb-4"></div>
                                <p className="text-teal-700 text-lg">Fetching document history...</p>
                            </div>
                        </div>
                    )}

                    {historyResult && !fetchingHistory && (
                        <div className="mt-12 animate-fade-in">
                            <ErrorBoundary>
                                {/* Reuse ResultsPanel to display the fetched analysis */}
                                <ResultsPanel analysis={historyResult} />
                            </ErrorBoundary>
                        </div>
                    )}

                    {!historyResult && !fetchingHistory && documentCode && (
                        <div className="mt-12 p-8 text-center bg-white rounded-xl shadow-lg max-w-3xl mx-auto">
                            <h3 className="text-xl font-medium text-teal-700 mb-3">No Document Found</h3>
                            <p className="text-gray-600">The document code you entered was not found. Please double-check the code and try again.</p>
                        </div>
                    )}

                    {/* Section for documents sent to the student (placeholder for future API) */}
                    <div className="bg-white rounded-xl p-8 shadow-lg max-w-3xl mx-auto mt-8">
                        <h3 className="text-xl font-bold text-teal-700 mb-4">Documents Shared With You</h3>
                        <p className="text-gray-600 mb-4">
                            This section will display documents that have been analyzed by your supervisor and specifically shared with your email address.
                        </p>
                        <p className="text-gray-500">
                            (Feature coming soon: Requires backend API to list documents by recipient email for the logged-in user.)
                        </p>
                        {/* You could add a loading spinner or a list of documents here */}
                    </div>

                </div>
            </main>
            <Footer />
        </div>
    );
};

export default StudentDashboard;