import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { ChevronUp, ChevronDown } from "lucide-react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import { format } from "date-fns";
import { fetchMyDocuments, checkDocumentHistory } from "@/lib/api";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { useToast } from "@/components/ui/use-toast"; // Import useToast
import { useNavigate } from "react-router-dom"; // Import useNavigate
import { Button } from "@/components/ui/button"; // Import Button
import { DocumentAnalysis, DocumentHistoryRecord } from "@/types/analysis"; // Import types

const HistoryPage = () => {
    const [myDocuments, setMyDocuments] = useState([]);
    const [expandedDocumentHistory, setExpandedDocumentHistory] = useState({});
    const [expandedDocId, setExpandedDocId] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const { toast } = useToast();
    const navigate = useNavigate();

    useEffect(() => {
        const loadMyDocuments = async () => {
            setIsLoading(true);
            try {
                // Check authentication and user type
                const userType = localStorage.getItem('user_type');
                const accessToken = localStorage.getItem('access_token');

                if (!accessToken) {
                    toast({
                        title: "Authentication Required",
                        description: "Please log in to view your document history.",
                        variant: "destructive"
                    });
                    navigate('/login');
                    return;
                }

                if (userType === 'student') {
                    toast({
                        title: "Access Denied",
                        description: "Students view history on their dashboard. Please use the Student Dashboard.",
                        variant: "destructive"
                    });
                    navigate('/student-dashboard'); // Redirect students
                    return;
                }

                const documents = await fetchMyDocuments();
                setMyDocuments(documents);
            } catch (error) {
                console.error("Failed to fetch my documents:", error);
                toast({
                    title: "Error",
                    description: error.response?.data?.error || "Failed to load your documents.",
                    variant: "destructive"
                });
            } finally {
                setIsLoading(false);
            }
        };
        loadMyDocuments();
    }, [toast, navigate]); // Depend on toast and navigate

    const handleToggleExpand = async (document) => {
        if (expandedDocId === document.id) {
            setExpandedDocId(null); // Collapse if already expanded
            return;
        }

        if (expandedDocumentHistory[document.id]) {
            setExpandedDocId(document.id); // Just expand
            return;
        }

        try {
            toast({
                title: "Fetching Document History",
                description: `Loading detailed history for ${document.title || document.document_code}...`,
            });
            const fetchedDoc = await checkDocumentHistory(document.document_code);
            if (fetchedDoc && fetchedDoc.history_records) {
                setExpandedDocumentHistory(prev => ({
                    ...prev,
                    [document.id]: fetchedDoc.history_records.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
                }));
                setExpandedDocId(document.id);
            } else {
                toast({
                    title: "No History Found",
                    description: "Could not retrieve detailed history for this document.",
                    variant: "info"
                });
            }
        } catch (error) {
            console.error("Error fetching detailed history:", error);
            toast({
                title: "Error",
                description: error.response?.data?.error || "Failed to load detailed history.",
                variant: "destructive"
            });
        }
    };

    const getChange = (latestScore, prevScore) => {
        if (prevScore === undefined || prevScore === null) return null;
        const diff = latestScore - prevScore;
        return {
            change: diff.toFixed(1),
            isUp: diff >= 0,
        };
    };

    return (
        <div className="min-h-screen flex flex-col bg-teal-50">
            <Navbar />
            <main className="flex-grow py-12 container mx-auto px-6 space-y-8">
                <h2 className="text-3xl font-bold text-teal-800 text-center mb-8">My Document History</h2>

                {isLoading ? (
                    <div className="flex justify-center items-center h-64">
                        <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-teal-500 border-r-transparent"></div>
                        <p className="ml-4 text-teal-700 text-lg">Loading documents...</p>
                    </div>
                ) : myDocuments.length === 0 ? (
                    <div className="p-8 text-center bg-white rounded-xl shadow-lg">
                        <h3 className="text-xl font-medium text-teal-700 mb-3">No Documents Found</h3>
                        <p className="text-gray-600">You haven't uploaded any documents yet. Upload a document from the home page to see its history here.</p>
                        <Button
                            onClick={() => navigate('/')}
                            className="mt-4 px-6 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors"
                        >
                            Go to Upload Page
                        </Button>
                    </div>
                ) : (
                    myDocuments.map((doc) => {
                        const latest = doc;

                        const historyRecords = expandedDocumentHistory[doc.id] || [];
                        const prev = historyRecords.length > 1 ? historyRecords[historyRecords.length - 2] : null;

                        const pChange = getChange(latest.plagiarism_score, prev?.plagiarism_score);
                        const aiChange = getChange(latest.ai_score, prev?.ai_score);

                        return (
                            <Card
                                key={doc.id}
                                className="shadow-md bg-white rounded-xl cursor-pointer transition hover:shadow-lg"
                                onClick={() => handleToggleExpand(doc)}
                            >
                                <CardContent className="p-6 space-y-4">
                                    <div className="flex justify-between items-center">
                                        <h3 className="text-lg font-semibold text-teal-700">
                                            {latest.title || `Document ${latest.document_code}`}
                                        </h3>
                                        <span className="text-sm text-gray-500">
                                            Last analyzed: {format(new Date(latest.created_at), "PPPpp")}
                                        </span>
                                    </div>

                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                        <div className="text-center">
                                            <p className="text-sm text-gray-500">Plagiarism</p>
                                            <p className="text-xl font-bold text-red-600">
                                                {latest.plagiarism_score}%
                                                {pChange && (
                                                    <span className="ml-2 inline-flex items-center text-sm text-gray-500">
                                                        {pChange.isUp ? (
                                                            <ChevronUp className="w-4 h-4 text-red-500" />
                                                        ) : (
                                                            <ChevronDown className="w-4 h-4 text-green-500" />
                                                        )}
                                                        {Math.abs(Number(pChange.change))}%
                                                    </span>
                                                )}
                                            </p>
                                        </div>
                                        <div className="text-center">
                                            <p className="text-sm text-gray-500">AI Score</p>
                                            <p className="text-xl font-bold text-yellow-600">
                                                {latest.ai_score}%
                                                {aiChange && (
                                                    <span className="ml-2 inline-flex items-center text-sm text-gray-500">
                                                        {aiChange.isUp ? (
                                                            <ChevronUp className="w-4 h-4 text-yellow-600" />
                                                        ) : (
                                                            <ChevronDown className="w-4 h-4 text-blue-600" />
                                                        )}
                                                        {Math.abs(Number(aiChange.change))}%
                                                    </span>
                                                )}
                                            </p>
                                        </div>
                                        <div className="text-center">
                                            <p className="text-sm text-gray-500">Originality</p>
                                            <p className="text-xl font-bold text-green-600">
                                                {latest.originality_score}%
                                            </p>
                                        </div>
                                        <div className="text-center">
                                            <p className="text-sm text-gray-500">Doc Code</p>
                                            <p className="text-xl font-bold text-gray-700">
                                                {latest.document_code}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Display loading spinner for history if currently fetching for this doc */}
                                    {expandedDocId === doc.id && !expandedDocumentHistory[doc.id] && (
                                        <div className="pt-4 flex justify-center items-center">
                                            <div className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-teal-500 border-r-transparent"></div>
                                            <p className="ml-2 text-teal-700 text-sm">Loading history...</p>
                                        </div>
                                    )}

                                    {expandedDocId === doc.id && expandedDocumentHistory[doc.id] && (
                                        <div className="pt-4">
                                            <h4 className="text-lg font-semibold mb-3 text-teal-700">Analysis Trend</h4>
                                            <ResponsiveContainer width="100%" height={250}>
                                                <LineChart data={historyRecords}>
                                                    <CartesianGrid strokeDasharray="3 3" />
                                                    <XAxis
                                                        dataKey="created_at"
                                                        tickFormatter={(v) => format(new Date(v), "MM/dd HH:mm")}
                                                        angle={-45}
                                                        textAnchor="end"
                                                        height={60}
                                                        interval="preserveStartEnd"
                                                    />
                                                    <YAxis domain={[0, 100]} />
                                                    <Tooltip labelFormatter={(label) => `Analyzed: ${format(new Date(label), "PPPpp")}`} />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="plagiarism_score"
                                                        stroke="#ef4444" // Red
                                                        name="Plagiarism"
                                                    />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="ai_score"
                                                        stroke="#fbbf24" // Yellow
                                                        name="AI Score"
                                                    />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="originality_score"
                                                        stroke="#10b981" // Green
                                                        name="Originality"
                                                    />
                                                </LineChart>
                                            </ResponsiveContainer>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        );
                    })
                )}
            </main>
            <Footer />
        </div>
    );
};

export default HistoryPage;