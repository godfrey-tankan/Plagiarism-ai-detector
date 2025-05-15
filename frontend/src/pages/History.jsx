import { useEffect, useState } from "react";
import axios from "axios";
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
import { documentHistory } from "@/lib/api";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const HistoryPage = () => {
    const [history, setHistory] = useState([]);
    const [expandedDoc, setExpandedDoc] = useState(null);

    useEffect(() => {
        const fetchHistory = async () => {
            const res = await documentHistory({
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("access_token")}`,
                },
            });
            setHistory(res.data);
        };
        fetchHistory();
    }, []);

    const grouped = history.reduce((acc, item) => {
        const docId = item.document;
        if (!acc[docId]) acc[docId] = [];
        acc[docId].push(item);
        return acc;
    }, {});

    return (
        <div className="min-h-screen flex flex-col bg-teal-50">
            <Navbar />
            <main className="flex-grow py-12 container mx-auto px-6 space-y-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">Document History</h2>

                {Object.entries(grouped).map(([docId, entries]) => {
                    const sorted = entries.sort(
                        (a, b) => new Date(a.created_at) - new Date(b.created_at)
                    );
                    const latest = sorted[sorted.length - 1];
                    const prev = sorted.length > 1 ? sorted[sorted.length - 2] : null;

                    const getChange = (field) => {
                        if (!prev) return null;
                        const diff = latest[field] - prev[field];
                        return {
                            change: diff.toFixed(1),
                            isUp: diff >= 0,
                        };
                    };

                    const pChange = getChange("plagiarism_score");
                    const aiChange = getChange("ai_score");

                    return (
                        <Card
                            key={docId}
                            className="shadow-md bg-white rounded-xl cursor-pointer transition hover:shadow-lg"
                            onClick={() =>
                                setExpandedDoc((prevId) => (prevId === docId ? null : docId))
                            }
                        >
                            <CardContent className="p-6 space-y-4">
                                <div className="flex justify-between items-center">
                                    <h3 className="text-lg font-semibold text-teal-700">
                                        {latest.document_name}
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
                                                    {Math.abs(pChange.change)}%
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
                                                    {Math.abs(aiChange.change)}%
                                                </span>
                                            )}
                                        </p>
                                    </div>
                                </div>

                                {expandedDoc === docId && (
                                    <div className="pt-4">
                                        <ResponsiveContainer width="100%" height={250}>
                                            <LineChart data={sorted}>
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis
                                                    dataKey="created_at"
                                                    tickFormatter={(v) => format(new Date(v), "MM/dd")}
                                                />
                                                <YAxis />
                                                <Tooltip />
                                                <Line
                                                    type="monotone"
                                                    dataKey="plagiarism_score"
                                                    stroke="#f87171"
                                                    name="Plagiarism"
                                                />
                                                <Line
                                                    type="monotone"
                                                    dataKey="ai_score"
                                                    stroke="#fbbf24"
                                                    name="AI Score"
                                                />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    );
                })}
            </main>
            <Footer />
        </div>
    );
};

export default HistoryPage;
