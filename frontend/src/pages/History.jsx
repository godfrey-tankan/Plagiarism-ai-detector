import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { documentHistory } from "@/lib/api";
import { Button } from "@/components/ui/button";
import Footer from '@/components/Footer';
import Navbar from '@/components/Navbar';


const HistoryPage = () => {
    const [history, setHistory] = useState([]);

    useEffect(() => {
        const fetchHistory = async () => {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            try {
                const response = await documentHistory({
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                    }
                });
                setHistory(response.data);
            } catch (err) {
                console.error("Error fetching history", err);
            }
        };

        fetchHistory();
    }, []);

    return (
        <div className="min-h-screen flex flex-col bg-teal-50" >
            <Navbar />

            <h2 className="text-2xl font-bold mb-4 flex">Document History</h2>
            {history.length === 0 ? (
                <p>No history available.</p>
            ) : (
                <div className="space-y-4">
                    {history.map(item => (
                        <Card key={item.id} className="p-4">
                            <p><strong>Doc ID:</strong> {item.document}</p>
                            <p><strong>Plagiarism Score:</strong> {item.plagiarism_score}%</p>
                            <p><strong>AI Score:</strong> {item.ai_score}%</p>
                            <p><strong>Created:</strong> {new Date(item.created_at).toLocaleString()}</p>
                        </Card>
                    ))}
                </div>
            )}
            {/* <Footer /> */}
        </div>
    );
};

export default HistoryPage;
