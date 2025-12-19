import { useState } from 'react';
import './AssistantPage.css';
import { assistantService } from '../services/assistant.service';

interface Message {
  role: 'user' | 'assistant';
  text: string;
}

export default function AssistantPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    const content = input.trim();
    if (!content) return;
    setMessages((prev) => [...prev, { role: 'user', text: content }]);
    setInput('');
    setLoading(true);
    try {
      const res = await assistantService.chat(content);
      const answer = res?.data?.answer ?? 'No response';
      setMessages((prev) => [...prev, { role: 'assistant', text: answer }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: 'Sorry, I had trouble responding. Please try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="assistant-page">
      <div className="assistant-header">
        <h2>Seg-Mind Assistant</h2>
        <p>Ask about setup, features, or generate reports.</p>
      </div>
      <div className="chat-container">
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>{m.text}</div>
          ))}
          {loading && <div className="message assistant">Thinking...</div>}
        </div>
        <div className="input-row">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your question..."
          />
          <button onClick={sendMessage} disabled={loading}>Send</button>
        </div>
      </div>
    </div>
  );
}
