import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { collaborationService } from '../../services/file.service';
import { MessageSquare, Send, Trash2, Edit2, X, Check, User } from 'lucide-react';
import './Collaboration.css';

interface DiscussionThreadProps {
  fileId: number;
}

export const DiscussionThread = ({ fileId }: DiscussionThreadProps) => {
  const [newComment, setNewComment] = useState('');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState('');
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ['discussion', fileId],
    queryFn: () => collaborationService.getDiscussion(fileId),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const addCommentMutation = useMutation({
    mutationFn: (comment: string) => collaborationService.addComment(fileId, comment),
    onSuccess: () => {
      setNewComment('');
      queryClient.invalidateQueries({ queryKey: ['discussion', fileId] });
    },
  });

  const updateCommentMutation = useMutation({
    mutationFn: ({ id, comment }: { id: number; comment: string }) => 
      collaborationService.updateComment(id, comment),
    onSuccess: () => {
      setEditingId(null);
      setEditText('');
      queryClient.invalidateQueries({ queryKey: ['discussion', fileId] });
    },
  });

  const deleteCommentMutation = useMutation({
    mutationFn: (id: number) => collaborationService.deleteComment(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discussion', fileId] });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newComment.trim()) return;
    addCommentMutation.mutate(newComment);
  };

  const handleEdit = (id: number, currentText: string) => {
    setEditingId(id);
    setEditText(currentText);
  };

  const handleSaveEdit = (id: number) => {
    if (!editText.trim()) return;
    updateCommentMutation.mutate({ id, comment: editText });
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <div className="discussion-thread">
        <div className="panel-header">
          <h4><MessageSquare size={18} /> Discussion</h4>
        </div>
        <div className="loading-state">Loading discussion...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="discussion-thread">
        <div className="panel-header">
          <h4><MessageSquare size={18} /> Discussion</h4>
        </div>
        <div className="error-state">Failed to load discussion</div>
      </div>
    );
  }

  const comments = data?.comments || [];

  return (
    <div className="discussion-thread">
      <div className="panel-header">
        <h4><MessageSquare size={18} /> Discussion ({comments.length})</h4>
      </div>

      <div className="comments-container">
        {comments.length === 0 ? (
          <div className="empty-state">
            <MessageSquare size={24} />
            <p>No comments yet</p>
            <span>Start the discussion below</span>
          </div>
        ) : (
          <div className="comments-list">
            {comments.map((comment) => (
              <div 
                key={comment.id} 
                className={`comment-item ${comment.doctor.is_current_user ? 'own-comment' : ''}`}
              >
                <div className="comment-avatar">
                  <User size={16} />
                </div>
                <div className="comment-content">
                  <div className="comment-header">
                    <span className="comment-author">
                      {comment.doctor.name}
                      {comment.doctor.is_current_user && <span className="you-badge">You</span>}
                    </span>
                    {comment.doctor.specialization && (
                      <span className="comment-specialization">{comment.doctor.specialization}</span>
                    )}
                    <span className="comment-time">{formatDate(comment.created_at)}</span>
                    {comment.updated_at && comment.updated_at !== comment.created_at && (
                      <span className="edited-badge">(edited)</span>
                    )}
                  </div>
                  
                  {editingId === comment.id ? (
                    <div className="edit-form">
                      <textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        rows={2}
                      />
                      <div className="edit-actions">
                        <button 
                          className="save-btn" 
                          onClick={() => handleSaveEdit(comment.id)}
                          disabled={updateCommentMutation.isPending}
                        >
                          <Check size={14} /> Save
                        </button>
                        <button 
                          className="cancel-btn" 
                          onClick={() => { setEditingId(null); setEditText(''); }}
                        >
                          <X size={14} /> Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <p className="comment-text">{comment.text}</p>
                  )}
                  
                  {comment.doctor.is_current_user && editingId !== comment.id && (
                    <div className="comment-actions">
                      <button 
                        className="action-btn edit" 
                        onClick={() => handleEdit(comment.id, comment.text)}
                      >
                        <Edit2 size={12} /> Edit
                      </button>
                      <button 
                        className="action-btn delete" 
                        onClick={() => {
                          if (confirm('Delete this comment?')) {
                            deleteCommentMutation.mutate(comment.id);
                          }
                        }}
                        disabled={deleteCommentMutation.isPending}
                      >
                        <Trash2 size={12} /> Delete
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="comment-form">
        <textarea
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Add your comment or observation..."
          rows={2}
        />
        <button 
          type="submit" 
          className="send-btn"
          disabled={!newComment.trim() || addCommentMutation.isPending}
        >
          {addCommentMutation.isPending ? 'Sending...' : <><Send size={16} /> Send</>}
        </button>
      </form>
    </div>
  );
};
