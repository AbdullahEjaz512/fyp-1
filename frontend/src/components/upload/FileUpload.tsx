import { useState, useRef } from 'react';
import type { DragEvent } from 'react';
import { Upload, File, X, AlertCircle } from 'lucide-react';
import './FileUpload.css';

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  acceptedFormats?: string[];
  maxFiles?: number;
}

export const FileUpload = ({ 
  onFilesSelected, 
  acceptedFormats = ['.dcm', '.nii', '.nii.gz'],
  maxFiles = 10 
}: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFiles = (files: FileList | null): File[] => {
    if (!files || files.length === 0) return [];

    const validFiles: File[] = [];
    const errors: string[] = [];

    Array.from(files).forEach(file => {
      const fileName = file.name.toLowerCase();
      const isValid = acceptedFormats.some(format => fileName.endsWith(format));

      if (!isValid) {
        errors.push(`${file.name}: Invalid format`);
      } else if (validFiles.length < maxFiles) {
        validFiles.push(file);
      } else {
        errors.push(`Maximum ${maxFiles} files allowed`);
      }
    });

    if (errors.length > 0) {
      setError(errors.join(', '));
      setTimeout(() => setError(''), 5000);
    }

    return validFiles;
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = validateFiles(e.dataTransfer.files);
    if (files.length > 0) {
      onFilesSelected(files);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = validateFiles(e.target.files);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="file-upload-container">
      {error && (
        <div className="upload-error">
          <AlertCircle size={18} />
          {error}
        </div>
      )}

      <div
        className={`upload-dropzone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={acceptedFormats.join(',')}
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />

        <div className="upload-content">
          <Upload size={64} className="upload-icon" />
          <h3 className="upload-title">Drag & Drop MRI Files</h3>
          <p className="upload-text">or click to browse</p>
          <p className="upload-formats">
            Supported formats: {acceptedFormats.join(', ')}
          </p>
          <button type="button" className="btn btn-primary upload-button">
            <File size={18} />
            Select Files
          </button>
        </div>
      </div>
    </div>
  );
};

interface UploadedFileItemProps {
  file: File;
  onRemove: () => void;
  uploadProgress?: number;
  isUploading?: boolean;
  error?: string;
}

export const UploadedFileItem = ({ 
  file, 
  onRemove, 
  uploadProgress, 
  isUploading,
  error 
}: UploadedFileItemProps) => {
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className={`uploaded-file-item ${error ? 'error' : ''}`}>
      <div className="file-info">
        <File size={24} className="file-icon" />
        <div className="file-details">
          <div className="file-name">{file.name}</div>
          <div className="file-size">{formatFileSize(file.size)}</div>
          {error && <div className="file-error">{error}</div>}
        </div>
      </div>

      {isUploading && uploadProgress !== undefined && (
        <div className="upload-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <span className="progress-text">{uploadProgress}%</span>
        </div>
      )}

      <button
        type="button"
        onClick={onRemove}
        className="btn-icon remove-file"
        disabled={isUploading}
      >
        <X size={18} />
      </button>
    </div>
  );
};
