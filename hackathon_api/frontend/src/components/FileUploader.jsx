import React, { useState } from 'react';
import { uploadSingleImage, uploadBatchImages, uploadZipFolder } from '../services/api';

const FileUploader = ({ onResultsReady, isProcessing, setIsProcessing }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadType, setUploadType] = useState('single');

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    setIsProcessing(true);
    try {
      let result;
      
      if (uploadType === 'single') {
        result = await uploadSingleImage(selectedFiles[0]);
      } else if (uploadType === 'batch') {
        result = await uploadBatchImages(selectedFiles);
      } else if (uploadType === 'zip') {
        result = await uploadZipFolder(selectedFiles[0]);
      }
      
      onResultsReady(result.data);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Ошибка при загрузке файлов');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="file-uploader">
      <div className="upload-options">
        <h3>📤 Выберите способ загрузки:</h3>
        
        <div className="option-group">
          <label className="option-label">
            <input 
              type="radio" 
              value="single" 
              checked={uploadType === 'single'}
              onChange={(e) => setUploadType(e.target.value)}
            />
            <span className="option-text">
              <strong>📷 Одно изображение</strong>
              <small>Быстрая проверка одного фото</small>
            </span>
          </label>

          <label className="option-label">
            <input 
              type="radio" 
              value="batch" 
              checked={uploadType === 'batch'}
              onChange={(e) => setUploadType(e.target.value)}
            />
            <span className="option-text">
              <strong>🖼️ Несколько изображений</strong>
              <small>Выберите несколько файлов</small>
            </span>
          </label>

          <label className="option-label">
            <input 
              type="radio" 
              value="zip" 
              checked={uploadType === 'zip'}
              onChange={(e) => setUploadType(e.target.value)}
            />
            <span className="option-text">
              <strong>📁 ZIP архив</strong>
              <small>Загрузите папку с изображениями</small>
            </span>
          </label>
        </div>
      </div>

      <div className="file-input-section">
        <div className="file-input">
          <input
            type="file"
            multiple={uploadType !== 'single' && uploadType !== 'zip'}
            accept={uploadType === 'zip' ? '.zip' : '.jpg,.jpeg,.png'}
            onChange={handleFileSelect}
            disabled={isProcessing}
          />
        </div>

        {selectedFiles.length > 0 && (
          <div className="file-list">
            <h4>Выбранные файлы ({selectedFiles.length}):</h4>
            <ul>
              {selectedFiles.slice(0, 5).map((file, index) => (
                <li key={index}>{file.name}</li>
              ))}
              {selectedFiles.length > 5 && <li>... и еще {selectedFiles.length - 5} файлов</li>}
            </ul>
          </div>
        )}
      </div>

      <div className="upload-actions">
        <button 
          onClick={handleUpload} 
          disabled={selectedFiles.length === 0 || isProcessing}
          className={`upload-button ${isProcessing ? 'processing' : ''}`}
        >
          {isProcessing ? (
            <>
              <div className="spinner"></div>
              Обработка...
            </>
          ) : (
            'Запустить распознавание'
          )}
        </button>
      </div>

      <div className="upload-tips">
        <h4>💡 Рекомендации:</h4>
        <ul>
          <li>Загрузите изображения с 1-11 инструментами для тестирования</li>
          <li>Модель найдет изображения, где комплект не полный</li>
          <li>Система автоматически выделит найденные инструменты</li>
        </ul>
      </div>
    </div>
  );
};

export default FileUploader;