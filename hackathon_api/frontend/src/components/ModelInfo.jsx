import React, { useState, useEffect } from 'react';
import { getModelInfo } from '../services/api';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await getModelInfo();
        setModelInfo(response.data);
      } catch (error) {
        console.error('Failed to fetch model info:', error);
      }
    };

    fetchModelInfo();
  }, []);

  return (
    <div className="model-info">
      <h3>Информация о модели</h3>
      <div className="model-stats">
        <div className="model-stat">
          <strong>Режим работы:</strong> CPU Based
        </div>
        <div className="model-stat">
          <strong>Количество классов:</strong> 11 инструментов
        </div>
        {modelInfo && (
          <div className="model-stat">
            <strong>Статус модели:</strong> 
            <span className={modelInfo.status === 'loaded' ? 'status-loaded' : 'status-error'}>
              {modelInfo.status === 'loaded' ? ' ✅ Загружена' : ' ❌ Ошибка'}
            </span>
          </div>
        )}
      </div>
      <div className="model-tips">
        <p>💡 <strong>Рекомендация:</strong> Загрузите папку с изображениями 1-11 инструментов для комплексного тестирования</p>
      </div>
    </div>
  );
};

export default ModelInfo;