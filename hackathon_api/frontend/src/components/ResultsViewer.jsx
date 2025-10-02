import React from 'react';

const ResultsViewer = ({ results, onNewTest }) => {
  const { transaction_id, sequence_number, processing_info, detection_results, files_processed } = results;

  return (
    <div className="results-viewer">
      <div className="results-header">
        <h2>📊 Результаты распознавания</h2>
        <div className="transaction-info">
          <span>Транзакция: <strong>{transaction_id}</strong></span>
          <span>Номер: <strong>#{sequence_number}</strong></span>
        </div>
      </div>

      {/* Processing Statistics */}
      <div className="stats-section">
        <h3>📈 Статистика обработки</h3>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{processing_info.total_images_processed}</div>
            <div className="stat-label">Обработано изображений</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{processing_info.avg_processing_time_ms}ms</div>
            <div className="stat-label">Среднее время</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{processing_info.mode}</div>
            <div className="stat-label">Режим работы</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{processing_info.flops}</div>
            <div className="stat-label">Производительность</div>
          </div>
        </div>
      </div>

      {/* Detection Summary */}
      <div className="summary-section">
        <h3>🛠️ Сводка по инструментам</h3>
        <div className={`summary-card ${detection_results.summary.status}`}>
          <div className="summary-main">
            <div className="summary-numbers">
              <span className="detected">{detection_results.summary.total_detected}</span>
              <span className="separator">из</span>
              <span className="expected">{detection_results.summary.total_expected}</span>
            </div>
            <div className="summary-percent">
              {detection_results.summary.match_percent}%
            </div>
          </div>
          <div className="summary-status">
            {detection_results.summary.status === 'success' ? '✅ Все инструменты найдены' : '⚠️ Требуется проверка'}
          </div>
        </div>
      </div>

      {/* Alerts */}
      {detection_results.alerts.length > 0 && (
        <div className="alerts-section">
          <h3>⚠️ Внимание, требуют ручной проверки:</h3>
          {detection_results.alerts.map((alert, index) => (
            <div key={index} className="alert-card">
              <div className="alert-icon">⚠️</div>
              <div className="alert-content">
                <div className="alert-message">{alert.message}</div>
                {alert.images_requiring_check && (
                  <div className="alert-details">
                    Проблемные изображения: {alert.images_requiring_check.join(', ')}
                  </div>
                )}
                {alert.missing_tools && (
                  <div className="alert-details">
                    Отсутствуют: {alert.missing_tools.join(', ')}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Detected Tools */}
      <div className="tools-section">
        <h3>✅ Найденные инструменты:</h3>
        <div className="tools-grid">
          {detection_results.detected_items.map((tool, index) => (
            <div key={index} className="tool-card">
              <div className="tool-header">
                <span className="tool-name">{tool.class_name}</span>
                <span className="tool-confidence">{tool.confidence_avg}</span>
              </div>
              <div className="tool-details">
                <span>Найден в {tool.frames_seen} изображениях</span>
                <span>ID: {tool.class_id}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Missing Tools */}
      {detection_results.missing_items.length > 0 && (
        <div className="missing-section">
          <h3>❌ Отсутствующие инструменты:</h3>
          <div className="missing-grid">
            {detection_results.missing_items.map((tool, index) => (
              <div key={index} className="missing-card">
                <div className="missing-name">{tool.name}</div>
                <div className="missing-id">ID: {tool.class_id}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="actions-section">
        <a 
          href="http://localhost:9001" 
          target="_blank" 
          rel="noopener noreferrer"
          className="action-button primary"
        >
          📁 Посмотреть изображения в MinIO
        </a>
        <button onClick={onNewTest} className="action-button secondary">
          🧪 Новый тест
        </button>
      </div>

      {/* Files Processed */}
      <div className="files-section">
        <h3>📄 Обработанные файлы:</h3>
        <div className="files-list">
          {files_processed.slice(0, 10).map((file, index) => (
            <div key={index} className={`file-item ${file.status}`}>
              <span className="file-name">{file.filename}</span>
              <span className="file-status">
                {file.status === 'success' ? `✅ ${file.detected_count} инструментов` : '❌ Ошибка'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResultsViewer;