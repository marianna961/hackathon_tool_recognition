import React, { useState } from 'react';

const ResultsViewer = ({ results, onNewTest }) => {
  // Обрабатываем результаты как массив, даже если передан один объект
  const allResults = Array.isArray(results) ? results : [results];

  if (!allResults.length) {
    return <div>Нет данных для отображения</div>;
  }

  const [showJson, setShowJson] = useState(false);

  return (
    <div className="results-viewer">
      <div className="results-header">
        <h2>📊 Результаты распознавания</h2>
        <button onClick={() => setShowJson(!showJson)} className="json-toggle">
          {showJson ? 'Скрыть JSON' : 'Показать JSON'}
        </button>
      </div>

      {/* Общая сводка (если несколько результатов) */}
      {allResults.length > 1 && (
        <div className="summary-section">
          <h3>🛠️ Общая сводка</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{allResults.length}</div>
              <div className="stat-label">Обработано изображений</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {Math.round(allResults.reduce((sum, r) => sum + (r.processing_time_ms || 0), 0) / allResults.length)}ms
              </div>
              <div className="stat-label">Среднее время</div>
            </div>
          </div>
        </div>
      )}

      {/* Результаты по каждому изображению */}
      <div className="results-list">
        {allResults.map((result, index) => {
          const {
            transaction_id,
            sequence_number,
            status,
            raw_url,
            viz_url,
            processing_time_ms,
            summary,
            detected_items,
            missing_items,
            alerts,
            filename,
          } = result;

          const isIncomplete = summary.total_detected < 11;

          return (
            <div key={index} className={`result-item ${isIncomplete ? 'incomplete' : ''}`}>
              <div className="transaction-info">
                <span>Транзакция: <strong>{transaction_id}</strong></span>
                <span>Номер: <strong>#{sequence_number}</strong></span>
                <span>Статус: <strong>{status}</strong></span>
                {isIncomplete && <span className="incomplete-warning">⚠️ Недостаточно инструментов</span>}
              </div>

              {/* Статистика обработки */}
              <div className="stats-section">
                <h3>📈 Статистика</h3>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-value">{processing_time_ms}ms</div>
                    <div className="stat-label">Время обработки</div>
                  </div>
                </div>
              </div>

              {/* Сводка по инструментам */}
              <div className="summary-section">
                <h3>🛠️ Сводка</h3>
                <div className={`summary-card ${summary.status}`}>
                  <div className="summary-main">
                    <div className="summary-numbers">
                      <span className="detected">{summary.total_detected}</span>
                      <span className="separator">из</span>
                      <span className="expected">{summary.total_expected}</span>
                    </div>
                    <div className="summary-percent">
                      {summary.match_percent}%
                    </div>
                  </div>
                  <div className="summary-status">
                    {summary.status === 'success' ? '✅ Все инструменты найдены' : '⚠️ Требуется проверка'}
                  </div>
                </div>
              </div>

              {/* Оповещения */}
              {alerts && alerts.length > 0 && (
                <div className="alerts-section">
                  <h3>⚠️ Внимание</h3>
                  {alerts.map((alert, idx) => (
                    <div key={idx} className="alert-card">
                      <div className="alert-icon">⚠️</div>
                      <div className="alert-content">{alert}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Инструменты (объединенная секция) */}
              <div className="tools-section">
                <h3>🛠️ Инструменты</h3>
                <div className="tools-grid">
                  <div className="tool-card">
                    <h4>Найденные инструменты:</h4>
                    <ul className="tool-list">
                      {detected_items.map((tool, idx) => (
                        <li key={idx} className="tool-item">
                          <span className="tool-check">✅</span>
                          <span className="tool-name">{tool.class_name}</span>
                          <span className="tool-confidence">({tool.aggregated_confidence.toFixed(2)})</span>
                        </li>
                      ))}
                    </ul>
                    <h4>Отсутствующие инструменты:</h4>
                    <ul className="tool-list">
                      {missing_items.map((tool, idx) => (
                        <li key={idx} className="tool-item">
                          <span className="tool-check">❌</span>
                          <span className="tool-name">{tool.name}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              {/* Ссылка на изображения */}
              {(raw_url || viz_url) && (
                <div className="image-links">
                  <h3>📷 Изображения</h3>
                  {raw_url && (
                    <a href={raw_url} target="_blank" rel="noopener noreferrer" className="action-button primary">
                      Сырое изображение
                    </a>
                  )}
                  {viz_url && (
                    <a href={viz_url} target="_blank" rel="noopener noreferrer" className="action-button primary">
                      Визуализация
                    </a>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* JSON-отчет */}
      {showJson && (
        <div className="json-section">
          <h3>📋 JSON-отчет</h3>
          <div className="json-container">
            {allResults.map((result, index) => (
              <pre key={index} className="json-pre">
                {JSON.stringify(result, null, 2)}
              </pre>
            ))}
          </div>
        </div>
      )}

      {/* Действия */}
      <div className="actions-section">
        <button onClick={onNewTest} className="action-button secondary">
          🧪 Новый тест
        </button>
      </div>
    </div>
  );
};

export default ResultsViewer;