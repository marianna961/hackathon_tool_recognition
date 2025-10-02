FROM node:18-alpine

WORKDIR /app

# Копируем package файлы первыми (для кэширования)
COPY package.json package-lock.json* ./

# Устанавливаем зависимости
RUN npm install

# Копируем исходный код
COPY . .

# Создаем .env файл для Docker
RUN echo "VITE_API_URL=http://localhost:8000" > .env

# Открываем порт
EXPOSE 3000

# Запускаем dev сервер
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]