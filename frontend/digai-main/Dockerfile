FROM node:20 as builder
WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn
COPY . .

ENV VITE_CHATBOT_API=http://localhost:5000
ENV VITE_RECOMM_API=http://localhost:5001


RUN yarn build

FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx
EXPOSE 80
