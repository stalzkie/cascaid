FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
# Vite inlines VITE_* env vars at build time, not container start -- must be a
# build ARG, not a runtime `environment:` entry in docker-compose.
ARG VITE_DASHBOARD_API_URL=http://localhost:8001
ENV VITE_DASHBOARD_API_URL=$VITE_DASHBOARD_API_URL
RUN npm run build

FROM node:20-alpine
WORKDIR /app
RUN npm install -g serve
COPY --from=build /app/dist ./dist
EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]
