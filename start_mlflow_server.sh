#!/bin/bash

# Остановка предыдущих экземпляров MLFlow сервера
kill $(lsof -t -i:5000) 2>/dev/null

# Переход в директорию mlruns
cd artifacts/mlruns || exit 1

# Запуск MLFlow сервера
mlflow server --backend-store-uri file:///$(pwd) --default-artifact-root ../artifacts --host 0.0.0.0 --port 5000

echo "MLFlow server started successfully!"
