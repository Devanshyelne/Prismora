web: python -m scripts.fetch_model_assets && exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180 --access-logfile - app:app
