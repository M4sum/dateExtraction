web: gunicorn app:app
web: gunicorn --bind 0.0.0.0:${PORT} wsgi
web: export FLASK_ENV=development
web: export FLASK_APP=app.py
web: flask run