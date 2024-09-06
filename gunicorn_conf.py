# gunicorn_conf.py

import multiprocessing

# Number of worker processes (2-4 x $(NUM_CORES))
workers = multiprocessing.cpu_count() * 2 + 1

# Using Uvicorn workers to serve ASGI applications
worker_class = "uvicorn.workers.UvicornWorker"

# The address and port to bind to
bind = "0.0.0.0:8000"

# Enable logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Graceful timeout for workers to handle requests
timeout = 120

# Enable graceful reloads (useful during development; disable for production)
reload = False