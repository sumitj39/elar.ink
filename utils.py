import os
import time

from prometheus_client import REGISTRY, Histogram
from sqlalchemy import create_engine


def bypass_https():
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry

    # Bypass SSL verification
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.verify = False  # THIS IS RISKY; DON'T USE IN PRODUCTION
    requests.sessions.Session.request = session.request


def get_db_engine():
    DATABASE_URL = os.environ.get(
        "DATABASE_URL", "postgresql://alaradmin:protected@localhost:5432/alardict"
    )
    return create_engine(DATABASE_URL)


from prometheus_client import Histogram

METRICS = {}  # Store created metrics


class timed:
    def __init__(self, metric_name):
        if metric_name not in METRICS:
            # Create a new histogram for this metric_name
            METRICS[metric_name] = Histogram(
                metric_name, f"Time spent in {metric_name}"
            )
        self.metric = METRICS[metric_name]

    # For use as a context manager
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.metric.observe(elapsed_time)  # Observe the elapsed time

    # For use as a decorator
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
