import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from a2wsgi import ASGIMiddleware  # noqa: E402
from api.main import app  # noqa: E402

application = ASGIMiddleware(app)