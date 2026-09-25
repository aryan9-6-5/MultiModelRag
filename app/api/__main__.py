"""Entry point: python -m app.api"""
import uvicorn

from app.config.settings import settings

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
