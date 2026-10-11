from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api_routes import router
from app.config import settings
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(title=settings.APP_NAME, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=settings.CORS_ORIGINS.strip() != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"message": settings.APP_NAME}
