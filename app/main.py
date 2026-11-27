from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db
from app.routers import auth, cases, images, patients


app = FastAPI(
    title="Artery Image API",
    description="Image upload, preprocessing and stenosis analysis",
    version="0.2.0",
)

# Allow the React development server to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register application routers.
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(cases.router)
app.include_router(images.router)


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable",
        ) from exc

    return {"status": "healthy", "database": "connected"}


@app.get("/")
def root():
    return {
        "message": "Artery Image API is running",
        "status": "Connected",
    }
