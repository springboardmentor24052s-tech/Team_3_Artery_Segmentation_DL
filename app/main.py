
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from app.routers import auth, patients, cases

app = FastAPI(
    title="Artery Segmentation and Stenosis Detection API"
)

app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(cases.router)
