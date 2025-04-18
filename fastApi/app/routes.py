from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import pandas as pd
from app.utils import preprocess_data
from app.forecast import get_forecast
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
templates = Jinja2Templates(directory="templates")

router = APIRouter()

# Load and preprocess dataset
df = preprocess_data("../building_consumption.csv")


# Define the dashboard route
@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    # This renders the forecast.html template
    return templates.TemplateResponse("forecast.html", {"request": request})

@router.get("/api/data")
async def get_filtered_data(building_ids: str = ""):
    ids = building_ids.split(",") if building_ids else df['campus_id'].unique().tolist()
    filtered = df[df['campus_id'].isin(ids)]
    return filtered.to_dict(orient="records")

@router.get("/api/forecast")
async def get_forecast_api(building_ids: str = ""):
    ids = building_ids.split(",") if building_ids else df['campus_id'].unique().tolist()
    filtered = df[df['campus_id'].isin(ids)]
    forecast = get_forecast(filtered)
    return forecast

