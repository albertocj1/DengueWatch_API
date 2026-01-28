
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, field_validator, create_model
from typing import List, Dict, Type
import tensorflow as tf
import joblib
import types, sys, datetime
from urllib.parse import quote_plus



# Fake tz_util to avoid the import error
if 'bson.tz_util' not in sys.modules:
    tz_util = types.SimpleNamespace(utc=datetime.timezone.utc)
    sys.modules['bson.tz_util'] = tz_util

from pymongo import MongoClient
from bson import Binary
import datetime
from fastapi.middleware.cors import CORSMiddleware
import io

# =====================================================
# 🚀 FASTAPI APP
# =====================================================
app = FastAPI(
    title="Dengue Early Warning System API",
    description="CNN-LSTM dengue risk forecasting with MongoDB integration",
    version="5.1"
)

origins = [
    "https://dengue-watch-website.vercel.app", 
    "http://127.0.0.1:5500",  # your Live Server address
    "http://localhost:5500",  # optional, in case Live Server uses localhost
    "http://localhost:8000",  # optional, for API testing
    "http://127.0.0.1:3000",  # Live Preview
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 📦 MODEL & SCALER PATH
# =====================================================
MODEL_PATH = "dengue_classification_model.keras"
SCALER_PATH = "scaler_classification.pkl"

# =====================================================
# ⏱ WINDOW SIZE
# =====================================================
WINDOW = 4

# =====================================================
# 🧠 FEATURES
# =====================================================
final_feature_columns = [
    'YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'RH', 'SUNSHINE',
    'POPULATION', 'LAND AREA', 'POP_DENSITY',
    'CASES_lag1', 'CASES_lag2', 'CASES_lag3', 'CASES_lag4',
    'RAINFALL_lag1', 'RAINFALL_lag2', 'RAINFALL_lag3', 'RAINFALL_lag4',
    'TMAX_lag1', 'TMAX_lag2', 'TMAX_lag3', 'TMAX_lag4',
    'TMIN_lag1', 'TMIN_lag2', 'TMIN_lag3', 'TMIN_lag4',
    'TMEAN_lag1', 'TMEAN_lag2', 'TMEAN_lag3', 'TMEAN_lag4',
    'RH_lag1', 'RH_lag2', 'RH_lag3', 'RH_lag4',
    'SUNSHINE_lag1', 'SUNSHINE_lag2', 'SUNSHINE_lag3', 'SUNSHINE_lag4',
    'CASES_roll2_mean', 'CASES_roll4_mean', 'CASES_roll2_sum', 'CASES_roll4_sum',
    'RAINFALL_roll2_mean', 'RAINFALL_roll4_mean',
    'RAINFALL_roll2_sum', 'RAINFALL_roll4_sum',
    'TMEAN_roll2_mean', 'TMEAN_roll4_mean', 'TMEAN_roll2_sum',
    'TMEAN_roll4_sum', 'RH_roll2_mean', 'RH_roll4_mean',
    'RH_roll2_sum', 'RH_roll4_sum'
]

# =====================================================
# 🚨 RISK LABELS
# =====================================================
risk_labels = ["Low", "Moderate", "High", "VeryHigh"]

# =====================================================
# 🌆 CITY MAPPING (PRIMARY KEY = LAND AREA)
# =====================================================
land_area_to_city = {
    24.98: "MANILA CITY",
    171.71: "QUEZON CITY",
    55.8: "CALOOCAN CITY",
    32.69: "LAS PINAS CITY",
    21.57: "MAKATI CITY",
    15.71: "MALABON CITY",
    9.29: "MANDALUYONG CITY",
    21.52: "MARIKINA CITY",
    39.75: "MUNTINLUPA CITY",
    8.94: "NAVOTAS CITY",
    46.57: "PARANAQUE CITY",
    13.97: "PASAY CITY",
    48.46: "PASIG CITY",
    10.4: "PATEROS",
    5.95: "SAN JUAN CITY",
    45.21: "TAGUIG CITY",
    47.02: "VALENZUELA CITY"
}

# =====================================================
# 📥 LOAD MODEL & SCALER
# =====================================================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

# =====================================================
# 🔌 MONGODB SETUP (Atlas Hard-coded, escaped)
# =====================================================
MONGO_USER = "albertochristianjoshua_db_user"
MONGO_PASSWORD = "password052304@"  # Replace with your actual password
MONGO_CLUSTER = "cluster1.7vey5de.mongodb.net"
MONGO_DB = "dengue_db"

# Escape special characters in username and password
user_escaped = quote_plus(MONGO_USER)
password_escaped = quote_plus(MONGO_PASSWORD)

# Build the URI
MONGO_URI = f"mongodb+srv://{user_escaped}:{password_escaped}@{MONGO_CLUSTER}/?retryWrites=true&w=majority"

# Connect to MongoDB Atlas
try:
    client = MongoClient(MONGO_URI)
    client.admin.command("ping")
    print("✅ Connected to MongoDB Atlas successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to connect to MongoDB Atlas: {e}")

db = client[MONGO_DB]
collection = db["forecasts"]
alerts_collection = db["alert_recommendations"]


def save_forecast_to_db(city: str, risk_level: str):
    doc = {
        "city": city,
        "risk_level": risk_level,
        "forecast_week": "Next Week",
        "created_at": datetime.datetime.utcnow()
    }
    collection.insert_one(doc)

# =====================================================
# 📌 PYDANTIC MODELS
# =====================================================
feature_fields: Dict[str, Type] = {col: float for col in final_feature_columns}
FeatureInput = create_model("FeatureInput", **feature_fields)

class DengueForecastInput(BaseModel):
    features: List[FeatureInput]

    @field_validator("features")
    @classmethod
    def check_window(cls, v):
        if len(v) != WINDOW:
            raise ValueError(f"Expected {WINDOW} timesteps, got {len(v)}")
        return v

class DengueForecastOutput(BaseModel):
    city: str
    forecast_week: str
    risk_level: str

class CityRequest(BaseModel):
    city: str

class AlertResponse(BaseModel):
    city: str
    risk_level: str
    risk_assessment: str
    actions: List[str]

class AlertRequest(BaseModel):
    city: str
    risk_level: str
    risk_assessment: str
    actions: List[str]


# =====================================================
# 🔧 PREPROCESSING
# =====================================================
def preprocess_input(data: DengueForecastInput):
    try:
        records = [step.model_dump() for step in data.features]
        df = pd.DataFrame(records, columns=final_feature_columns)

        land_area = round(df["LAND AREA"].iloc[0], 2)
        city = land_area_to_city.get(land_area, "Unknown City")

        scaled = scaler.transform(df)
        X = scaled.reshape(1, WINDOW, len(final_feature_columns))

        return X, city
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

# =====================================================
# 🔮 FORECAST ENDPOINT
# =====================================================
@app.post("/forecast", response_model=DengueForecastOutput)
async def forecast_next_week(input_data: DengueForecastInput):
    try:
        X, city = preprocess_input(input_data)
        preds = model.predict(X)
        risk = risk_labels[int(np.argmax(preds[0]))]

        save_forecast_to_db(city, risk)

        return DengueForecastOutput(
            city=city,
            forecast_week="Next Week",
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cases-latest")
async def get_latest_cases_per_city():
    """
    Returns the latest number of cases per city from the latest uploaded CSV.
    """
    try:
        # Get the latest CSV upload
        csv_doc = db["raw_csv_uploads"].find_one(sort=[("uploaded_at", -1)])
        if not csv_doc:
            return []

        # Read CSV
        csv_bytes = bytes(csv_doc["data"])
        df = pd.read_csv(io.BytesIO(csv_bytes))

        # Clean & normalize columns
        df.columns = [c.strip().upper() for c in df.columns]
        df["CITY"] = df["CITY"].str.strip().str.upper()
        df["CASES"] = pd.to_numeric(df["CASES"], errors="coerce")
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
        df = df.dropna(subset=["CITY", "CASES", "DATE"])

        # Take the latest row per city
        latest_rows = df.sort_values("DATE").groupby("CITY", as_index=False).tail(1)

        results = []
        for _, row in latest_rows.iterrows():
            results.append({
                "city": row["CITY"],
                "total_cases": int(row["CASES"]),
                "last_updated": row["DATE"].strftime("%Y-%m-%d")
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest cases: {e}")

@app.get("/api/risk-latest")
async def get_latest_risk_per_city():
    """
    Returns the latest dengue risk per city from the forecasts collection.
    """
    try:
        pipeline = [
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$city",
                "city": {"$first": "$city"},
                "risk_level": {"$first": "$risk_level"},
                "forecast_week": {"$first": "$forecast_week"},
                "created_at": {"$first": "$created_at"}
            }},
            {"$project": {"_id": 0}}
        ]
        return list(collection.aggregate(pipeline))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest risk: {e}")


# =====================================================
# 🗺️ FETCH LATEST FORECAST FOR ALL CITIES
# =====================================================
@app.get("/api/latest-forecast")
async def get_latest_forecast_per_city():
    """
    Returns the latest risk level for all cities.
    """
    try:
        pipeline = [
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$city",
                "city": {"$first": "$city"},
                "risk_level": {"$first": "$risk_level"},
                "forecast_week": {"$first": "$forecast_week"},
                "created_at": {"$first": "$created_at"}
            }},
            {"$project": {"_id": 0}}
        ]
        return list(collection.aggregate(pipeline))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # =====================================================
# 🚨 FETCH ALERT DETAILS (USED BY FRONTEND MODAL)
# =====================================================
@app.get("/alerts", response_model=AlertResponse)
async def get_alert(city: str, risk_level: str):
    """
    Fetch alert recommendations based on city and risk level.
    Used by the frontend alert modal.
    """
    doc = alerts_collection.find_one(
        {
            "city": city.strip().upper(),
            "risk_level": risk_level
        },
        {"_id": 0}
    )

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="No alert recommendations found."
        )

    return doc

from fastapi.responses import JSONResponse
import pandas as pd
import io

@app.get("/api/raw-csv-data")
async def get_raw_csv_data():
    """
    Fetch the latest uploaded CSV from raw_csv_uploads collection and return as JSON
    """
    try:
        # Get the latest uploaded CSV
        doc = db["raw_csv_uploads"].find_one(sort=[("uploaded_at", -1)])
        if not doc:
            return JSONResponse(content={"data": []})

        # Read the CSV bytes into pandas
        csv_bytes = doc["data"]
        df = pd.read_csv(io.BytesIO(csv_bytes))

        # Optional: only return first 50 rows for preview
        preview = df.head(50).to_dict(orient="records")

        return {"data": preview}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



# =====================================================
# 🏙️ FETCH LATEST FORECAST FOR A SINGLE CITY
# =====================================================
@app.post("/get-risk-level")
async def get_risk_level(request: CityRequest):
    """
    Fetch the latest dengue risk level for a specific city.
    """
    city_name = request.city.strip().upper()
    doc = collection.find_one(
        {"city": city_name},
        sort=[("created_at", -1)]
    )
    if not doc:
        raise HTTPException(status_code=404, detail=f"No data found for city: {city_name}")
    
    return {
        "city": city_name,
        "risk_level": doc.get("risk_level", "Unknown"),
        "forecast_week": doc.get("forecast_week", "Unknown")
    }

# =====================================================
# ❤️ HEALTH CHECK
# =====================================================
@app.get("/health")
async def health():
    return {
        "status": "OK",
        "model_loaded": True,
        "mongodb_connected": True,
        "window_size": WINDOW,
        "num_features": len(final_feature_columns),
        "city_detected_from_land_area": True
    }

from fastapi.responses import JSONResponse

from fastapi.responses import JSONResponse

@app.post("/alerts", response_model=AlertResponse)
async def save_alert(alert: AlertRequest):
    """
    Save or update alert recommendations for a city and risk level.
    """
    try:
        # Standardize city and risk_level
        city_name = alert.city.strip().upper()
        risk_level = alert.risk_level.strip()

        # Build document
        doc = {
            "city": city_name,
            "risk_level": risk_level,
            "risk_assessment": alert.risk_assessment.strip(),
            "actions": [a.strip() for a in alert.actions if a.strip()],  # remove empty actions
            "created_at": datetime.datetime.utcnow()
        }

        # Check if alert already exists
        existing = alerts_collection.find_one({"city": city_name, "risk_level": risk_level})

        if existing:
            alerts_collection.update_one({"_id": existing["_id"]}, {"$set": doc})
        else:
            alerts_collection.insert_one(doc)

        # Return cleaned document matching AlertResponse
        response_doc = {
            "city": doc["city"],
            "risk_level": doc["risk_level"],
            "risk_assessment": doc["risk_assessment"],
            "actions": doc["actions"]
        }

        return JSONResponse(status_code=200, content=response_doc)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save alert: {e}")



# =====================================================
# CSV Preprocessing + Recursive Forecasting + Save CSV
# =====================================================
from fastapi import UploadFile, File
from bson import Binary

@app.post("/preprocessing")
async def preprocessing(file: UploadFile = File(...), weeks_ahead: int = 4):
    """
    Accepts a raw CSV file containing multiple cities,
    saves the CSV in MongoDB,
    preprocesses each city independently,
    and returns recursive forecasts for up to `weeks_ahead` weeks.
    Default is 4 weeks, max is 8 weeks.
    """
    try:
        # Cap weeks ahead at 8
        weeks_ahead = min(weeks_ahead, 8)

        # -------------------------------
        # Read CSV
        # -------------------------------
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # -------------------------------
        # Save raw CSV in MongoDB
        # -------------------------------
        db["raw_csv_uploads"].insert_one({
            "filename": file.filename,
            "uploaded_at": datetime.datetime.utcnow(),
            "cities": list(df["LAND AREA"].unique()),
            "data": Binary(contents)
        })

        # -------------------------------
        # Required columns (raw)
        # -------------------------------
        required_cols = [
            "YEAR", "MONTH", "DAY",
            "RAINFALL", "TMAX", "TMIN", "TMEAN",
            "RH", "SUNSHINE",
            "CASES", "POPULATION", "LAND AREA"
        ]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        # -------------------------------
        # Sort by date globally
        # -------------------------------
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
        df = df.sort_values("DATE")

        results = []

        # -------------------------------
        # Process per city
        # -------------------------------
        for land_area, city_df in df.groupby("LAND AREA"):
            city_df = city_df.sort_values("DATE").reset_index(drop=True)

            # Incidence per 100k
            city_df["CASES"] = (city_df["CASES"] / city_df["POPULATION"]) * 100000

            # Forecast container for recursive steps
            city_forecasts = []

            # Copy for recursive predictions
            temp_df = city_df.copy()

            for week in range(1, weeks_ahead + 1):
                # -------------------------------
                # Lag features
                # -------------------------------
                for lag in range(1, WINDOW + 1):
                    temp_df[f"CASES_lag{lag}"] = temp_df["CASES"].shift(lag)
                    temp_df[f"RAINFALL_lag{lag}"] = temp_df["RAINFALL"].shift(lag)
                    temp_df[f"TMAX_lag{lag}"] = temp_df["TMAX"].shift(lag)
                    temp_df[f"TMIN_lag{lag}"] = temp_df["TMIN"].shift(lag)
                    temp_df[f"TMEAN_lag{lag}"] = temp_df["TMEAN"].shift(lag)
                    temp_df[f"RH_lag{lag}"] = temp_df["RH"].shift(lag)
                    temp_df[f"SUNSHINE_lag{lag}"] = temp_df["SUNSHINE"].shift(lag)

                # -------------------------------
                # Rolling features
                # -------------------------------
                temp_df["CASES_roll2_mean"] = temp_df["CASES"].rolling(2).mean()
                temp_df["CASES_roll4_mean"] = temp_df["CASES"].rolling(4).mean()
                temp_df["CASES_roll2_sum"] = temp_df["CASES"].rolling(2).sum()
                temp_df["CASES_roll4_sum"] = temp_df["CASES"].rolling(4).sum()

                temp_df["RAINFALL_roll2_mean"] = temp_df["RAINFALL"].rolling(2).mean()
                temp_df["RAINFALL_roll4_mean"] = temp_df["RAINFALL"].rolling(4).mean()
                temp_df["RAINFALL_roll2_sum"] = temp_df["RAINFALL"].rolling(2).sum()
                temp_df["RAINFALL_roll4_sum"] = temp_df["RAINFALL"].rolling(4).sum()

                temp_df["TMEAN_roll2_mean"] = temp_df["TMEAN"].rolling(2).mean()
                temp_df["TMEAN_roll4_mean"] = temp_df["TMEAN"].rolling(4).mean()
                temp_df["TMEAN_roll2_sum"] = temp_df["TMEAN"].rolling(2).sum()
                temp_df["TMEAN_roll4_sum"] = temp_df["TMEAN"].rolling(4).sum()

                temp_df["RH_roll2_mean"] = temp_df["RH"].rolling(2).mean()
                temp_df["RH_roll4_mean"] = temp_df["RH"].rolling(4).mean()
                temp_df["RH_roll2_sum"] = temp_df["RH"].rolling(2).sum()
                temp_df["RH_roll4_sum"] = temp_df["RH"].rolling(4).sum()

                # Drop NaNs
                temp_df_clean = temp_df.dropna().reset_index(drop=True)
                if len(temp_df_clean) < WINDOW:
                    break

                # -------------------------------
                # Extract last WINDOW and predict
                # -------------------------------
                window_df = temp_df_clean.iloc[-WINDOW:][final_feature_columns]
                scaled = scaler.transform(window_df)
                X = scaled.reshape(1, WINDOW, len(final_feature_columns))
                preds = model.predict(X)
                risk = risk_labels[int(np.argmax(preds[0]))]
                city_name = land_area_to_city.get(round(land_area, 2), "Unknown City")

                if week == 1:
                    save_forecast_to_db(city_name, risk)


                # Forecast date = last date + 7 days
                last_date = temp_df["DATE"].iloc[-1]
                forecast_date = last_date + pd.Timedelta(days=7)

                city_forecasts.append({
                    "city": city_name,
                    "week": week,
                    "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                    "risk_level": risk
                })

                # Append placeholder row for next week
                new_row = temp_df.iloc[-1:].copy()
                new_row["DATE"] = forecast_date
                new_row["CASES"] = (100000 / new_row["POPULATION"].values[0]) * 0
                temp_df = pd.concat([temp_df, new_row], ignore_index=True)

            results.extend(city_forecasts)

        if not results:
            raise HTTPException(
                status_code=400,
                detail="No city had sufficient data for forecasting."
            )

        return {
            "status": "OK",
            "num_cities": len(set(r["city"] for r in results)),
            "weeks_forecasted": weeks_ahead,
            "forecasts": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CSV preprocessing failed: {e}"
        )
    
@app.get("/api/alerts")
async def get_weekly_alerts_from_latest_csv():
    """
    Generates weekly dengue alerts per city based on the latest uploaded CSV,
    and enriches them with risk assessment & recommended actions
    from alert_recommendations collection.
    """
    try:
        # 1. Get latest CSV upload
        csv_doc = db["raw_csv_uploads"].find_one(sort=[("uploaded_at", -1)])
        if not csv_doc:
            return []

        # 2. Read CSV
        csv_bytes = bytes(csv_doc["data"])
        df = pd.read_csv(io.BytesIO(csv_bytes))

        # 3. Clean & normalize
        df.columns = [c.strip().upper() for c in df.columns]
        df["CITY"] = df["CITY"].str.strip().str.upper()
        df["CASES"] = pd.to_numeric(df["CASES"], errors="coerce")

        df["DATE"] = pd.to_datetime(
            df[["YEAR", "MONTH", "DAY"]],
            errors="coerce"
        )

        df = df.dropna(subset=["CITY", "CASES", "DATE"])

        alerts = []

        # --- Get latest risk per city from forecasts collection ---
        pipeline = [
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$city",
                "city": {"$first": "$city"},
                "risk_level": {"$first": "$risk_level"},
                "forecast_week": {"$first": "$forecast_week"},
                "created_at": {"$first": "$created_at"}
            }},
            {"$project": {"_id": 0}}
        ]

        latest_risks = {
            r["city"].upper(): r["risk_level"]
            for r in db["forecasts"].aggregate(pipeline)
        }

        # 4. Per-city aggregation
        for city, city_df in df.groupby("CITY"):

            city_df = city_df.sort_values("DATE")
            latest_row = city_df.iloc[-1]
            latest_date = latest_row["DATE"]

            current_start = latest_date - pd.Timedelta(days=6)
            previous_start = current_start - pd.Timedelta(days=7)
            previous_end = current_start - pd.Timedelta(days=1)

            current_cases = city_df.loc[
                (city_df["DATE"] >= current_start) &
                (city_df["DATE"] <= latest_date),
                "CASES"
            ].sum()

            previous_cases = city_df.loc[
                (city_df["DATE"] >= previous_start) &
                (city_df["DATE"] <= previous_end),
                "CASES"
            ].sum()

            if previous_cases == 0:
                percent_change = 100.0 if current_cases > 0 else 0.0
            else:
                percent_change = ((current_cases - previous_cases) / previous_cases) * 100

            alert_level = latest_risks.get(city, "Low")

            # 🔥 NEW: Fetch alert recommendations
            alert_doc = alerts_collection.find_one(
                {
                    "city": city,
                    "risk_level": alert_level
                },
                {"_id": 0}
            )

            alerts.append({
                "city": city,
                "current_week_cases": int(current_cases),
                "previous_week_cases": int(previous_cases),
                "percent_change": round(percent_change, 1),
                "alert_level": alert_level,
                "risk_assessment": alert_doc.get("risk_assessment", "No assessment available.")
                    if alert_doc else "No assessment available.",
                "actions": alert_doc.get("actions", []) if alert_doc else [],
                "last_updated": latest_row["DATE"].strftime("%Y-%m-%d")
            })

        return alerts

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate alerts: {e}"
        )

