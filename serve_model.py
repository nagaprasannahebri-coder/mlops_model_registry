import os
from datetime import datetime
from dotenv import load_dotenv
import snowflake.connector
import uvicorn
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# --- Load environment variables ---
load_dotenv()

# --- Snowflake Configuration ---
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "MLOPS_WH")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "POWERCONSUMPTION")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
SNOWFLAKE_STAGE = os.getenv("SNOWFLAKE_STAGE", "ML_MODELS_STAGE")
LOCAL_MODEL_DIR = "./models"


# --- Function to Download Latest Model from Snowflake ---
def download_latest_onnx_model():
    """
    Connects to Snowflake, finds the latest ONNX model on the stage,
    downloads it to a local directory, and returns the path.
    """
    conn = None
    try:
        print("🔗 Connecting to Snowflake...")
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        print("✅ Connection successful.")

        cur = conn.cursor()

        # Ensure correct database/schema context
        print(f"📂 Setting context to {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}")
        cur.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
        cur.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")

        # Fully qualified stage path
        full_stage_path = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{SNOWFLAKE_STAGE}"

        print(f"📂 Listing files in stage '@{full_stage_path}'...")
        cur.execute(f"LIST @{full_stage_path}")
        all_files = cur.fetchall()

        # Filter .onnx files and pick the most recent
        onnx_files = []
        for file_row in all_files:
            file_name, _, _, last_modified_str = file_row
            if file_name.endswith(".onnx"):
                last_modified_dt = datetime.strptime(
                    last_modified_str, "%a, %d %b %Y %H:%M:%S %Z"
                )
                onnx_files.append((file_name, last_modified_dt))

        if not onnx_files:
            print("⚠️ No .onnx files found on stage.")
            return None

        onnx_files.sort(key=lambda x: x[1], reverse=True)
        latest_file_full_path = onnx_files[0][0]
        latest_file_name = os.path.basename(latest_file_full_path)
        print(f"✅ Found latest model: {latest_file_name}")

        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        local_directory_path = os.path.abspath(LOCAL_MODEL_DIR).replace("\\", "/")
        get_command = f"GET @{full_stage_path}/{latest_file_name} file://{local_directory_path}"

        print(f"⬇️ Executing download command: {get_command}")
        cur.execute(get_command)

        final_path = os.path.join(local_directory_path, latest_file_name)
        print(f"✅ Model downloaded to: {final_path}")
        return final_path

    except Exception as e:
        print(f"❌ Error during model download: {e}")
        return None
    finally:
        if conn:
            conn.close()
            print("🔒 Connection closed.")


# --- Input Schema for Prediction Endpoint ---
class InputFeatures(BaseModel):
    TEMPERATURE: float
    HUMIDITY: float
    WINDSPEED: float
    GENERALDIFFUSEFLOWS: float
    DIFFUSEFLOWS: float
    HOUR: int
    DAYOFWEEK: int
    QUARTER: int
    MONTH: int
    DAYOFYEAR: int
    POWER_LAG_1: float
    POWER_LAG_144: float
    POWER_ROLLING_MEAN_6: float
    POWER_ROLLING_MEAN_24: float


# --- FastAPI App ---
app = FastAPI(title="Power Consumption Predictor API", version="1.0")

# Globals for model session
ort_session = None
input_name = None


@app.on_event("startup")
def startup_event():
    """
    On startup, download the latest model and load it into memory.
    """
    global ort_session, input_name
    print("🚀 Starting FastAPI Server — initializing model...")
    local_model_path = download_latest_onnx_model()

    if local_model_path and os.path.exists(local_model_path):
        try:
            ort_session = ort.InferenceSession(local_model_path, providers=["CPUExecutionProvider"])
            input_name = ort_session.get_inputs()[0].name
            print("✅ Model loaded successfully and ready for predictions.")
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
    else:
        print("⚠️ No model loaded. The API will still run, but /predict will return an error.")


@app.get("/", summary="API Health Check")
def health_check():
    """
    Health check endpoint to verify server and model status.
    """
    status = "loaded" if ort_session else "not loaded"
    return {"status": "ok", "model_status": status}


@app.post("/predict", summary="Predict Power Consumption")
def predict(features: InputFeatures):
    """
    Receives JSON input and returns power consumption prediction.
    """
    if not ort_session:
        return {"error": "Model not loaded. Please check server logs."}

    ordered_features = [
        features.TEMPERATURE, features.HUMIDITY, features.WINDSPEED,
        features.GENERALDIFFUSEFLOWS, features.DIFFUSEFLOWS, features.HOUR,
        features.DAYOFWEEK, features.QUARTER, features.MONTH, features.DAYOFYEAR,
        features.POWER_LAG_1, features.POWER_LAG_144,
        features.POWER_ROLLING_MEAN_6, features.POWER_ROLLING_MEAN_24
    ]

    input_array = np.array(ordered_features, dtype=np.float32).reshape(1, -1)
    prediction = ort_session.run(None, {input_name: input_array})[0]
    predicted_value = float(prediction[0][0])
    return {"predicted_power_consumption": predicted_value}


if __name__ == "__main__":
    print("✅ FastAPI running on http://0.0.0.0:8055")
    uvicorn.run(app, host="0.0.0.0", port=8055)
