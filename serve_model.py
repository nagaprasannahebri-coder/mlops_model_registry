import os
from datetime import datetime
from dotenv import load_dotenv

# --- Part 1: Snowflake and Model Download ---
import snowflake.connector

# --- Part 2: Model Serving with FastAPI ---
import uvicorn
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv()


# Connection Parameters
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = "MLOPS_WH"
SNOWFLAKE_DATABASE = "POWERCONSUMPTION"
SNOWFLAKE_SCHEMA = "PUBLIC"
SNOWFLAKE_STAGE = "ML_MODELS_STAGE"
LOCAL_MODEL_DIR = "./models"

def download_latest_onnx_model():
    """
    Connects to Snowflake, finds the latest ONNX model on the stage,
    and downloads it to a local directory, returning the path.
    """
    conn = None
    try:
        print("Connecting to Snowflake...")
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        print("Connection successful.")
        cur = conn.cursor()
        
        print(f"Listing files in stage '@{SNOWFLAKE_STAGE}'...")
        cur.execute(f"LIST @{SNOWFLAKE_STAGE}")
        
        all_files = cur.fetchall()
        
        onnx_files = []
        for file_row in all_files:
            file_name, _, _, last_modified_str = file_row
            if file_name.endswith('.onnx'):
                last_modified_dt = datetime.strptime(last_modified_str, '%a, %d %b %Y %H:%M:%S %Z')
                onnx_files.append((file_name, last_modified_dt))

        if not onnx_files:
            print("Error: No .onnx files found on the stage.")
            return None

        onnx_files.sort(key=lambda x: x[1], reverse=True)
        latest_file_full_path = onnx_files[0][0]
        latest_file_name = os.path.basename(latest_file_full_path)
        
        print(f"Found latest model: {latest_file_name}")
        
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        local_directory_path = os.path.abspath(LOCAL_MODEL_DIR).replace('\\', '/')
        get_command = f"GET @{SNOWFLAKE_STAGE}/{latest_file_name} file://{local_directory_path}"
        
        print(f"Executing download command: {get_command}")
        cur.execute(get_command)
        
        final_path = os.path.join(local_directory_path, latest_file_name)
        print(f"\nSuccess! Model downloaded to: {final_path}")
        return final_path

    except Exception as e:
        print(f"An error occurred during download: {e}")
        return None
        
    finally:
        if conn:
            conn.close()
            print("\nConnection closed.")


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
    
    # Add example data for the API documentation
    class Config:
        schema_extra = {
            "example": {
                "TEMPERATURE": 10.5, "HUMIDITY": 80.0, "WINDSPEED": 0.085,
                "GENERALDIFFUSEFLOWS": 0.06, "DIFFUSEFLOWS": 0.1, "HOUR": 14,
                "DAYOFWEEK": 4, "QUARTER": 4, "MONTH": 12, "DAYOFYEAR": 340,
                "POWER_LAG_1": 35000, "POWER_LAG_144": 28000,
                "POWER_ROLLING_MEAN_6": 34500, "POWER_ROLLING_MEAN_24": 33000
            }
        }

# Global variables to hold the loaded model session and its input name
ort_session = None
input_name = None

# Create the FastAPI app instance
app = FastAPI(title="Power Consumption Predictor API", version="1.0")

@app.on_event("startup")
def startup_event():
    """
    On startup, download the model and load it into memory.
    This ensures the model is ready before the first request comes in.
    """
    global ort_session, input_name
    print("--- Server starting up: Initializing model ---")
    
    local_model_path = download_latest_onnx_model()
    
    if local_model_path and os.path.exists(local_model_path):
        try:
            print(f"Loading ONNX model from {local_model_path}...")
            ort_session = ort.InferenceSession(local_model_path, providers=["CPUExecutionProvider"])
            input_name = ort_session.get_inputs()[0].name
            print("Model loaded successfully and is ready for predictions.")
        except Exception as e:
            print(f"FATAL: Error loading ONNX model: {e}")
    else:
        print("FATAL: Could not download or find model. The '/predict' endpoint will not work.")

@app.get("/", summary="API Health Check")
def health_check():
    """A simple endpoint to confirm the API is running and if the model is loaded."""
    model_status = "loaded" if ort_session else "not loaded"
    return {"status": "ok", "model_status": model_status}

@app.post("/predict", summary="Predict Power Consumption")
def predict(features: InputFeatures):
    """
    Receives input features in a JSON body and returns a power consumption prediction.
    """
    if not ort_session:
        return {"error": "Model is not loaded. Please check server startup logs for errors."}, 500

    # The model expects features in the f0, f1... order.
    # ensure the input is ordered correctly.
    ordered_features = [
        features.TEMPERATURE, features.HUMIDITY, features.WINDSPEED,
        features.GENERALDIFFUSEFLOWS, features.DIFFUSEFLOWS, features.HOUR,
        features.DAYOFWEEK, features.QUARTER, features.MONTH, features.DAYOFYEAR,
        features.POWER_LAG_1, features.POWER_LAG_144,
        features.POWER_ROLLING_MEAN_6, features.POWER_ROLLING_MEAN_24
    ]

    # Convert the list to a NumPy array with the correct shape and type
    input_array = np.array(ordered_features, dtype=np.float32).reshape(1, -1)

    # Run inference
    prediction = ort_session.run(None, {input_name: input_array})[0]
    
    # Extract the single float value from the model's output array
    predicted_value = float(prediction[0][0])
    
    return {"predicted_power_consumption": predicted_value}


if __name__ == "__main__":
    # The `startup_event` function will be called automatically when the server starts.
    print("--- Starting FastAPI Server ---")
    print("Access the API at http://12.0.0.1:8055")
    print("API documentation available at http://127.0.0.1:8055/docs")
    

    uvicorn.run(app, host="0.0.0.0", port=8055)
