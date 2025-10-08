import snowflake.connector
import os
from dotenv import load_dotenv
import time

load_dotenv()

# --- Snowflake credentials (from GitHub Secrets or .env locally) ---
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "MLOPS_WH")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "POWERCONSUMPTION")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
SNOWFLAKE_STAGE = os.getenv("SNOWFLAKE_STAGE", "ML_MODELS_STAGE")

# --- Define pipeline steps (Stored Procedures) ---
PIPELINE_STEPS = [
    "CALL POWER_CONSUMPTION_DATA_INGESTION();",
    "CALL POWER_CONSUMPTION_FEATURE_ENGINEERING();",
    "CALL POWER_CONSUMPTION_MODEL_TRAINING();",
    "CALL POWER_CONSUMPTION_MODEL_EVALUATION();"
]


def trigger_pipeline():
    print("🔗 Connecting to Snowflake...")

    # Establish connection
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    cur = conn.cursor()
    print("✅ Connection successful.\n")

    # --- Execute each stored procedure step in order ---
    for idx, step in enumerate(PIPELINE_STEPS, start=1):
        step_name = step.split("(")[0].replace("CALL ", "").replace("();", "")
        print(f"🚀 Step {idx}: Executing {step_name} ...")
        try:
            cur.execute(step)
            result = cur.fetchall()
            print(f"✅ {step_name} executed successfully. Result: {result}\n")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            break

    # --- Verify stage contents ---
    print("🧠 Checking model stage for latest uploads...")
    try:
        cur.execute(f"LIST @{SNOWFLAKE_STAGE};")
        stage_files = cur.fetchall()

        if not stage_files:
            print("⚠️ No models found in the stage.")
        else:
            print(f"📦 Found {len(stage_files)} model files in stage @{SNOWFLAKE_STAGE}.\n")

            # Sort files by timestamp (column 3 = last_modified)
            latest_file = sorted(stage_files, key=lambda x: x[2], reverse=True)[0]
            print(f"✅ Latest model file: {latest_file[0]}")
            print(f"🕒 Last modified: {latest_file[2]}")
            print(f"📏 Size: {round(latest_file[1] / 1024, 2)} KB\n")

    except Exception as e:
        print(f"❌ Error while checking model stage: {e}")

    # --- Close connections ---
    cur.close()
    conn.close()
    print("🔒 Connection closed.")
    print("🏁 ML pipeline completed successfully!")


if __name__ == "__main__":
    trigger_pipeline()
