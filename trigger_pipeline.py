import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()

# --- Snowflake credentials (from GitHub Secrets or .env locally) ---
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "MLOPS_WH")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "POWERCONSUMPTION")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")

def trigger_pipeline():
    print("🔗 Connecting to Snowflake...")
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    cur = conn.cursor()
    print("✅ Connection successful.")

    # --- Trigger the first task manually ---
    print("🚀 Triggering ML pipeline...")
    cur.execute("EXECUTE TASK POWERCONSUMPTION.PUBLIC.TASK_1_DATA_INGESTION;")
    print("✅ ML pipeline triggered successfully.")

    # Optionally wait/check for success
    cur.execute("""
        SELECT state, scheduled_time, completed_time, error_message
        FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY())
        WHERE name = 'TASK_1_DATA_INGESTION'
        ORDER BY scheduled_time DESC LIMIT 1;
    """)
    print("📊 Task execution result:", cur.fetchall())

    cur.close()
    conn.close()
    print("🔒 Connection closed.")

if __name__ == "__main__":
    trigger_pipeline()
