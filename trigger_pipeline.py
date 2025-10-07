import snowflake.connector
import os
from dotenv import load_dotenv  

# Load environment variables from the .env file
load_dotenv() 

SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = "MLOPS_WH"
SNOWFLAKE_DATABASE = "POWERCONSUMPTION"
SNOWFLAKE_SCHEMA = "PUBLIC"

def trigger_snowflake_pipeline():
    """
    Connects to Snowflake and executes the root task to trigger the ML pipeline.
    """
    conn = None
    try:
        print("Connecting to Snowflake...")
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT
        )
        print("Connection successful.")
        
        cur = conn.cursor()
        
        commands = [
            f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE};",
            f"USE DATABASE {SNOWFLAKE_DATABASE};",
            f"USE SCHEMA {SNOWFLAKE_SCHEMA};",
            "EXECUTE TASK TASK_1_DATA_INGESTION;"
        ]
        
        for cmd in commands:
            print(f"Executing: {cmd}")
            cur.execute(cmd)
        
        print("\nPipeline successfully triggered in Snowflake!")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()
            print("\nConnection closed.")

# Main execution block
if __name__ == "__main__":
    if not all([SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT]):
        print("Error: Could not find credentials. Make sure you have a .env file with SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, and SNOWFLAKE_ACCOUNT.")
    else:
        trigger_snowflake_pipeline()