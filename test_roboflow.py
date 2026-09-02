"""
Quick test script to verify Roboflow configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Roboflow Configuration ===")
print(f"API Key: {os.getenv('ROBOFLOW_API_KEY')[:20]}..." if os.getenv('ROBOFLOW_API_KEY') else "NOT SET")
print(f"Workspace: {os.getenv('ROBOFLOW_WORKSPACE')}")
print(f"Workflow ID: {os.getenv('ROBOFLOW_WORKFLOW_ID')}")
print(f"Classes: {os.getenv('ROBOFLOW_CLASSES')}")
print(f"Confidence Threshold: {os.getenv('BUILDING_CONFIDENCE_THRESHOLD')}")

print("\n=== Testing Roboflow Connection ===")
try:
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.getenv('ROBOFLOW_API_KEY')
    ).configure(InferenceConfiguration(
        api_key_transport="header"
    ))
    
    print("✅ Roboflow client initialized successfully")
    print("\nNOTE: You need to provide the correct ROBOFLOW_WORKFLOW_ID in .env")
    print("Check your Roboflow dashboard for the workflow ID")
    
except Exception as e:
    print(f"❌ Error: {e}")
