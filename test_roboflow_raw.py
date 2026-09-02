"""
Raw test of Roboflow workflow to see actual response
"""
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image
import json

load_dotenv()

print("=== Testing Roboflow Workflow (Raw) ===\n")

# Configuration
api_key = os.getenv("ROBOFLOW_API_KEY")
workspace = os.getenv("ROBOFLOW_WORKSPACE")
workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID")
classes = os.getenv("ROBOFLOW_CLASSES")

print(f"Workspace: {workspace}")
print(f"Workflow ID: {workflow_id}")
print(f"Classes: {classes}\n")

# Initialize client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
).configure(InferenceConfiguration(
    api_key_transport="header"
))

print("✅ Client initialized\n")

# Test with image
test_image = "data/raw/test_satellite_image.png"
if not os.path.exists(test_image):
    test_image = "satelite-img.png"

if not os.path.exists(test_image):
    print("❌ No test image found")
    exit(1)

print(f"📸 Using image: {test_image}\n")

# Load and display image info
img = Image.open(test_image)
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}\n")

# Run workflow
print("🔍 Running workflow...\n")
try:
    result = client.run_workflow(
        workspace_name=workspace,
        workflow_id=workflow_id,
        images={"image": test_image},
        parameters={"classes": classes},
        use_cache=True
    )
    
    print("✅ Workflow completed successfully!\n")
    print("=== RAW RESULT ===")
    print(f"Type: {type(result)}")
    
    if isinstance(result, list) and len(result) > 0:
        print(f"\nFirst element keys: {result[0].keys()}")
        print(f"\nFirst element (without image data):")
        for key, value in result[0].items():
            if key != 'annotated_image':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: <base64 image data>")
    
    print(f"\nFull JSON Result (truncated):")
    result_str = json.dumps(result, indent=2, default=str)
    if len(result_str) > 2000:
        print(result_str[:2000] + "...[truncated]")
    else:
        print(result_str)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
