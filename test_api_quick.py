#!/usr/bin/env python3
"""
Quick API Test Script
Tests the SatQuery API with the sample image
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is healthy!")
            print(f"  Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to API at {API_URL}")
        print("  Make sure the server is running (start_server.bat)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_tools():
    """Test tools listing"""
    print("\n" + "="*60)
    print("TEST 2: List Available Tools")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/api/v1/tools", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data['count']} tools:")
            for tool in data['tools']:
                print(f"  - {tool}")
            return True
        else:
            print(f"✗ Failed to list tools: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_query():
    """Test query processing"""
    print("\n" + "="*60)
    print("TEST 3: Process Query")
    print("="*60)
    
    image_path = 'data/raw/test_satellite_image.png'
    query = "How many buildings are visible in this image?"
    
    print(f"Image: {image_path}")
    print(f"Query: {query}")
    print("\nSending request...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': ('test_image.png', f, 'image/png')}
            data = {'query': query}
            
            response = requests.post(
                f"{API_URL}/api/v1/query",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Query processed successfully!")
            print(f"\nSession ID: {result['session_id']}")
            print(f"Query ID: {result['query_id']}")
            print(f"Status: {result['status']}")
            print(f"\nAnswer: {result['results']['answer']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            if result.get('audit_log', {}).get('selected_tools'):
                print(f"\nTools Used:")
                for tool in result['audit_log']['selected_tools']:
                    print(f"  - {tool.get('tool_name', 'Unknown')}")
            
            return True
        else:
            print(f"✗ Query failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except FileNotFoundError:
        print(f"✗ Test image not found: {image_path}")
        print("  Run: python create_test_image.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SatQuery API Quick Test")
    print("="*60)
    print(f"Testing API at: {API_URL}")
    
    results = []
    
    # Test 1: Health
    results.append(("Health Check", test_health()))
    
    # Test 2: Tools
    results.append(("List Tools", test_tools()))
    
    # Test 3: Query
    results.append(("Process Query", test_query()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()
