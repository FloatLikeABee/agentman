#!/usr/bin/env python3
"""
Test script for the RAG System
"""

import requests
import json
import time
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test basic API connection"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ API connection successful")
            return True
        else:
            print(f"❌ API connection failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the backend is running.")
        return False

def test_system_status():
    """Test system status endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            status = response.json()
            print("✅ System status retrieved")
            print(f"   Ollama connected: {status.get('ollama_connected', False)}")
            print(f"   Available models: {len(status.get('available_models', []))}")
            print(f"   RAG collections: {len(status.get('rag_collections', []))}")
            print(f"   Active agents: {len(status.get('active_agents', []))}")
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_rag_functionality():
    """Test RAG functionality"""
    try:
        # Test data validation
        test_data = {
            "name": "test_collection",
            "description": "Test collection for validation",
            "format": "json",
            "content": '{"test": "data", "number": 42}',
            "tags": ["test", "validation"],
            "metadata": {"source": "test"}
        }
        
        response = requests.post(f"{API_BASE_URL}/rag/validate", json=test_data)
        if response.status_code == 200:
            validation = response.json()
            print("✅ Data validation working")
            print(f"   Valid: {validation.get('is_valid', False)}")
            print(f"   Record count: {validation.get('record_count', 0)}")
        else:
            print(f"❌ Data validation failed: {response.status_code}")
            return False
        
        # Test adding data
        response = requests.post(f"{API_BASE_URL}/rag/collections/test_collection/data", json=test_data)
        if response.status_code == 200:
            print("✅ Data addition working")
        else:
            print(f"❌ Data addition failed: {response.status_code}")
            return False
        
        # Test querying
        response = requests.post(f"{API_BASE_URL}/rag/collections/test_collection/query", 
                               params={"query": "test data", "n_results": 5})
        if response.status_code == 200:
            results = response.json()
            print("✅ RAG query working")
            print(f"   Results: {len(results.get('results', []))}")
        else:
            print(f"❌ RAG query failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ RAG functionality error: {e}")
        return False

def test_agent_creation():
    """Test agent creation"""
    try:
        # Get available models
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            models = response.json()
            if models:
                model_name = models[0]['name']
            else:
                print("⚠️  No models available, skipping agent test")
                return True
        else:
            print("❌ Could not get models")
            return False
        
        # Create test agent
        agent_config = {
            "name": "Test Agent",
            "description": "Test agent for validation",
            "agent_type": "rag",
            "model_name": model_name,
            "temperature": 0.7,
            "max_tokens": 2048,
            "rag_collections": ["test_collection"],
            "tools": [],
            "system_prompt": "You are a helpful test agent.",
            "is_active": True
        }
        
        response = requests.post(f"{API_BASE_URL}/agents", json=agent_config)
        if response.status_code == 200:
            result = response.json()
            agent_id = result.get('agent_id')
            print("✅ Agent creation working")
            print(f"   Agent ID: {agent_id}")
            
            # Test running agent
            run_data = {
                "query": "Hello, this is a test query.",
                "agent_id": agent_id,
                "context": None
            }
            
            response = requests.post(f"{API_BASE_URL}/agents/{agent_id}/run", json=run_data)
            if response.status_code == 200:
                result = response.json()
                print("✅ Agent execution working")
                print(f"   Response length: {len(result.get('response', ''))}")
            else:
                print(f"❌ Agent execution failed: {response.status_code}")
            
            return True
        else:
            print(f"❌ Agent creation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Agent functionality error: {e}")
        return False

def test_tools():
    """Test tool functionality"""
    try:
        response = requests.get(f"{API_BASE_URL}/tools")
        if response.status_code == 200:
            tools = response.json()
            print("✅ Tools listing working")
            print(f"   Available tools: {len(tools)}")
            for tool in tools:
                print(f"     - {tool.get('name')} ({tool.get('tool_type')})")
            return True
        else:
            print(f"❌ Tools listing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Tools functionality error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing RAG System...")
    print("=" * 50)
    
    tests = [
        ("API Connection", test_api_connection),
        ("System Status", test_system_status),
        ("RAG Functionality", test_rag_functionality),
        ("Agent Creation", test_agent_creation),
        ("Tools", test_tools),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RAG system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 