#!/usr/bin/env python3
"""
Test script for AI-Galaxy Idea Workflow.

This script submits test ideas to the AI-Galaxy API and monitors their progress
through the agent processing pipeline.
"""

import asyncio
import json
import time
from datetime import datetime
import aiohttp
import websockets

API_BASE_URL = "http://localhost:8080"
WS_URL = "ws://localhost:8080"

# Sample test ideas
TEST_IDEAS = [
    {
        "title": "AI-Powered Code Review Assistant",
        "description": "Create an intelligent code review assistant that uses machine learning to automatically detect potential bugs, security vulnerabilities, and suggest code improvements. The system should integrate with Git repositories and provide real-time feedback to developers.",
        "priority": "high",
        "tags": ["machine-learning", "code-analysis", "developer-tools", "automation"],
        "metadata": {
            "estimated_complexity": "high",
            "target_languages": ["python", "javascript", "go"],
            "integration_requirements": ["git", "github", "gitlab"]
        },
        "department_hint": "machine_learning"
    },
    {
        "title": "Real-time Collaborative Whiteboard",
        "description": "Build a web-based collaborative whiteboard application with real-time synchronization, drawing tools, and team collaboration features. Support for multiple users, version history, and export capabilities.",
        "priority": "medium",
        "tags": ["web-development", "real-time", "collaboration", "frontend"],
        "metadata": {
            "estimated_complexity": "medium",
            "technology_stack": ["react", "websockets", "canvas"],
            "user_capacity": "up to 50 concurrent users"
        },
        "department_hint": "web_development"
    },
    {
        "title": "Automated Data Pipeline for Analytics",
        "description": "Create an automated data pipeline that extracts data from multiple sources, transforms it according to business rules, and loads it into a data warehouse for analytics and reporting.",
        "priority": "high",
        "tags": ["data-engineering", "etl", "analytics", "automation"],
        "metadata": {
            "estimated_complexity": "high",
            "data_sources": ["databases", "apis", "files"],
            "target_warehouse": "snowflake"
        },
        "department_hint": "data_engineering"
    }
]


async def check_api_health():
    """Check if the API is healthy and ready."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"OK API Health: {health_data['status']}")
                    print(f"   Uptime: {health_data['uptime_seconds']:.1f} seconds")
                    print(f"   Services: {health_data['services']}")
                    return True
                else:
                    print(f"ERROR API Health Check Failed: {response.status}")
                    return False
    except Exception as e:
        print(f"ERROR Cannot connect to API: {str(e)}")
        return False


async def submit_idea(session, idea):
    """Submit an idea to the API."""
    try:
        async with session.post(f"{API_BASE_URL}/api/v1/ideas/", json=idea) as response:
            if response.status == 200:
                idea_data = await response.json()
                print(f"✅ Idea submitted successfully!")
                print(f"   ID: {idea_data['id']}")
                print(f"   Title: {idea_data['title']}")
                print(f"   Status: {idea_data['status']}")
                return idea_data
            else:
                error_text = await response.text()
                print(f"❌ Failed to submit idea: {response.status}")
                print(f"   Error: {error_text}")
                return None
    except Exception as e:
        print(f"❌ Error submitting idea: {str(e)}")
        return None


async def get_idea_status(session, idea_id):
    """Get the current status of an idea."""
    try:
        async with session.get(f"{API_BASE_URL}/api/v1/ideas/{idea_id}") as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"❌ Failed to get idea status: {response.status}")
                return None
    except Exception as e:
        print(f"❌ Error getting idea status: {str(e)}")
        return None


async def monitor_workflow_websocket(idea_id):
    """Monitor workflow updates via WebSocket."""
    try:
        uri = f"{WS_URL}/ws/workflows"
        async with websockets.connect(uri) as websocket:
            print(f"🔌 Connected to workflow WebSocket")
            
            # Subscribe to idea updates
            subscribe_message = {
                "type": "subscribe_idea",
                "idea_id": idea_id
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Listen for updates
            timeout_count = 0
            while timeout_count < 30:  # 5 minutes timeout
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    
                    print(f"📡 WebSocket Update: {data['type']}")
                    if data['type'] == 'workflow_event':
                        print(f"   Event: {data['data']['event']}")
                        print(f"   Timestamp: {data['data']['timestamp']}")
                        
                        if data['data']['event'] == 'transition_processed':
                            print(f"   State Change: {data['data']['from_state']} → {data['data']['to_state']}")
                            print(f"   Agent: {data['data']['agent_name']}")
                    
                    timeout_count = 0  # Reset timeout on successful message
                    
                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(f"⏱️  Waiting for workflow updates... ({timeout_count}/30)")
                    
    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")


async def get_agent_status(session):
    """Get current agent status."""
    try:
        async with session.get(f"{API_BASE_URL}/api/v1/agents/") as response:
            if response.status == 200:
                agent_data = await response.json()
                print(f"🤖 Agent Status ({agent_data['total_count']} total):")
                print(f"   Active: {agent_data['active_count']}")
                print(f"   Inactive: {agent_data['inactive_count']}")
                
                for agent in agent_data['agents'][:5]:  # Show first 5 agents
                    print(f"   • {agent['name']}: {agent['status']} ({agent['total_tasks_completed']} tasks)")
                
                return agent_data
            else:
                print(f"❌ Failed to get agent status: {response.status}")
                return None
    except Exception as e:
        print(f"❌ Error getting agent status: {str(e)}")
        return None


async def main():
    """Main test function."""
    print("AI-Galaxy Idea Workflow Test")
    print("=" * 50)
    
    # Check API health
    if not await check_api_health():
        print("ERROR: API is not available. Please start AI-Galaxy first.")
        return
    
    async with aiohttp.ClientSession() as session:
        # Get initial agent status
        print("\n📊 Current System Status:")
        await get_agent_status(session)
        
        # Submit a test idea
        print(f"\n💡 Submitting Test Idea...")
        test_idea = TEST_IDEAS[0]  # Use the first test idea
        idea_data = await submit_idea(session, test_idea)
        
        if not idea_data:
            print("❌ Failed to submit idea. Exiting.")
            return
        
        idea_id = idea_data['id']
        
        # Start WebSocket monitoring in background
        websocket_task = asyncio.create_task(monitor_workflow_websocket(idea_id))
        
        # Monitor idea progress
        print(f"\n⏱️  Monitoring Idea Progress...")
        for i in range(12):  # Monitor for 2 minutes
            await asyncio.sleep(10)  # Check every 10 seconds
            
            idea_status = await get_idea_status(session, idea_id)
            if idea_status:
                print(f"📈 Progress Update {i+1}:")
                print(f"   Status: {idea_status['status']}")
                print(f"   Updated: {idea_status['updated_at']}")
                
                if idea_status.get('department_assignment'):
                    print(f"   Department: {idea_status['department_assignment']}")
                if idea_status.get('institution_assignment'):
                    print(f"   Institution: {idea_status['institution_assignment']}")
        
        # Get final agent status
        print(f"\n📊 Final Agent Status:")
        await get_agent_status(session)
        
        # Cancel WebSocket monitoring
        websocket_task.cancel()
        
        print(f"\n✅ Test completed! Check the API documentation at {API_BASE_URL}/docs for more details.")


if __name__ == "__main__":
    asyncio.run(main())