#!/usr/bin/env python3
"""
Test script to verify the fixed idea workflow.
Created by Testing QA Specialist Agent
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def test_idea_workflow():
    """Test the complete idea workflow from creation to completion."""
    
    print("🧪 Testing QA Specialist - Idea Workflow Test")
    print("=" * 60)
    
    # Test idea data
    test_idea = {
        "title": "Smart Code Review Assistant",
        "description": "An AI-powered assistant that automatically reviews code, detects bugs, suggests improvements, and integrates with Git workflows. This system would analyze code patterns, security vulnerabilities, and best practices.",
        "priority": "high",
        "tags": ["ai", "code-review", "automation", "git"],
        "metadata": {
            "estimated_complexity": "medium",
            "target_languages": ["python", "javascript"],
            "integration_requirements": ["github", "gitlab"]
        }
    }
    
    print(f"📝 Creating test idea: '{test_idea['title']}'")
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Create the idea
        try:
            async with session.post(
                'http://localhost:8080/api/v1/ideas/',
                json=test_idea,
                timeout=10
            ) as response:
                if response.status == 201:
                    idea_data = await response.json()
                    idea_id = idea_data['id']
                    print(f"✅ Idea created successfully with ID: {idea_id}")
                    print(f"   Initial status: {idea_data['status']}")
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to create idea: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error creating idea: {str(e)}")
            return False
        
        # Step 2: Wait a moment for processing
        print("⏳ Waiting for agent processing...")
        await asyncio.sleep(5)
        
        # Step 3: Check idea status and workflow progression
        try:
            async with session.get(
                f'http://localhost:8080/api/v1/ideas/{idea_id}',
                timeout=10
            ) as response:
                if response.status == 200:
                    updated_idea = await response.json()
                    print(f"🔄 Idea status after processing: {updated_idea['status']}")
                    
                    # Check if it progressed beyond "created"
                    if updated_idea['status'] != 'created':
                        print("✅ SUCCESS: Idea progressed through workflow!")
                        print(f"   Current state: {updated_idea['status']}")
                        
                        # Check workflow history if available
                        if 'workflow_history' in updated_idea:
                            print("📊 Workflow History:")
                            for transition in updated_idea['workflow_history']:
                                print(f"   • {transition}")
                        
                        return True
                    else:
                        print("⚠️  Idea still in 'created' state - workflow may need more time")
                        return False
                else:
                    print(f"❌ Failed to fetch idea: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error fetching idea: {str(e)}")
            return False
        
        # Step 4: Check agent activity
        try:
            async with session.get('http://localhost:8080/api/v1/agents/', timeout=10) as response:
                if response.status == 200:
                    agents_data = await response.json()
                    print(f"\n🤖 Agent Status Summary:")
                    print(f"   Active agents: {agents_data['active_count']}/{agents_data['total_count']}")
                    
                    # Check for any agent activity
                    agents_with_tasks = 0
                    for agent in agents_data['agents']:
                        if agent['total_tasks_completed'] > 0:
                            agents_with_tasks += 1
                            print(f"   • {agent['name']}: {agent['total_tasks_completed']} tasks completed")
                    
                    if agents_with_tasks > 0:
                        print(f"✅ {agents_with_tasks} agents have completed tasks")
                    else:
                        print("⚠️  No agents have completed tasks yet")
                        
        except Exception as e:
            print(f"❌ Error checking agents: {str(e)}")

async def main():
    """Main test execution."""
    print("🚀 Starting Idea Workflow Test...")
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = await test_idea_workflow()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 WORKFLOW TEST PASSED: Ideas are now processing correctly!")
    else:
        print("⚠️  WORKFLOW TEST INCOMPLETE: May need more time or investigation")
    
    print(f"🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())