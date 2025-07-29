import requests
import json
import time

print('=== Final Workflow Test ===')

# Create a new test idea
test_idea = {
    'title': 'AI-Powered Bug Detection System', 
    'description': 'A system that uses machine learning to automatically detect and categorize bugs in code repositories, providing instant feedback to developers.',
    'priority': 'high',
    'tags': ['ai', 'bug-detection', 'ml', 'devtools']
}

print('Step 1: Creating new idea...')
try:
    response = requests.post('http://localhost:8080/api/v1/ideas/', json=test_idea, timeout=15)
    print(f'Response status: {response.status_code}')
    
    if response.status_code in [200, 201]:
        idea = response.json()
        idea_id = idea['id']
        print(f'SUCCESS: Idea created with ID {idea_id}')
        print(f'Initial status: {idea.get("status", "unknown")}')
        
        print('\nStep 2: Checking if idea appears in list...')
        time.sleep(2)  # Brief wait
        
        list_response = requests.get('http://localhost:8080/api/v1/ideas/', timeout=10)
        if list_response.status_code == 200:
            ideas_data = list_response.json()
            print(f'Ideas in system: {ideas_data["total_count"]}')
            
            if ideas_data['total_count'] > 0:
                print('SUCCESS: Ideas are now visible in list!')
                for idea_item in ideas_data['ideas']:
                    print(f'  - {idea_item["title"]} (Status: {idea_item["status"]})')
            else:
                print('ISSUE: Ideas list still empty despite creation')
        else:
            print(f'ERROR: Could not fetch ideas list: {list_response.status_code}')
            
        print('\nStep 3: Waiting for workflow processing...')
        time.sleep(8)  # Wait for agents to process
        
        status_response = requests.get(f'http://localhost:8080/api/v1/ideas/{idea_id}', timeout=10)
        if status_response.status_code == 200:
            updated_idea = status_response.json()
            new_status = updated_idea.get('status', 'unknown')
            print(f'Final status: {new_status}')
            
            if new_status != 'created':
                print('SUCCESS: Idea progressed through workflow!')
            else:
                print('NOTE: Idea still in created state (may need more processing time)')
        else:
            print(f'ERROR: Could not fetch updated idea: {status_response.status_code}')
            
    else:
        print(f'ERROR: Failed to create idea: {response.status_code}')
        print(f'Response: {response.text}')
        
except Exception as e:
    print(f'ERROR: {str(e)}')

print('\n=== Test Complete ===')