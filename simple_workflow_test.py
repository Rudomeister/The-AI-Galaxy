import requests
import json
import time

print('=== Testing Idea Workflow ===')

# Create test idea
test_idea = {
    'title': 'Smart Code Review Assistant', 
    'description': 'An AI-powered assistant that automatically reviews code, detects bugs, suggests improvements, and integrates with Git workflows.',
    'priority': 'high',
    'tags': ['ai', 'code-review', 'automation']
}

print('Creating idea...')
try:
    response = requests.post('http://localhost:8080/api/v1/ideas/', json=test_idea, timeout=10)
    if response.status_code == 201:
        idea = response.json()
        idea_id = idea['id']
        print(f'SUCCESS: Idea created with ID {idea_id}')
        print(f'Initial status: {idea.get("status", "unknown")}')
        
        # Wait for processing
        print('Waiting 5 seconds for processing...')
        time.sleep(5)
        
        # Check status
        status_response = requests.get(f'http://localhost:8080/api/v1/ideas/{idea_id}', timeout=10)
        if status_response.status_code == 200:
            updated_idea = status_response.json()
            new_status = updated_idea.get('status', 'unknown')
            print(f'Updated status: {new_status}')
            
            if new_status != 'created':
                print('SUCCESS: Idea progressed through workflow!')
            else:
                print('WAITING: Idea still in created state, may need more time')
        else:
            print(f'ERROR: Could not fetch idea status: {status_response.status_code}')
    else:
        print(f'ERROR: Failed to create idea: {response.status_code} - {response.text}')
except Exception as e:
    print(f'ERROR: {str(e)}')