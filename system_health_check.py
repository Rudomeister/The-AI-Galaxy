#!/usr/bin/env python3
"""
Comprehensive AI-Galaxy System Health Check
Created by Testing QA Specialist Agent
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple

class SystemHealthChecker:
    def __init__(self):
        self.results = {}
        self.failed_checks = []
        
    async def check_redis(self) -> Tuple[bool, str]:
        """Check Redis connectivity and basic operations."""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Basic ping test
            if not r.ping():
                return False, "Redis ping failed"
            
            # Test basic operations
            test_key = "health_check_test"
            r.set(test_key, "test_value", ex=10)
            if r.get(test_key) != "test_value":
                return False, "Redis read/write test failed"
            
            r.delete(test_key)
            return True, "Redis healthy - ping and R/W operations successful"
            
        except Exception as e:
            return False, f"Redis connection failed: {str(e)}"
    
    async def check_chromadb(self) -> Tuple[bool, str]:
        """Check ChromaDB API connectivity."""
        try:
            async with aiohttp.ClientSession() as session:
                # Test v2 API heartbeat
                async with session.get('http://localhost:8000/api/v2/heartbeat', timeout=5) as response:
                    if response.status == 200:
                        return True, "ChromaDB v2 API responding successfully"
                    else:
                        return False, f"ChromaDB returned status {response.status}"
                        
        except Exception as e:
            return False, f"ChromaDB connection failed: {str(e)}"
    
    async def check_backend_api(self) -> Tuple[bool, str]:
        """Check FastAPI backend health and endpoints."""
        try:
            async with aiohttp.ClientSession() as session:
                # Health check
                async with session.get('http://localhost:8080/health/', timeout=10) as response:
                    if response.status != 200:
                        return False, f"Health endpoint returned {response.status}"
                    
                    health_data = await response.json()
                    if not health_data.get('status') == 'healthy':
                        return False, f"Health check failed: {health_data}"
                
                # Agent list endpoint
                async with session.get('http://localhost:8080/api/v1/agents/', timeout=10) as response:
                    if response.status != 200:
                        return False, f"Agents endpoint returned {response.status}"
                    
                    agents_data = await response.json()
                    agent_count = agents_data.get('total_count', 0)
                    active_count = agents_data.get('active_count', 0)
                    
                    if agent_count == 0:
                        return False, "No agents found in system"
                    
                    if active_count == 0:
                        return False, "No active agents found"
                    
                    return True, f"Backend API healthy - {active_count}/{agent_count} agents active"
                    
        except Exception as e:
            return False, f"Backend API connection failed: {str(e)}"
    
    async def check_websocket(self) -> Tuple[bool, str]:
        """Check WebSocket connectivity."""
        try:
            import websockets
            
            # Test WebSocket connection to agents endpoint
            uri = "ws://localhost:8080/ws/agents"
            
            async with websockets.connect(uri, timeout=5) as websocket:
                # Send a test message and wait for response
                await websocket.send(json.dumps({"type": "ping", "data": "health_check"}))
                
                # Wait for any message back (don't need specific response)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3)
                    return True, "WebSocket connection successful"
                except asyncio.TimeoutError:
                    return True, "WebSocket connected (no immediate response expected)"
                    
        except Exception as e:
            return False, f"WebSocket connection failed: {str(e)}"
    
    async def check_frontend(self) -> Tuple[bool, str]:
        """Check frontend React app accessibility."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:3000/', timeout=10) as response:
                    if response.status != 200:
                        return False, f"Frontend returned status {response.status}"
                    
                    html_content = await response.text()
                    
                    # Check for React indicators
                    if 'react' not in html_content.lower() and 'vite' not in html_content.lower():
                        return False, "Frontend doesn't appear to be React/Vite app"
                    
                    return True, "Frontend React app accessible and responding"
                    
        except Exception as e:
            return False, f"Frontend connection failed: {str(e)}"
    
    async def check_containers(self) -> Tuple[bool, str]:
        """Check Docker container status."""
        try:
            import subprocess
            
            # Get container status
            result = subprocess.run(
                ['docker-compose', 'ps', '--format', 'json'],
                cwd='.',
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return False, f"Docker compose ps failed: {result.stderr}"
            
            # Parse container status
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        container = json.loads(line)
                        containers.append(container)
                    except json.JSONDecodeError:
                        continue
            
            if not containers:
                return False, "No containers found"
            
            running_containers = [c for c in containers if c.get('State') == 'running']
            
            if len(running_containers) < 4:  # Expect at least 4 containers
                return False, f"Only {len(running_containers)} containers running, expected at least 4"
            
            container_names = [c.get('Service', 'unknown') for c in running_containers]
            
            return True, f"Containers healthy: {', '.join(container_names)}"
            
        except Exception as e:
            return False, f"Container check failed: {str(e)}"
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive results."""
        print("🏥 AI-Galaxy System Health Check Starting...")
        print("=" * 60)
        
        checks = [
            ("Containers", self.check_containers()),
            ("Redis Database", self.check_redis()),
            ("ChromaDB Vector Store", self.check_chromadb()),
            ("Backend API", self.check_backend_api()),
            ("WebSocket Connection", self.check_websocket()),
            ("Frontend React App", self.check_frontend()),
        ]
        
        # Run all checks concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*[check[1] for check in checks], return_exceptions=True)
        end_time = datetime.now()
        
        # Process results
        check_results = {}
        passed = 0
        failed = 0
        
        for i, (check_name, _) in enumerate(checks):
            result = results[i]
            
            if isinstance(result, Exception):
                status = False
                message = f"Check failed with exception: {str(result)}"
            else:
                status, message = result
            
            check_results[check_name] = {
                "status": "✅ PASS" if status else "❌ FAIL",
                "message": message,
                "success": status
            }
            
            if status:
                passed += 1
                print(f"✅ {check_name}: {message}")
            else:
                failed += 1
                print(f"❌ {check_name}: {message}")
                self.failed_checks.append(check_name)
        
        print("=" * 60)
        
        overall_status = failed == 0
        total_time = (end_time - start_time).total_seconds()
        
        summary = {
            "overall_status": "HEALTHY" if overall_status else "UNHEALTHY",
            "passed_checks": passed,
            "failed_checks": failed,
            "total_checks": len(checks),
            "check_duration_seconds": round(total_time, 2),
            "timestamp": datetime.now().isoformat(),
            "details": check_results
        }
        
        if overall_status:
            print(f"🎉 SYSTEM HEALTHY: {passed}/{len(checks)} checks passed in {total_time:.2f}s")
        else:
            print(f"🚨 SYSTEM ISSUES: {failed}/{len(checks)} checks failed")
            print(f"Failed components: {', '.join(self.failed_checks)}")
        
        return summary

async def main():
    """Main health check execution."""
    checker = SystemHealthChecker()
    results = await checker.run_all_checks()
    
    # Save results to file
    with open('health_check_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "HEALTHY" else 1)

if __name__ == "__main__":
    asyncio.run(main())