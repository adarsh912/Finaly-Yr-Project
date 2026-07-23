#!/usr/bin/env python3
"""
Legal AI System - Auto Startup Script
This script automatically starts both the backend and frontend servers.
"""

import os
import sys
import time
import subprocess
import signal
import threading
from pathlib import Path

class LegalAISystem:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        self.venv_path = None
        
    def print_banner(self):
        """Print startup banner"""
        print("=" * 60)
        print("🚀 Legal AI System - Auto Startup")
        print("=" * 60)
        print("Starting backend and frontend servers...")
        print("=" * 60)
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("📋 Checking prerequisites...")
        
        # Check if we're in the right directory
        if not os.path.exists("api_server.py"):
            print("❌ Error: api_server.py not found. Please run this script from the Legal_AI_System directory.")
            return False
            
        # Check if virtual environment exists (either locally or at root)
        for path in ["venv", "../.venv", ".venv"]:
            if os.path.exists(path):
                self.venv_path = path
                break
                
        if not self.venv_path:
            print("❌ Error: Virtual environment not found. Please create it first:")
            print("   python -m venv venv")
            print("   source venv/bin/activate")
            print("   pip install -r requirements.txt")
            return False
            
        # Check if frontend directory exists
        if not os.path.exists("frontend/legal-ai-frontend"):
            print("❌ Error: Frontend directory not found.")
            return False
            
        # Check if node_modules exists
        if not os.path.exists("frontend/legal-ai-frontend/node_modules"):
            print("⚠️  Warning: node_modules not found. Installing dependencies...")
            try:
                subprocess.run(["npm", "install"], 
                             cwd="frontend/legal-ai-frontend", 
                             check=True, 
                             capture_output=True)
                print("✅ Frontend dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error installing frontend dependencies: {e}")
                return False
                
        print("✅ Prerequisites check passed.")
        return True
        
    def start_backend(self):
        """Start the backend server"""
        print("🔧 Starting backend server...")
        
        # Activate virtual environment and start backend
        if os.name == 'nt':  # Windows
            python_cmd = os.path.join(self.venv_path, "Scripts", "python.exe")
        else:  # Unix/Linux/macOS
            python_cmd = os.path.join(self.venv_path, "bin", "python")
            
        try:
            # Start backend process
            self.backend_process = subprocess.Popen(
                [python_cmd, "api_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait for backend to start
            print("⏳ Waiting for backend to initialize...")
            time.sleep(10)
            
            # Check if backend is running
            if self.backend_process.poll() is None:
                print("✅ Backend server started successfully on http://localhost:5001")
                return True
            else:
                print("❌ Backend server failed to start.")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
            
    def start_frontend(self):
        """Start the frontend server"""
        print("🎨 Starting frontend server...")
        
        try:
            # Start frontend process
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd="frontend/legal-ai-frontend",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait for frontend to start
            print("⏳ Waiting for frontend to initialize...")
            time.sleep(15)
            
            # Check if frontend is running
            if self.frontend_process.poll() is None:
                print("✅ Frontend server started successfully on http://localhost:3000")
                return True
            else:
                print("❌ Frontend server failed to start.")
                return False
                
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
            return False
            
    def monitor_processes(self):
        """Monitor backend and frontend processes"""
        def monitor_backend():
            if self.backend_process:
                for line in iter(self.backend_process.stdout.readline, ''):
                    if self.running:
                        print(f"[BACKEND] {line.rstrip()}")
                    else:
                        break
                        
        def monitor_frontend():
            if self.frontend_process:
                for line in iter(self.frontend_process.stdout.readline, ''):
                    if self.running:
                        print(f"[FRONTEND] {line.rstrip()}")
                    else:
                        break
                        
        # Start monitoring threads
        backend_thread = threading.Thread(target=monitor_backend, daemon=True)
        frontend_thread = threading.Thread(target=monitor_frontend, daemon=True)
        
        backend_thread.start()
        frontend_thread.start()
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down Legal AI System...")
        self.running = False
        self.stop_servers()
        sys.exit(0)
        
    def stop_servers(self):
        """Stop both servers"""
        print("🛑 Stopping servers...")
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            print("✅ Backend server stopped.")
            
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            print("✅ Frontend server stopped.")
            
    def run(self):
        """Main run method"""
        try:
            self.print_banner()
            
            # Check prerequisites
            if not self.check_prerequisites():
                return False
                
            # Start backend
            if not self.start_backend():
                return False
                
            # Start frontend
            if not self.start_frontend():
                self.stop_servers()
                return False
                
            # Set up signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Start monitoring
            self.monitor_processes()
            
            print("\n" + "=" * 60)
            print("🎉 Legal AI System is now running!")
            print("=" * 60)
            print("📱 Frontend: http://localhost:3000")
            print("🔧 Backend:  http://localhost:5001")
            print("=" * 60)
            print("Press Ctrl+C to stop all servers")
            print("=" * 60)
            
            # Keep the script running
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal...")
            self.stop_servers()
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.stop_servers()
            return False
            
        return True

def main():
    """Main function"""
    # Change to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create and run the system
    system = LegalAISystem()
    success = system.run()
    
    if success:
        print("✅ Legal AI System stopped successfully.")
    else:
        print("❌ Legal AI System encountered an error.")
        sys.exit(1)

if __name__ == "__main__":
    main() 