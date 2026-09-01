#!/usr/bin/env python3
"""
SatQuery CLI - Command Line Interface
Interactive interface for satellite image queries
"""

import sys
import requests
from pathlib import Path

from cli.image_uploader import ImageUploader
from cli.query_console import QueryConsole
from cli.results_viewer import ResultsViewer
from config.settings import API_HOST, API_PORT
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SatQueryCLI:
    """
    Main CLI application
    """
    
    def __init__(self):
        self.uploader = ImageUploader()
        self.console = QueryConsole()
        self.viewer = ResultsViewer()
        # Use localhost instead of 0.0.0.0 for client connections
        api_host = "localhost" if API_HOST == "0.0.0.0" else API_HOST
        self.api_url = f"http://{api_host}:{API_PORT}"
        self.session_id = None
    
    def run(self):
        """
        Main CLI loop
        """
        self.print_banner()
        
        while True:
            print("\n=== SatQuery Menu ===")
            print("1. Process new query")
            print("2. View query history")
            print("3. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self.process_query()
            elif choice == '2':
                self.console.show_history()
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid option")
    
    def process_query(self):
        """
        Process a single query
        """
        # Step 1: Get image
        image_path = input("\nEnter image path (or 'cancel' to return): ").strip()
        
        if image_path.lower() == 'cancel':
            return
        
        image_data = self.uploader.select_image(image_path)
        if not image_data:
            print("Failed to load image")
            return
        
        print(f"Image loaded: {image_data['filename']} ({image_data['size']} bytes)")
        
        # Step 2: Get query
        query = self.console.get_query("\nEnter your query: ")
        if not query:
            print("Query cannot be empty")
            return
        
        # Step 3: Send to API
        print("\nProcessing query...")
        
        try:
            response = self.send_request(query, image_data)
            
            if response:
                self.viewer.display_results(response)
                
                # Ask to export
                export = input("Export results? (y/n): ").strip().lower()
                if export == 'y':
                    output_file = input("Output file name: ").strip()
                    if not output_file:
                        output_file = "results.json"
                    self.viewer.export_results(response, output_file)
        
        except Exception as e:
            self.viewer.display_error(str(e))
    
    def send_request(self, query: str, image_data: dict):
        """
        Send request to API
        """
        url = f"{self.api_url}/api/v1/query"
        
        files = {
            'image': (image_data['filename'], image_data['data'])
        }
        
        data = {
            'query': query,
            'session_id': self.session_id
        }
        
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            self.session_id = result.get('session_id')
            
            return result
        
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to API server at {self.api_url}. Is the server running?")
        except requests.exceptions.Timeout:
            raise Exception("Request timed out")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error: {e}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def print_banner(self):
        """
        Print CLI banner
        """
        print("\n" + "="*60)
        print("  SatQuery - Satellite Image Query & Analysis System")
        print("="*60)
        print(f"  API Server: {self.api_url}")
        print("="*60 + "\n")


def main():
    """
    CLI entry point
    """
    cli = SatQueryCLI()
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
