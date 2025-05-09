#!/usr/bin/env python
import argparse
from src.gui.server import run_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Yin-Yang game web interface')
    parser.add_argument('--host', type=str, default='localhost', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser automatically')
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser) 