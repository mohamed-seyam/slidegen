import uvicorn 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the slideGen FastAPI server."
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to run the server on (default: 8000)",
    )

    parser.add_argument(
        "--reload",
        type=str,
        default="true",
        help="Enable auto-reload (default: true)",
    )

    args = parser.parse_args()
    reload = args.reload.lower() == "true"

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=args.port,
        reload=reload,
    )
    
