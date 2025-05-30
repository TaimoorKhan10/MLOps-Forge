"""Command-line interface for serving the model API."""

import os
import sys
from typing import Optional

import typer
import uvicorn
from rich.console import Console

from mlops_production_system.api.api import app as fastapi_app
from mlops_production_system.config.settings import settings
from mlops_production_system.utils.logging_utils import setup_logging, get_logger

# Set up CLI app
app = typer.Typer(help="MLOps Production System - Model Serving CLI")
console = Console()
logger = get_logger(__name__)


@app.command()
def serve(
    host: str = typer.Option(
        settings.API.HOST, "--host", "-h", help="Host to bind the server to"
    ),
    port: int = typer.Option(
        settings.API.PORT, "--port", "-p", help="Port to bind the server to"
    ),
    reload: bool = typer.Option(
        settings.API.RELOAD, "--reload", "-r", help="Enable auto-reload on code changes"
    ),
    workers: int = typer.Option(
        settings.API.WORKERS, "--workers", "-w", help="Number of worker processes"
    ),
    log_level: str = typer.Option(
        "info", "--log-level", "-l", help="Logging level"
    ),
    model_path: Optional[str] = typer.Option(
        None, "--model-path", "-m", help="Path to the model file"
    ),
):
    """Start the model serving API."""
    # Set up logging
    setup_logging(level=log_level.upper())
    
    try:
        # Log configuration
        logger.info("API server configuration:")
        logger.info(f"  Host: {host}")
        logger.info(f"  Port: {port}")
        logger.info(f"  Reload: {reload}")
        logger.info(f"  Workers: {workers}")
        logger.info(f"  Log level: {log_level}")
        
        if model_path:
            # Set model path in environment for the FastAPI app to use
            os.environ["MODEL_PATH"] = model_path
            logger.info(f"  Model path: {model_path}")
        
        # Start the server
        console.print(f"[bold green]Starting API server at http://{host}:{port}[/bold green]")
        
        uvicorn.run(
            "mlops_production_system.api.api:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level.lower(),
        )
        
    except Exception as e:
        logger.exception(f"Error starting API server: {e}")
        console.print(f"[bold red]Error starting API server: {e}[/bold red]")
        raise typer.Exit(code=1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
