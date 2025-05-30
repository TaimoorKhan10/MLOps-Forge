"""Command-line interface for model monitoring."""

import os
from typing import Optional

import typer
from rich.console import Console

from mlops_production_system.config.settings import settings
from mlops_production_system.monitoring.monitoring import ModelMonitor
from mlops_production_system.utils.logging_utils import setup_logging, get_logger

# Set up CLI app
app = typer.Typer(help="MLOps Production System - Model Monitoring CLI")
console = Console()
logger = get_logger(__name__)


@app.command()
def start_prometheus(
    port: int = typer.Option(
        settings.MONITORING.PROMETHEUS_PORT, 
        "--port", "-p", 
        help="Port to bind the Prometheus metrics server to"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level"
    ),
):
    """Start the Prometheus metrics server."""
    # Set up logging
    setup_logging(level=log_level)
    
    try:
        # Create model monitor
        model_monitor = ModelMonitor()
        
        # Start Prometheus server
        console.print(f"[bold green]Starting Prometheus metrics server on port {port}[/bold green]")
        model_monitor.start_prometheus_server(port=port)
        
        # Keep the server running
        console.print("[bold yellow]Server is running. Press Ctrl+C to stop.[/bold yellow]")
        try:
            # This will keep the script running until interrupted
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("[bold yellow]Server stopped.[/bold yellow]")
        
    except Exception as e:
        logger.exception(f"Error starting Prometheus server: {e}")
        console.print(f"[bold red]Error starting Prometheus server: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def detect_drift(
    reference_data: str = typer.Option(
        ..., "--reference-data", "-r", help="Path to reference data file"
    ),
    current_data: str = typer.Option(
        ..., "--current-data", "-c", help="Path to current data file"
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output-path", "-o", help="Path to save the drift dashboard"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level"
    ),
):
    """Detect data drift and generate a dashboard."""
    # Set up logging
    setup_logging(level=log_level)
    
    try:
        # Validate file paths
        if not os.path.exists(reference_data):
            raise ValueError(f"Reference data file {reference_data} does not exist")
        
        if not os.path.exists(current_data):
            raise ValueError(f"Current data file {current_data} does not exist")
        
        # Create model monitor
        model_monitor = ModelMonitor(reference_data_path=reference_data)
        
        # Load current data
        import pandas as pd
        if current_data.endswith(".csv"):
            current_df = pd.read_csv(current_data)
        elif current_data.endswith(".parquet"):
            current_df = pd.read_parquet(current_data)
        else:
            raise ValueError(f"Unsupported file format: {current_data}")
        
        # Detect drift
        console.print("[bold green]Detecting data drift...[/bold green]")
        drift_scores = model_monitor.detect_drift(current_data=current_df)
        
        # Print drift scores
        console.print("[bold green]Drift scores:[/bold green]")
        for feature, score in drift_scores.items():
            console.print(f"  {feature}: {score:.4f}")
        
        # Generate dashboard
        if output_path:
            console.print("[bold green]Generating drift dashboard...[/bold green]")
            dashboard_path = model_monitor.generate_drift_dashboard(
                output_path=output_path, current_data=current_df
            )
            if dashboard_path:
                console.print(f"[bold green]Dashboard saved to {dashboard_path}[/bold green]")
        
    except Exception as e:
        logger.exception(f"Error detecting drift: {e}")
        console.print(f"[bold red]Error detecting drift: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def generate_dashboard(
    monitoring_data_dir: str = typer.Option(
        None, "--monitoring-data-dir", "-d", 
        help="Directory containing monitoring data files"
    ),
    output_path: str = typer.Option(
        None, "--output-path", "-o", help="Path to save the dashboard"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level"
    ),
):
    """Generate a monitoring dashboard from collected data."""
    # Set up logging
    setup_logging(level=log_level)
    
    try:
        # Determine monitoring data directory
        monitoring_data_dir = monitoring_data_dir or os.path.join(
            settings.DATA.PROCESSED_DATA_DIR, "monitoring"
        )
        
        # Determine output path
        output_path = output_path or os.path.join(
            monitoring_data_dir, "monitoring_dashboard.html"
        )
        
        # Check if monitoring data directory exists
        if not os.path.exists(monitoring_data_dir):
            raise ValueError(f"Monitoring data directory {monitoring_data_dir} does not exist")
        
        # Load monitoring data
        console.print("[bold green]Loading monitoring data...[/bold green]")
        import pandas as pd
        import glob
        
        # Find CSV files
        csv_files = glob.glob(os.path.join(monitoring_data_dir, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {monitoring_data_dir}")
        
        # Load and combine data
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        monitoring_data = pd.concat(dfs, ignore_index=True)
        console.print(f"Loaded {len(monitoring_data)} monitoring records")
        
        # Generate dashboard
        console.print("[bold green]Generating monitoring dashboard...[/bold green]")
        
        # Create simple HTML dashboard with pandas styling
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                h1 {{
                    color: #2C3E50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Monitoring Dashboard</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total predictions: {len(monitoring_data)}</p>
                    <p>Time range: {monitoring_data['timestamp'].min()} to {monitoring_data['timestamp'].max()}</p>
                </div>
                
                <h2>Monitoring Data</h2>
                {monitoring_data.head(100).to_html()}
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        with open(output_path, "w") as f:
            f.write(html)
        
        console.print(f"[bold green]Dashboard saved to {output_path}[/bold green]")
        
    except Exception as e:
        logger.exception(f"Error generating dashboard: {e}")
        console.print(f"[bold red]Error generating dashboard: {e}[/bold red]")
        raise typer.Exit(code=1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
