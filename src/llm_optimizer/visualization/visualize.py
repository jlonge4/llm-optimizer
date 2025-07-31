#!/usr/bin/env python3
"""
Pareto LLM Optimizer Dashboard

This script starts a local HTTP server to serve an interactive visualization
page for benchmark data using Plotly.js.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time
import hashlib

# Add scipy for curve fitting
try:
    import numpy as np
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
    print("scipy and numpy successfully imported")
except ImportError as e:
    SCIPY_AVAILABLE = False
    print(f"Warning: scipy not available. Error: {e}. Fit lines will not be calculated.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ParetoLLMOptimizer:
    def __init__(self, config_file: str = "visualization_config.json"):
        self.config_file = config_file
        self.config = self.load_config(config_file)
        
    def load_config(self, config_file: str) -> dict:
        """Load visualization configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}

    def get_data_hash(self, data_dict: dict, data_files: list = None) -> str:
        """Generate hash for the data to use as cache key."""
        # If data_files is provided, hash those specific files
        if data_files:
            try:
                # Sort files to ensure consistent hash
                data_files.sort()
                
                # Combine all data file contents
                combined_content = ""
                for json_file in data_files:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        combined_content += f.read()
                
                return hashlib.md5(combined_content.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"Failed to hash data files: {e}, falling back to processed data hash")
        
        # Fallback to hashing the processed data
        data_str = json.dumps(data_dict["data"], sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def get_fit_cache_path(self, data_hash: str) -> str:
        """Get the path for the fit cache file."""
        return f"fit_cache_{data_hash}.json"

    def load_fit_cache(self, data_hash: str) -> dict:
        """Load fit lines from cache if available."""
        cache_path = self.get_fit_cache_path(data_hash)
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                logger.info(f"Loaded fit cache from {cache_path}")
                return cache_data
        except FileNotFoundError:
            logger.info(f"No fit cache found at {cache_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing fit cache: {e}")
            return {}

    def save_fit_cache(self, data_hash: str, fit_lines_data: dict):
        """Save fit lines to cache file."""
        cache_path = self.get_fit_cache_path(data_hash)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(fit_lines_data, f, indent=2)
            logger.info(f"Saved fit cache to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving fit cache: {e}")

    def extract_value(self, data: dict, path: str):
        """Extract value from nested dictionary using dot notation path."""
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def flatten_data(self, data: list) -> list:
        """Flatten the nested data structure according to the data mapping."""
        if not self.config or 'data_mapping' not in self.config:
            return data

        flattened = []
        mapping = self.config['data_mapping']

        for item in data:
            flattened_item = {}
            for field_name, data_path in mapping.items():
                value = self.extract_value(item, data_path)
                if value is not None:
                    flattened_item[field_name] = value
            flattened.append(flattened_item)

        return flattened

    def calculate_fit_lines(self, data: list, x_field: str, y_field: str, color_field: str, fit_type: str = 'polynomial') -> dict:
        """Calculate fit lines for each color group using scipy."""
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, skipping fit line calculation")
            return {}
            
        logger.info(f"Calculating fit lines for {x_field} vs {y_field}, color by {color_field}, fit type: {fit_type}")
        fit_lines = {}
        
        # Group data by color field
        grouped_data = {}
        for point in data:
            if point.get(x_field) is not None and point.get(y_field) is not None and point.get(color_field) is not None:
                color_value = point[color_field]
                if color_value not in grouped_data:
                    grouped_data[color_value] = []
                grouped_data[color_value].append({
                    'x': point[x_field],
                    'y': point[y_field]
                })
        
        logger.info(f"Found {len(grouped_data)} color groups: {list(grouped_data.keys())}")
        
        # Calculate fit line for each group
        for color_value, points in grouped_data.items():
            if len(points) < 3:  # Need at least 3 points for polynomial fit
                logger.warning(f"Not enough points for {color_value}: {len(points)} points (need at least 3)")
                continue
                
            x_values = [p['x'] for p in points]
            y_values = [p['y'] for p in points]
            
            # Check if x values are all the same (no variation)
            if len(set(x_values)) < 2:
                logger.warning(f"No x-axis variation for {color_value}: all x values are {x_values[0]}")
                continue
                
            # Check if y values are all the same (no variation)
            if len(set(y_values)) < 2:
                logger.warning(f"No y-axis variation for {color_value}: all y values are {y_values[0]}")
                continue
            
            logger.info(f"Calculating fit for {color_value} with {len(points)} points")
            
            try:
                x_values_np = np.array(x_values)
                y_values_np = np.array(y_values)
                
                # Fit polynomial
                coeffs = np.polyfit(x_values_np, y_values_np, 2)
                poly_func = np.poly1d(coeffs)
                
                # Generate fit line points
                x_min, x_max = np.min(x_values_np), np.max(x_values_np)
                x_fit = np.linspace(x_min, x_max, 100)
                y_fit = poly_func(x_fit)
                
                # Calculate R²
                y_pred = poly_func(x_values_np)
                ss_res = np.sum((y_values_np - y_pred) ** 2)
                ss_tot = np.sum((y_values_np - np.mean(y_values_np)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                fit_lines[color_value] = {
                    'x': x_fit.tolist(),
                    'y': y_fit.tolist(),
                    'equation': f'y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}',
                    'r2': r2
                }
                logger.info(f"Polynomial fit for {color_value}: coeffs={coeffs}, R²={r2:.4f}")
                        
            except Exception as e:
                logger.warning(f"Failed to calculate fit line for {color_value}: {e}")
                continue
        
        logger.info(f"Calculated fit lines for {len(fit_lines)} groups")
        return fit_lines

    def load_benchmark_data(self, data_file: str) -> dict:
        """Load benchmark data from JSON files."""
        data = []
        constraints = {}
        json_files = [Path(data_file)]
        
        # Load data from the specified file
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
            logger.info(f"Loaded data from {data_file}")
        except Exception as e:
            logger.error(f"Error loading {data_file}: {e}")
        
        # Flatten the data according to the mapping
        flattened_data = self.flatten_data(data)
        logger.info(f"Loaded {len(flattened_data)} benchmark records from {data_file}")
        logger.info(f"Found constraints: {constraints}")
        return {"data": flattened_data, "constraints": constraints, "data_files": json_files}

    def get_available_fields(self, data: list) -> set:
        """Get fields that have data in the dataset."""
        available_fields = set()
        
        for item in data:
            for field_name in item.keys():
                if item[field_name] is not None and item[field_name] != "":
                    available_fields.add(field_name)
        
        return available_fields

    def get_field_options(self, data: list = None) -> dict:
        """Get field options organized by category, with disabled fields for missing data."""
        if not self.config or 'fields' not in self.config:
            return {}
        
        # Get available fields if data is provided
        available_fields = set()
        if data is not None:
            available_fields = self.get_available_fields(data)
            logger.info(f"Available fields with data: {available_fields}")
        
        categories = {}
        for field_id, field_info in self.config['fields'].items():
            category = field_info.get('category', 'other')
            if category not in categories:
                categories[category] = {
                    'label': self.config.get('categories', {}).get(category, {}).get('label', category.title()),
                    'description': self.config.get('categories', {}).get(category, {}).get('description', ''),
                    'fields': {}
                }
            
            # Add disabled flag for fields without data
            field_info_copy = field_info.copy()
            if data is not None and field_id not in available_fields:
                field_info_copy['disabled'] = True
                field_info_copy['disabled_reason'] = 'No data available'
                logger.info(f"Field '{field_id}' disabled - no data available")
            else:
                field_info_copy['disabled'] = False
                
            categories[category]['fields'][field_id] = field_info_copy
        
        return categories

    def get_field_categories(self) -> dict:
        """Get field category information for hover display."""
        if not self.config or 'field_categories' not in self.config:
            return {}
        return self.config['field_categories']

    def create_html_page(self, data_dict: dict) -> str:
        """Create the HTML page with embedded data and configuration."""
        # Load the HTML template
        # Try to find template relative to this file's location
        current_dir = Path(__file__).parent
        template_path = current_dir / "template.html"
        if not template_path.exists():
            logger.error(f"Template file not found at {template_path}")
            return ""

        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Check for cached fit lines
        data_files = data_dict.get("data_files", [])
        data_hash = self.get_data_hash(data_dict, data_files)
        fit_lines_data = self.load_fit_cache(data_hash)
        
        # Calculate fit lines if not cached
        if not fit_lines_data and SCIPY_AVAILABLE:
            logger.info(f"Calculating fit lines for frontend-selectable combinations")
            
            # Get available fields with data
            available_fields = self.get_available_fields(data_dict["data"])
            
            # Get fields that can be used for X/Y axes (latency and performance only)
            x_y_fields = []
            color_fields = []
            for field_name, field_info in self.config.get('fields', {}).items():
                # Only include fields that have data for fit line calculations
                if field_name not in available_fields:
                    continue
                    
                category = field_info.get('category', '')
                if category in ['latency', 'performance']:
                    x_y_fields.append(field_name)
                elif category == 'configuration':
                    color_fields.append(field_name)
            
            logger.info(f"Available X/Y axis fields for fit lines: {x_y_fields}")
            logger.info(f"Available color fields for fit lines: {color_fields}")
            
            # Calculate fit lines for each possible x-y combination and color field
            for x_field in x_y_fields:
                for y_field in x_y_fields:
                    if x_field != y_field:
                        for color_field in color_fields:
                            key = f"{x_field}_vs_{y_field}_by_{color_field}"
                            fit_lines = self.calculate_fit_lines(
                                data_dict["data"],
                                x_field,
                                y_field,
                                color_field,
                                'polynomial'
                            )
                            fit_lines_data[key] = fit_lines
            
            logger.info(f"Calculated fit lines for {len(fit_lines_data)} frontend-selectable combinations")
            
            # Save to cache
            self.save_fit_cache(data_hash, fit_lines_data)
        elif fit_lines_data:
            logger.info(f"Using cached fit lines for {len(fit_lines_data)} combinations")
        else:
            logger.warning("SCIPY_AVAILABLE is False, skipping fit line calculation")

        # Prepare data for embedding
        data_json = json.dumps(data_dict["data"], indent=2)
        constraints_json = json.dumps(data_dict["constraints"], indent=2)
        config_json = json.dumps(self.config, indent=2)
        field_options_json = json.dumps(self.get_field_options(data_dict["data"]), indent=2)
        field_categories_json = json.dumps(self.get_field_categories(), indent=2)
        defaults_json = json.dumps(self.config.get('defaults', {}), indent=2)
        fit_lines_json = json.dumps(fit_lines_data, indent=2)

        # Get UI configuration
        ui_config = self.config.get('ui', {})

        # Replace placeholders in the template
        html_content = html_content.replace('{title}', ui_config.get('title', 'Pareto LLM Optimizer Dashboard'))
        html_content = html_content.replace('{subtitle}', ui_config.get('subtitle', 'Interactive Performance Analysis & Optimization Tool'))
        html_content = html_content.replace('{description}', ui_config.get('description', 'Select different metrics for X and Y axes to analyze performance relationships and identify Pareto optimal configurations. Hover over data points to see detailed configuration information.'))
        html_content = html_content.replace('{data_json}', data_json)
        html_content = html_content.replace('{constraints_json}', constraints_json)
        html_content = html_content.replace('{config_json}', config_json)
        html_content = html_content.replace('{field_options_json}', field_options_json)
        html_content = html_content.replace('{field_categories_json}', field_categories_json)
        html_content = html_content.replace('{defaults_json}', defaults_json)
        html_content = html_content.replace('{fit_lines_json}', fit_lines_json)

        return html_content

    def generate_dashboard(self, data_file: str) -> str:
        """Generate the dashboard HTML file."""
        # Load benchmark data
        data_dict = self.load_benchmark_data(data_file)
        
        # Create HTML page
        html_content = self.create_html_page(data_dict)
        
        # Write to file
        html_file = "pareto_llm_dashboard.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Created Pareto LLM Dashboard: {html_file}")
        return html_file

    def start_server(self, port: int = 8080):
        """Start the HTTP server."""
        try:
            server = HTTPServer(('localhost', port), SimpleHTTPRequestHandler)
            logger.info(f"Starting server at http://localhost:{port}")
            logger.info(f"Dashboard available at: http://localhost:{port}/pareto_llm_dashboard.html")
            logger.info("Press Ctrl+C to stop the server")
            
            # Open the dashboard in the default browser
            import webbrowser
            webbrowser.open(f'http://localhost:{port}/pareto_llm_dashboard.html')
            
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Error starting server: {e}")

def main():
    parser = argparse.ArgumentParser(description="Pareto LLM Optimizer Dashboard")
    parser.add_argument("--data", required=True, help="Data file path (e.g., sample_data.json)")
    parser.add_argument("--config", default="visualization_config.json", help="Configuration file path")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    args = parser.parse_args()
    
    # Create optimizer instance
    optimizer = ParetoLLMOptimizer(args.config)
    
    # Generate dashboard
    html_file = optimizer.generate_dashboard(args.data)
    
    # Start server
    optimizer.start_server(args.port)

if __name__ == "__main__":
    main()