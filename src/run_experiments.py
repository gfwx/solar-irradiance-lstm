import os
import subprocess
import json
import re
import sys

def run_command(command):
    print(f"Running: {command}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}")
        print(stderr.decode())
        return None
    return stdout.decode()

def parse_metrics(output):
    metrics = {}
    # Look for patterns like "RMSE: 123.45 W/m^2"
    rmse_match = re.search(r"RMSE: ([\d\.]+) W/m\^2", output)
    nrmse_match = re.search(r"nRMSE: ([\d\.]+)%", output)
    r2_match = re.search(r"R2 Score: ([\d\.\-]+)", output)
    
    if rmse_match: metrics['RMSE'] = float(rmse_match.group(1))
    if nrmse_match: metrics['nRMSE'] = float(nrmse_match.group(1))
    if r2_match: metrics['R2'] = float(r2_match.group(1))
    
    return metrics

def main():
    horizons = [24, 72, 168]
    results = {}
    
    # Use venv python
    python_exec = os.path.abspath(".venv/bin/python")
    if not os.path.exists(python_exec):
        print(f"Warning: {python_exec} not found. Using sys.executable.")
        python_exec = sys.executable
    
    for h in horizons:
        print(f"\n{'='*50}")
        print(f"Processing {h}h Horizon")
        print(f"{'='*50}")
        
        model_path = f"models/bilstm_{h}h.pth"
        
        # Train
        train_cmd = f"{python_exec} src/train.py --output_window {h} --model_path {model_path} --epochs 20"
        run_command(train_cmd)
        
        # Evaluate
        eval_cmd = f"{python_exec} src/evaluate.py --output_window {h} --model_path {model_path}"
        output = run_command(eval_cmd)
        
        if output:
            print(output)
            metrics = parse_metrics(output)
            results[h] = metrics
        else:
            print(f"Failed to evaluate {h}h model.")
            
    print("\n" + "="*50)
    print("FINAL RESULTS TABLE")
    print("="*50)
    print(f"{'Horizon':<10} | {'RMSE':<10} | {'nRMSE':<10} | {'R2':<10}")
    print("-" * 46)
    for h in horizons:
        m = results.get(h, {})
        rmse = m.get('RMSE', 'N/A')
        nrmse = m.get('nRMSE', 'N/A')
        r2 = m.get('R2', 'N/A')
        
        if isinstance(rmse, float): rmse = f"{rmse:.2f}"
        if isinstance(nrmse, float): nrmse = f"{nrmse:.2f}%"
        if isinstance(r2, float): r2 = f"{r2:.4f}"
        
        print(f"{h:<10} | {rmse:<10} | {nrmse:<10} | {r2:<10}")

if __name__ == "__main__":
    main()
