#!/usr/bin/env python3
"""
Monoculus Backend API (Data Relay)
Flask server that receives detection data from integrated_monitor_2.py
and serves it to the monoculus-app frontend.
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables to store real-time data from monitor script
system_state = {
    'npu_connected': False,
    'last_heartbeat': 0, # Timestamp for connection check
    
    'is_home': True,
    'last_update': None,
    'fps': 0,
    'status': 'monitoring',
    
    # Event Status
    'fire': {
        'detected': False,
        'type': None, 
        'level': None,
        'conf': 0,
        'last_time': None,
        'image_url': None
    },
    'intruder': {
        'detected': False,
        'conf': 0,
        'last_time': None,
        'image_url': None
    },
    'fall': {
        'detected': False,
        'type': None,
        'last_time': None,
        'image_url': None
    },
    
    # Sleep Data
    'sleep': {
        'is_monitoring': False,
        'toss_turn_count': 0,
        'temperature': 21.0,
        'positions': [],
        'start_time': None,
        'posture': None,
        'ha_data': None
    },
    
    # System Logs (Recent 50)
    'logs': []
}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR, exist_ok=True)

@app.route('/')
def index():
    """Serve the standalone dashboard"""
    return send_from_directory('static', 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status for frontend"""
    global system_state
    
    # Calculate connection status based on heartbeat
    now = datetime.now().timestamp()
    if now - system_state['last_heartbeat'] > 15: # 15s timeout
        system_state['npu_connected'] = False
        system_state['status'] = 'DISCONNECTED'
        system_state['fps'] = 0
    else:
        system_state['npu_connected'] = True
        
    return jsonify(system_state)

@app.route('/api/update', methods=['POST'])
def update_data():
    """Receiver endpoint for integrated_monitor_2.py"""
    global system_state
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Update heartbeat
        system_state['last_heartbeat'] = datetime.now().timestamp()
        
        # Update system state based on what was received
        for key in data:
            if key in system_state:
                if isinstance(system_state[key], dict) and isinstance(data[key], dict):
                    # Recursive update for nested dicts (fire, intruder, fall)
                    system_state[key].update(data[key])
                else:
                    system_state[key] = data[key]
        
        # Handle new Log entry
        if 'log' in data:
            log_entry = data['log']
            # Add unique ID if not present
            if 'id' not in log_entry:
                log_entry['id'] = int(datetime.now().timestamp() * 1000)
            
            system_state['logs'].insert(0, log_entry) # Add to top
            if len(system_state['logs']) > 50:
                system_state['logs'].pop()
        
        system_state['last_update'] = datetime.now().strftime('%H:%M:%S')
        return jsonify({'message': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset counters"""
    global system_state
    
    # Only reset sticky counters, status should be auto-synced by NPU 
    system_state['sleep']['toss_turn_count'] = 0
    
    # Force reset alerts only if NPU is disconnected (otherwise NPU will overwrite)
    now = datetime.now().timestamp()
    if now - system_state['last_heartbeat'] > 15:
        system_state['fire']['detected'] = False
        system_state['intruder']['detected'] = False
        system_state['fall']['detected'] = False
        
    return jsonify({'message': 'Reset successful'})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve detection images from monoculus-app public folder"""
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    # Use adhoc SSL context for immediate HTTPS support
    # Requires: pip install pyOpenSSL
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
