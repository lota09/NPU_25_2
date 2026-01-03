
import subprocess
import time
import os
import socket
import json
import threading

class RobustPresenceDetector:
    def __init__(self, mac_list, interface="wlan0", subnet_prefix="192.168.50"):
        self.mac_list = [m.upper().strip() for m in mac_list]
        self.interface = interface
        self.subnet_prefix = subnet_prefix
        self.history_file = os.path.join(os.path.dirname(__file__), "db_ips.json")
        self.ip_history = self._load_history() # MAC -> Last known IP
        self.is_home = False
        self.last_found_time = 0

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.ip_history, f)
        except: pass

    def _udp_knock(self):
        """Sends a UDP broadcast packet to wake up devices and refresh ARP tables."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.1)
        # Send to port 5353 (mDNS) or 9 (WOL) - common ports that trigger network stack
        data = b'\xff\xff\xff\xff\xff\xff' # Wake-on-LAN style placeholder
        try:
            sock.sendto(data, ('255.255.255.255', 9))
            sock.sendto(data, (f'{self.subnet_prefix}.255', 9))
            # Also try mDNS port to wake up modern smartphones
            sock.sendto(b'', (f'{self.subnet_prefix}.255', 5353))
        except: pass
        finally: sock.close()

    def scan(self):
        """Perform a multi-stage scan to check if any MAC is home."""
        found = False
        
        # 1. Passive Discovery: Check current neighbor table
        current_neighbor_map = self._get_neighbor_map() # MAC -> (IP, Status)
        
        # Always update history for ANY found trusted MAC
        for mac, info in current_neighbor_map.items():
            if mac in self.mac_list:
                self.ip_history[mac] = info[0] # info[0] is IP
                # STALE means "I remember it but haven't checked recently"
                # If it's REACHABLE, we trust it. 
                # If it's STALE/DELAY/PROBE, we must PROBE it to be sure.
                if info[2] == "REACHABLE":
                    found = True
        
            # 2. Reactive Probe: If not found or only STALEs, try to "wake up" and verify
        if not found:
            # Wake up devices silently (UDP Knocking)
            self._udp_knock()
            
            # Direct Ping Probe (If IP is known)
            for mac in self.mac_list:
                ip = self.ip_history.get(mac)
                if ip:
                    res = subprocess.run(["ping", "-c", "1", "-W", "0.8", ip], 
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0:
                        found = True
                        break
            
            # Final check of neighbor table after probing
            if not found:
                time.sleep(0.2)
                current_neighbor_map = self._get_neighbor_map()
                for mac in self.mac_list:
                    if mac in current_neighbor_map:
                        if current_neighbor_map[mac][2] == "REACHABLE":
                            found = True
                            break

        # 3. Decision Logic: Immediate (No Grace Period)
        self.is_home = found
        if found:
            self.last_found_time = time.time()
            self._save_history() # Persist the latest IP mapping
        
        return self.is_home

    def _get_neighbor_map(self):
        """Returns a map of MAC -> (IP, Interface, Status) from ip neighbor show."""
        neighbor_map = {}
        try:
            # Try different ip command paths
            ip_cmd = None
            for path in ["/usr/sbin/ip", "/sbin/ip", "ip"]:
                try:
                    subprocess.run([path, "--version"], capture_output=True, check=True)
                    ip_cmd = path
                    break
                except:
                    continue
            
            if not ip_cmd:
                # Fallback: Use /proc/net/arp instead
                try:
                    with open('/proc/net/arp', 'r') as f:
                        for line in f.readlines()[1:]:  # Skip header
                            parts = line.split()
                            if len(parts) >= 6:
                                ip = parts[0]
                                mac = parts[3].upper().strip()
                                if mac != "00:00:00:00:00:00":
                                    neighbor_map[mac] = (ip, self.interface, "UNKNOWN")
                except:
                    pass
                return neighbor_map
            
            # filter by interface for more accuracy
            res = subprocess.run([ip_cmd, "neighbor", "show", "dev", self.interface], 
                                 capture_output=True, text=True, check=False)
            for line in res.stdout.splitlines():
                if "lladdr" in line:
                    parts = line.split()
                    ip = parts[0]
                    # Format: 192.168.50.231 dev wlP2p33s0 lladdr 1a:95:e8:88:9e:ba REACHABLE
                    try:
                        idx = parts.index("lladdr")
                        mac = parts[idx+1].upper().strip()
                        status = parts[-1]
                        neighbor_map[mac] = (ip, self.interface, status)
                    except: continue
        except Exception as e:
            print(f"Error reading neighbor table: {e}")
        return neighbor_map

if __name__ == "__main__":
    # Test Block
    MAC_LIST = [m.upper() for m in ["1A:95:E8:88:9E:BA"]] # Note 20
    detector = RobustPresenceDetector(MAC_LIST, interface="wlP2p33s0", subnet_prefix="192.168.50")
    
    print(f"🚀 Running Robust Scan (Target: {MAC_LIST})...")
    status = detector.scan()
    print(f"🏠 Result: {'HOME' if status else 'AWAY'}")
    print(f"📋 Current IP Map: {detector.ip_history}")
