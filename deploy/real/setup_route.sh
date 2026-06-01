#!/bin/bash

# Set your Wi-Fi interface name
WIFI_INTERFACE="wlP1p1s0"

# Get the IP address of the Wi-Fi interface dynamically
WIFI_IP=$(ip -4 addr show $WIFI_INTERFACE | awk '/inet /{print $2}' | cut -d/ -f1)

# Check if an IP address was found
if [ -z "$WIFI_IP" ]; then
    echo "No IP address found for interface $WIFI_INTERFACE."
    exit 1
fi

echo "Found IP address $WIFI_IP on interface $WIFI_INTERFACE."

# Derive the network address (zero host bits) for the /24 subnet route.
WIFI_NET=$(echo "$WIFI_IP" | awk -F. '{print $1"."$2"."$3".0"}')
echo "Using subnet $WIFI_NET/24."

setup_table() {
    local table=$1
    echo "Resetting routing rules and table $table..."
    # Remove any prior duplicate rules pointing at this table.
    while sudo ip rule del from "$WIFI_IP" table "$table" 2>/dev/null; do :; done
    sudo ip route flush table "$table" 2>/dev/null || true

    sudo ip route add "$WIFI_NET/24" dev "$WIFI_INTERFACE" table "$table"
    sudo ip route add default via 192.168.123.1 dev "$WIFI_INTERFACE" table "$table"
    sudo ip rule add from "$WIFI_IP" table "$table"
}

setup_table 8080
setup_table 8012

# Verify the changes
echo "Routing setup complete. Current routing rules:"
sudo ip rule show

echo "Current routing table 8080:"
sudo ip route show table 8080
echo "Current routing table 8012:"
sudo ip route show table 8012