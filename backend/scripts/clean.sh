#!/bin/sh
echo "--- Cleaning previous installations ---"
rm -rf /opt/venv
find / -name 'python3*' -path '*/site-packages/*' -delete 2>/dev/null