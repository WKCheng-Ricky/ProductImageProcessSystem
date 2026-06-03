set -e

echo "===  Image Processing Setup ==="
echo ""

# Check if running in correct directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the test_photo directory"
    exit 1
fi

# Install system dependencies
echo "[1/4] Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0

# Create virtual environment
echo ""
echo "[2/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "[3/4] Upgrading pip..."
pip install --upgrade pip

echo ""
echo "[4/4] Installing web interface dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python main.py --test"
echo ""
echo "To launch the web interface, run:"
echo "  python main.py --serve --port 8000"
echo "  Then open http://localhost:8000 in your browser"
