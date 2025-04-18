#Python 3.8.12



# Create virtual environment
python3 -m venv env

# Activate the environment (Linux/macOS)
source env/bin/activate

# If on Windows, run this instead:
# .\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run Home.py

