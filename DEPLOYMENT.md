# Streamlit Deployment

streamlit_app.py is the main application file for web deployment.

## Run Locally

```bash
# Install streamlit
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Click "New app"
4. Select this repository
5. Set main file: `streamlit_app.py`
6. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

## Features

- Upload custom CSV or use example dataset
- Choose solver (Auto, ALNS, or CBC MIP)
- Configure optimization parameters
- View interactive route maps
- Download results (HTML map + CSV solution)
- Responsive design with progress tracking
