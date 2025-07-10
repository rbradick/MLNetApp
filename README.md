
# MLNet Saliency Heatmap App

This is a Streamlit app that generates visual saliency heatmaps using the MLNet deep learning model. It predicts where human attention is most likely to focus in an image.

## 🔥 Features

- Upload an image (JPG/PNG)
- Generates a heatmap showing visual saliency
- Uses pretrained MLNet model hosted on Hugging Face
- Fully deployable to Render or Hugging Face Spaces

## 🧠 Model

This app loads `mlnet.pth` from:
[https://huggingface.co/rbradick76/mlnet-saliency](https://huggingface.co/rbradick76/mlnet-saliency)

## 🚀 Deploy on Render

1. Create a new GitHub repository
2. Upload all files in this repo
3. Go to [Render.com](https://dashboard.render.com), click **"New Web Service"**
4. Connect your GitHub repo
5. Use the following settings:

| Setting            | Value                               |
|--------------------|-------------------------------------|
| **Environment**     | Python                              |
| **Build Command**   | `pip install -r requirements.txt`   |
| **Start Command**   | `streamlit run app.py --server.port=$PORT` |
| **Environment Variables** | `PYTHONUNBUFFERED=1` (optional: `STREAMLIT_HOME=/tmp`) |

## 📦 Requirements

See `requirements.txt` for Python dependencies.

## ✅ License

MIT License
