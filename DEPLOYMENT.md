# 🚀 Deployment Guide - SentiMind AI

## ⚠️ Important: Vercel Compatibility

**Vercel does NOT support Streamlit applications** because:
- Streamlit requires a long-running web server
- Vercel is designed for serverless functions (short-lived requests)
- Streamlit apps need persistent WebSocket connections

---

## ✅ **RECOMMENDED: Deploy to Streamlit Cloud (FREE)**

### Why Streamlit Cloud?
- ✅ **FREE** for public repos
- ✅ **Built specifically for Streamlit apps**
- ✅ **Automatic deployment** from GitHub
- ✅ **HTTPS** with custom domains
- ✅ **Easy setup** (5 minutes)

### Step-by-Step Deployment:

#### 1. **Visit Streamlit Cloud**
Go to: https://share.streamlit.io/

#### 2. **Sign in with GitHub**
- Click "Sign in with GitHub"
- Authorize Streamlit

#### 3. **Deploy New App**
- Click "New app"
- **Repository:** Piyush-yadav03/Sentiment-Intelligence
- **Branch:** main
- **Main file path:** sentimind.py
- Click "Deploy!"

#### 4. **Wait for Deployment** (2-3 minutes)
Streamlit will automatically:
- Install dependencies from `requirements.txt`
- Build the application
- Generate a public URL

#### 5. **Your App is Live!**
You'll get a URL like:
```
https://sentimind-ai.streamlit.app
```
or
```
https://piyush-yadav03-sentiment-intelligence-sentimind-xyz123.streamlit.app
```

---

## 🎯 **Alternative Options:**

### Option 2: Deploy to Hugging Face Spaces (FREE)

**Pros:**
- Free for public repos
- Good for ML apps
- Supports Streamlit

**Steps:**
1. Go to: https://huggingface.co/spaces
2. Create new Space
3. Choose "Streamlit" as SDK
4. Connect GitHub repo or upload files
5. Deploy!

### Option 3: Deploy to Render (FREE Tier Available)

**Pros:**
- Supports Streamlit
- Free tier available
- Custom domains

**Steps:**
1. Go to: https://render.com/
2. Create new Web Service
3. Connect GitHub: Piyush-yadav03/Sentiment-Intelligence
4. **Build Command:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Start Command:**
   ```bash
   streamlit run sentimind.py --server.port=$PORT --server.address=0.0.0.0
   ```
6. Deploy!

### Option 4: Deploy to Railway (Paid after trial)

**Pros:**
- Easy deployment
- Good performance

**Steps:**
1. Go to: https://railway.app/
2. Create new project
3. Deploy from GitHub
4. Set start command: `streamlit run sentimind.py`

### Option 5: Heroku (Requires Credit Card)

**Pros:**
- Well-documented
- Reliable

**Steps:**
1. Create `Procfile`:
   ```
   web: streamlit run sentimind.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Push to Heroku
3. Deploy

---

## 📝 **Vercel Alternative: Create API Version**

If you must use Vercel, you need to create a REST API version (without Streamlit UI):

### Create `api/predict.py`:
```python
from http.server import BaseHTTPRequestHandler
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)
        
        text = data.get('text', '')
        
        # Load model (should be cached in production)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        result = {
            "sentiment": "positive" if probs[1] > probs[0] else "negative",
            "confidence": float(max(probs)),
            "probabilities": {
                "negative": float(probs[0]),
                "positive": float(probs[1])
            }
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
```

**Then create frontend separately on Vercel/Netlify.**

---

## 🏆 **Best Choice for Your Project:**

# **Deploy to Streamlit Cloud**

### Why?
1. ✅ **FREE** for public repos
2. ✅ **No modifications needed** - works with your current code
3. ✅ **Automatic updates** when you push to GitHub
4. ✅ **Professional URL** you can share
5. ✅ **Perfect for academic projects**
6. ✅ **Easy to demo** during presentations

---

## 🚀 **Quick Deploy to Streamlit Cloud:**

### 1. Click this link:
https://share.streamlit.io/

### 2. Sign in with GitHub

### 3. Click "New app" and enter:
```
Repository: Piyush-yadav03/Sentiment-Intelligence
Branch: main
Main file path: sentimind.py
```

### 4. Click "Deploy!"

### 5. Share your live app URL! 🎉

---

## 📊 **Comparison:**

| Platform | Cost | Streamlit Support | Setup Time | Best For |
|----------|------|-------------------|------------|----------|
| **Streamlit Cloud** | FREE | ✅ Native | 5 min | **RECOMMENDED** |
| Hugging Face | FREE | ✅ Yes | 10 min | ML Apps |
| Render | FREE tier | ✅ Yes | 15 min | Production |
| Railway | Trial free | ✅ Yes | 10 min | General apps |
| **Vercel** | FREE | ❌ NO | N/A | **NOT COMPATIBLE** |

---

## ⚠️ **Why NOT Vercel?**

Vercel error message says:
> "No python entrypoint found"

This is because:
1. Vercel expects serverless functions (Flask/FastAPI)
2. Streamlit requires a persistent server
3. Streamlit uses WebSockets for real-time updates
4. Vercel's serverless model doesn't support this

**Solution:** Use Streamlit Cloud instead! It's designed specifically for this.

---

## 🎓 **For Your Project Submission:**

You can mention:
- "Deployed on Streamlit Cloud for live demonstration"
- "Accessible at: [your-streamlit-url]"
- "Free, production-ready hosting"

---

## 📞 **Need Help?**

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum:** https://discuss.streamlit.io/

---

**Deploy to Streamlit Cloud now - it's FREE and takes 5 minutes! 🚀**
