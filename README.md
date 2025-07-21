# 🔖 SmartLink Saver – Auto-Summarizing Bookmark Manager

Welcome to **SmartLink Saver**, a minimal yet powerful full-stack bookmarking tool that lets users:

- 🔐 Sign up / log in securely  
- 🌐 Save any URL  
- ✨ Auto-generate summaries using the **Jina AI free endpoint**  
- 🖼️ Fetch titles and favicons  
- 🗂️ Organize and view bookmarks with optional search  

---

## 📌 Features

✅ Email + password authentication (with password hashing)  
✅ JWT-free session-based login  
✅ Bookmark saving: URL → Title + Favicon + Summary  
✅ Integrated [Jina AI](https://jina.ai/) for **free** extractive summaries (no key needed!)  
✅ Responsive bookmark listing with:
- Title
- Summary
- Favicon  
✅ Delete bookmarks  
✅ Semantic search with Jina Embeddings  
✅ CSV-based storage for demo simplicity  
✅ Configurable summarizer/embedding service  
✅ Deployment-ready with Flask  

---

## 🚀 Live Demo

🌐 [Demo on Render / localhost] *(Add your live link or leave blank)*

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML + Jinja + JS (vanilla) |
| Backend | Flask (Python) |
| Database | SQLite3 |
| Auth | Flask-Login |
| AI Services | Jina AI (summarizer & embeddings) |
| Storage | `bookmarks.csv` |
| Hosting | Localhost / Render |

---

## 🧪 Testing

🧪 Includes test route for:
- ✅ Jina summarization API
- ✅ Jina embedding API

📋 Basic validation of summarization & embedding with fallback.

---

## ✍️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/smartlink-saver.git
cd smartlink-saver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py