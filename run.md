Here's a complete **Markdown guide** on how to run the project, assuming you have cloned it from [https://github.com/godfrey-tankan/Plagiarism-ai-detector.git](https://github.com/godfrey-tankan/Plagiarism-ai-detector.git). This guide covers installing dependencies, setting up both the **Django backend** and **React frontend**, and running the full stack locally.

---

# 🧪 Plagiarism & AI Detector – Setup Guide

This project uses **Django (Python)** for the backend and **React (JavaScript)** for the frontend. Follow the steps below to set up and run the project locally.

## 📁 Project Structure

```
Plagiarism-ai-detector/
├── frontend/         # React frontend
├── manage.py         # Django backend entry point
├── requirements.txt  # Python dependencies
├── .env              # (optional) Environment variables
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/godfrey-tankan/Plagiarism-ai-detector.git
cd Plagiarism-ai-detector
```

---

## 🧩 Backend Setup (Django)

### Prerequisites

* Python 3.8 or newer
* pip
* virtualenv (recommended)

### 1. Create and Activate Virtual Environment

```bash
python -m venv env
source env/bin/activate       # Linux/macOS
# OR
env\Scripts\activate          # Windows
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Apply Migrations

```bash
python manage.py migrate
```

### 4. Run the Backend Server

```bash
python manage.py runserver
```

By default, the Django backend runs on [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🌐 Frontend Setup (React)

### Prerequisites

* Node.js (v14 or later)
* npm or yarn

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Node Dependencies

```bash
npm install
# OR
yarn install
```

### 3. Run the React App

```bash
npm run dev
# OR
yarn dev
```

The frontend runs on [http://localhost:8080](http://localhost:8080)

---

## ⚙️ Configuration (Optional)

If your backend API URL differs or you want to use environment variables, make sure to:

* Set up a `.env` file in `frontend/` and configure `VITE_BACKEND_URL` to be http://localhost:8000
* Update Django `ALLOWED_HOSTS` or CORS settings if needed (`corsheaders` is recommended)

---

## ✅ Accessing the App

Once both servers are running:

* **Frontend**: [http://localhost:5173](http://localhost:8080)
* **Backend API**: [http://127.0.0.1:8000/api/](http://127.0.0.1:8000/api/)

---

## 🛠 Common Commands

### Create Superuser for Admin Access

```bash
python manage.py createsuperuser
```

### Collect Static Files (for production)

```bash
python manage.py collectstatic
```

---

