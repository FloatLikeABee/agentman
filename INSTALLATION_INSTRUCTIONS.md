# Installation Instructions

This document covers **only** installing and running the backend and frontend. It does not cover Ollama, Playwright, SMTP, or other optional services.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Backend Installation](#2-backend-installation)
3. [Frontend Installation](#3-frontend-installation)
4. [Environment Configuration](#4-environment-configuration)
5. [Running the Application](#5-running-the-application)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

### 1.1 Python (Backend)

| Requirement | Details |
|-------------|---------|
| **Supported versions** | **Python 3.10**, **3.11**, **3.12**, or **3.13** |
| **Recommended** | 3.11 or 3.12 (best compatibility with dependencies) |
| **Check your version** | `python --version` or `python3 --version` |

- **Windows**: Download the installer from [python.org](https://www.python.org/downloads/). During setup, check **"Add Python to PATH"**.
- **macOS**: Use the official installer, Homebrew (`brew install python@3.12`), or [pyenv](https://github.com/pyenv/pyenv).
- **Linux**: Use your package manager (e.g. `sudo apt install python3.11 python3.11-venv python3-pip`) or pyenv.

Ensure `pip` is available:

```bash
pip --version
# or
python -m pip --version
```

### 1.2 Node.js and npm (Frontend)

| Requirement | Details |
|-------------|---------|
| **Node.js** | **v16** or newer (v18 LTS or v20 LTS recommended) |
| **npm** | Comes with Node.js (v7+); or use yarn if you prefer |
| **Check versions** | `node --version` and `npm --version` |

- **Windows**: Download the LTS installer from [nodejs.org](https://nodejs.org/).
- **macOS**: Use the official installer or Homebrew: `brew install node`.
- **Linux**: Use [NodeSource](https://github.com/nodesource/distributions) or your distro’s package manager.

### 1.3 Git (optional)

Required only if you clone the repo. Install from [git-scm.com](https://git-scm.com/).

---

## 2. Backend Installation

### 2.1 Get the project

If you have the code as a folder, go into it. Otherwise clone:

```bash
git clone <repository-url>
cd <project-folder>
```

The backend lives in the **project root** (where `main.py` and `requirements.txt` are).

### 2.2 Create a virtual environment (recommended)

Using a venv keeps backend dependencies isolated.

**Windows (Command Prompt or PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Windows (Git Bash):**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` (or similar) in your prompt. All following `pip` and `python` commands should be run with this environment activated.

### 2.3 Install Python dependencies

From the **project root** (same directory as `requirements.txt`):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- Installation can take several minutes (e.g. `sentence-transformers`, `chromadb`, `numpy`).
- If a package fails, see [Troubleshooting](#6-troubleshooting).

### 2.4 Verify backend installation

```bash
python -c "from src.api import app; print('Backend OK')"
```

If you see `Backend OK`, the backend can be started.

---

## 3. Frontend Installation

### 3.1 Go to the frontend directory

From the project root:

```bash
cd frontend
```

### 3.2 Install Node dependencies

```bash
npm install
```

- If you see peer dependency warnings, you can often continue. If install fails, try:
  ```bash
  npm install --legacy-peer-deps
  ```

### 3.3 Verify frontend installation

```bash
npm run build
```

If the build finishes without errors, the frontend is ready to run.

---

## 4. Environment Configuration

The backend reads settings from a `.env` file in the **project root** (next to `main.py`).

### 4.1 Create `.env`

Copy the example and edit:

**Windows (Command Prompt):**

```cmd
copy env.example .env
```

**Windows (PowerShell) / macOS / Linux:**

```bash
cp env.example .env
```

### 4.2 Minimum required settings

For a basic run, ensure these exist in `.env` (values are examples):

```env
# API (required to start the server)
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Persistence (required for RAG/vector features)
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Embedding model (required for RAG)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Other variables in `env.example` (e.g. Ollama, SMTP, API keys) are **optional** for core installation. You can leave them as-is or remove them; the app will start with the above.

---

## 5. Running the Application

### 5.1 Start the backend

From the **project root**, with the virtual environment activated:

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

- API: **http://localhost:8000**
- API docs: **http://localhost:8000/docs**

### 5.2 Start the frontend

Open a **second terminal**. Go to the frontend folder and start the dev server:

```bash
cd frontend
npm start
```

- App: **http://localhost:3000**
- The frontend is configured to proxy API requests to `http://localhost:8000` (see `proxy` in `frontend/package.json`).

### 5.3 Access the app

Open a browser and go to **http://localhost:3000**.

---

## 6. Troubleshooting

### 6.1 Python / Backend

#### "python: command not found" / "python3: command not found"

- **Windows**: Reinstall Python and check "Add Python to PATH", or use the full path (e.g. `C:\Users\...\AppData\Local\Programs\Python\Python311\python.exe`).
- **macOS / Linux**: Install Python and use `python3` and `pip3` if your system reserves `python` for an older version.

#### "No module named 'src'" or "No module named 'uvicorn'"

- Run all backend commands from the **project root** (where `main.py` is).
- Ensure the virtual environment is activated and you installed dependencies:
  ```bash
  pip install -r requirements.txt
  ```

#### `pip install -r requirements.txt` fails on a specific package

- **Upgrade pip and retry:**
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  ```
- **Python version:** Use 3.10–3.12 if 3.13 gives errors (e.g. with `sentence-transformers` or `chromadb`).
- **Binary packages (e.g. numpy):** On Windows, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if you see compilation errors. Pre-built wheels usually work on 3.10–3.12.
- **SSL/network errors:** Use a mirror or proxy if behind a strict firewall; for Hugging Face, see `HF_MIRROR` / `HF_PROXY` in `env.example` (optional).

#### "Address already in use" or "Port 8000 is already in use"

- Another process is using the port. Either:
  - Stop that process, or
  - Use another port: set `API_PORT=8001` in `.env` and start the backend again. Then change the frontend proxy in `frontend/package.json` to `"proxy": "http://localhost:8001"` and restart the frontend.

#### Backend starts but frontend gets "Network Error" or "Failed to fetch"

- Backend must be running at the URL the frontend uses (default: `http://localhost:8000`).
- Check `proxy` in `frontend/package.json` matches `API_HOST` and `API_PORT` (e.g. `http://localhost:8000`).
- If you use a different host/port, ensure CORS allows it (e.g. `CORS_ORIGINS` in `.env` includes your frontend URL).

#### Windows: `NotImplementedError` when using subprocesses (e.g. in browser automation)

- The app sets `WindowsProactorEventLoopPolicy` in `main.py` for this. Ensure you start the backend with `python main.py` (or that `main` is imported so the policy is applied). If you run uvicorn directly, the policy is still applied when `main` is loaded.

---

### 6.2 Node / Frontend

#### "node: command not found" / "npm: command not found"

- Install Node.js from [nodejs.org](https://nodejs.org/) and restart the terminal. Ensure the install path is in your system PATH.

#### `npm install` fails with peer dependency or engine errors

- Try:
  ```bash
  npm install --legacy-peer-deps
  ```
- Or use a current Node LTS version (e.g. 18 or 20).

#### `npm start` fails or the app does not open

- Run from the **frontend** directory: `cd frontend` then `npm start`.
- If the default port 3000 is in use, the script may offer to run on 3001; accept or free port 3000.
- Clear cache and reinstall if needed:
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  npm start
  ```
  (On Windows, use `rmdir /s /q node_modules` and delete `package-lock.json` if you prefer.)

#### Frontend builds but shows a blank page or only "Cannot GET /"

- You are likely opening the backend URL (e.g. http://localhost:8000) instead of the frontend (http://localhost:3000). Use **http://localhost:3000** for the React app.
- Ensure `npm start` is running and no build errors appear in the terminal.

---

### 6.3 Environment and paths

#### Changes in `.env` have no effect

- Restart the backend after editing `.env`.
- Ensure `.env` is in the **project root** (same folder as `main.py`), not inside `frontend` or `src`.

#### "File not found" or wrong working directory

- Always run:
  - **Backend:** from the project root (where `main.py` and `requirements.txt` are).
  - **Frontend:** from the `frontend` directory for `npm install` and `npm start`.

---

### 6.4 Quick reference

| Symptom | Likely cause | Action |
|--------|----------------|--------|
| Backend won't start | Wrong directory / venv not activated / missing deps | Run from project root, activate venv, `pip install -r requirements.txt` |
| Frontend won't start | Wrong directory / Node not installed / port in use | Run from `frontend`, install Node, use another port if needed |
| 500 or import errors | Missing or broken Python packages | Recreate venv, `pip install -r requirements.txt` |
| API not reachable from UI | Backend not running or wrong port/proxy | Start backend, match proxy and CORS to backend URL |
| Blank page in browser | Opened backend URL instead of frontend | Open http://localhost:3000 |

---

For optional features (Ollama, Playwright, SMTP, etc.), see the main **README.md**.
