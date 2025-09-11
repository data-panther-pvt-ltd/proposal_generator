Frontend for AI Proposal Generator (Next.js + TypeScript)

## Prerequisites
- Node.js 18+
- Backend FastAPI server running from project root (`uvicorn api_main:app --reload`)
- Backend must have `OPENAI_API_KEY` set. Optionally set `BACKEND_URL` for frontend.

## Run
1. From project root, run backend:
```bash
uvicorn api_main:app --reload
```
2. From `frontend/` directory, install and start dev server:
```bash
npm install
npm run dev
```
Open http://localhost:3000

## Config
- Frontend calls backend at `BACKEND_URL` or defaults to `http://127.0.0.1:8000`.
- Uploads are saved to `../uploads` relative to `frontend` directory.

## Notes
- Generation is queued as a backend background task; results are written to backend output directory per `config/settings.yml`.
