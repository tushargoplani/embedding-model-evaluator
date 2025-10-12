# main.py
from fastapi import FastAPI
from dotenv import load_dotenv
import os
from app.routes.views import router as views_router

load_dotenv()

app = FastAPI()

port = int(os.getenv("PORT", 8000))

# Include your routers here
app.include_router(views_router)

# Run the server programmatically (for development)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)