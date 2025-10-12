import os
import time
from fastapi import APIRouter, UploadFile, File, Body, HTTPException
from app.modules.file import Kb_Import
from app.modules.query import Query
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/")
def health_check():
    return {"message": "Server started successfully"}

@router.post("/add-file-import")
async def add_file_import(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        filename, ext = os.path.splitext(file.filename)

        if ext != ".pdf":
            raise Exception("Only PDF files are allowed")

        unique_name = f"{filename}_{int(time.time())}{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        response = await Kb_Import().add_file_import(file_path)

        return {
            "responseData": response,
            "message": "File imported successfully",
            "statusCode": 200,
            "success": True,
        }
    except Exception as e:
        print("Error occurred: ", str(e))
        return {
            "error": str(e), 
            "statusCode": 500,
            "success": False
        }
    
@router.post("/evaluate-embedding-model")
async def query(payload: dict = Body(...)):
    try:
        if "queries" not in payload or len(payload["queries"]) == 0:
            raise Exception("Queries are required")

        response = await Query().evaluate_embedding_model(payload)

        return {
            "responseData": response,
            "message": "File imported successfully",
            "statusCode": 200,
            "success": True,
        }
    except Exception as e:
        print("Error occurred: ", str(e))
        return {
            "error": str(e), 
            "statusCode": 500,
            "success": False
        }