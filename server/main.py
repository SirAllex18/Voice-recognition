import uvicorn
from typing import Union
from fastapi import FastAPI, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi import HTTPException, Form
from inference import load_model, predict_speaker
import json
from bson import ObjectId
import os
from shutil import rmtree
from typing import Optional

app = FastAPI()
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SPEAKERS = 202
model_path = "speaker_cnn_model.pth"
model = load_model(model_path, num_speakers=NUM_SPEAKERS, device=device)


with open("label_map.json", "r") as f:
    label_dict = json.load(f)
reverse_label_map = {int_label: folder_name for folder_name, int_label in label_dict.items()}


SECRET_KEY = "someReallyHardToDecipher"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MONGO_CONNECTION_STRING = "mongodb+srv://SirAllex:NewDawn25@clustervc.rgxoh.mongodb.net/?retryWrites=true&w=majority&appName=ClusterVC"
client = AsyncIOMotorClient(MONGO_CONNECTION_STRING)
db = client.get_database("VoiceRecognitionDB")
users_collection = db.get_collection("Users")

app = FastAPI()

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def create_access_token(data: dict, expires_delta: Union[int, None] = None):
    to_encode = data.copy()
    if expires_delta:
        to_encode.update({"exp": expires_delta})
    else:
        to_encode.update({"exp": ACCESS_TOKEN_EXPIRE_MINUTES * 60})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

class User(BaseModel):
    nume: str
    prenume: str
    mail: str
    password: str
    file_path: Optional[str] = None

@app.post("/register")
async def register_user(user: User):
    nume = user.nume
    prenume = user.prenume
    mail = user.mail
    password = user.password

    existing_user = await users_collection.find_one({"nume": nume, "prenume": prenume})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = pwd_context.hash(password)
    new_user = {"nume": nume, "prenume": prenume, "mail": mail, "password": hashed_password }
    await users_collection.insert_one(new_user)

    access_token = create_access_token(data={"sub": f"{nume} {prenume}"})

    return {"access_token": access_token, "user": {"nume": nume, "prenume": prenume, "mail": mail}}

@app.post("/login")
async def login_user(email: str = Form(...), password: str = Form(...)):

    user = await users_collection.find_one({"mail": email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not pwd_context.verify(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(data={"sub": user["mail"]})
    return {
        "access_token": access_token,
        "user": {"nume": user["nume"], "prenume": user["prenume"], "mail": user["mail"]},
    }

@app.get("/")
def root():
    return {"message": "Server is running"}

@app.post("/recognize")
async def recognize_speaker(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    speaker_label, confidence = predict_speaker(
        model=model,
        audio_input=audio_bytes,
        device=device,
        threshold=0.6,   
        is_file_path=False,  
        reverse_label_map=reverse_label_map
    )

    if speaker_label == "unknown":
        return {"speaker_id": "unknown_speaker", "confidence": confidence}
    else:
        return {"speaker_id": speaker_label, "confidence": confidence}
    
@app.get("/users")
async def get_users():
    users = []
    async for user in users_collection.find():
        user["_id"] = str(user["_id"])
        users.append(user)
    
    return {"status": "success", "users": users}


@app.post("/authenticate")
async def authenticate_speaker(
    claimed_id: str = Form(...), 
    file: UploadFile = File(...)
):
    """
    Endpoint to authenticate a user who claims to be `claimed_id`.
    If no folder matches the `claimed_id`, return early with a failure response.
    """

    base_path = os.path.join("data", "training_data", "wav")
    claimed_id_path = os.path.join(base_path, claimed_id)

    if not os.path.isdir(claimed_id_path):
        return {
            "status": "fail",
            "message": "Unknown speaker"
        }

    audio_bytes = await file.read()

    predicted_label, confidence = predict_speaker(
        model=model,
        audio_input=audio_bytes,
        device=device,
        threshold=0.6,      
        is_file_path=False,
        reverse_label_map=reverse_label_map
    )

    if predicted_label == claimed_id and confidence > 0.6:
        return {
            "status": "success",
            "message": f"Voice matches claimed ID: {claimed_id}",
            "confidence": confidence
        }
    else:
        return {
            "status": "fail",
            "message": f"Voice does not match claimed ID: {claimed_id}",
            "predicted_label": predicted_label,
            "confidence": confidence
        }


@app.post("/enroll_voice")
async def enroll_voice(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    email: str = Form(...)
):
    """
    Enroll a new speaker's voice by saving their .wav file into:
    data/training_data/wav/<user_id>/<random_subfolder>/
    
    This endpoint also updates the user's document in the MongoDB collection
    by adding a 'file_path' field that stores the path to the saved folder.
    """
    import os
    import uuid
    

    audio_bytes = await file.read()
    

    base_path = os.path.join("data", "training_data", "wav")
    user_path = os.path.join(base_path, user_id)      
    os.makedirs(user_path, exist_ok=True)
    
    subfolder_name = str(uuid.uuid4())                
    subfolder_path = os.path.join(user_path, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    
    wav_path = os.path.join(subfolder_path, file.filename)
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)

    user = await users_collection.find_one({"mail": email})
    if not user:
        raise HTTPException(status_code=401, detail="No user found")
    

    file_path_to_save = subfolder_path 
    update_result = await users_collection.update_one(
        {"mail": email},  
        {"$set": {"file_path": file_path_to_save}}
    )

    if update_result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to update user with file path")
    
    return {
        "status": "success",
        "message": f"Voice sample saved under ID '{user_id}'. "
                   "Please re-run loadData.py and trainCNN.py to retrain the model.",
        "file_path": file_path_to_save
    }


@app.delete("/deleteUserVoice/{user_id}")
async def delete_enrolled_voice(user_id: str):
    """
    Delete the root folder (named after the user) containing the user's enrolled voice files.
    Also remove the user from the database.
    """
    try:
        object_id = ObjectId(user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    user = await users_collection.find_one({"_id": object_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    file_path = user.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="No file path associated with this user")

    root_folder_path = os.path.dirname(file_path) 

    if not os.path.exists(root_folder_path):
        raise HTTPException(status_code=404, detail="Root folder not found on disk")

    try:
        rmtree(root_folder_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Error deleting root folder: {e}")
    
    # delete_result = await users_collection.delete_one({"_id": object_id})
    # if delete_result.deleted_count == 0:
    #     raise HTTPException(status_code=500, detail="Failed to delete user from database")
    return {"status": "success", "message": "Root folder deleted successfully"}

   
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
