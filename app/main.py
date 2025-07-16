# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 00:17:44 2025

@author: Snehal
"""
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.predict import predict_image

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    prediction = predict_image(file.file)
    return {"prediction": prediction}
