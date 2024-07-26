import pyrebase
import datetime
import PIL
import streamlit as st
from ultralytics import YOLO

#config firebase
config = {
    "apiKey": "AIzaSyAqOTZ1kW18Wu9I5K5bAWzSqG6g7rJLsDA",
    "authDomain": "coba-aja-deh-4514e.firebaseapp.com",
    "databaseURL": "https://coba-aja-deh-4514e-default-rtdb.firebaseio.com",
    "projectId" : "coba-aja-deh-4514e",
    "storageBucket": "coba-aja-deh-4514e.appspot.com",
    "messagingSenderId": "1081659574106",
    "appId": "1:1081659574106:web:c35a71d0aeebc4d3439563",
    "measurementId": "G-JGY9ZS4REC"
}
kolA1, kolA2 = st.columns(2)
with kolA1:
    st.image ("LOGO SIC.png",width=150)
with kolA2 :
    st.title ("EAGLE VISION")

#init firebase
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

jumlah_siswa = 25

tanggal = st.date_input("Pilih tanggal", value = datetime.datetime.now())
wkt = st.time_input("Pilih Waktu")
tawa = str(tanggal)+str("-")+str(wkt)
status_tombol = st.button("ambil foto", type= "secondary")

if status_tombol == True :
    img = storage.child("data/photo.jpg").download(" ", filename="photo.jpg")
    model = YOLO("model1.pt")
    uploaded_image = PIL.Image.open("photo.jpg")
    res = model.predict(uploaded_image,conf=0.5,save=True)
    box = res[0].boxes.xyxy.tolist()
    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption=tawa,use_column_width=True)
    st.write("Jumlah Siswa : "+str(len(box)))
    st.write ("Jumlah Tidak masuk : "+str(jumlah_siswa-len(box)))