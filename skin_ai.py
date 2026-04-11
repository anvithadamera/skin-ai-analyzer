# =========================
# FINAL ULTRA STRICT SKIN AI (MOBILE + RENDER FIXED)
# =========================

from flask import Flask, render_template_string, Response, jsonify, request
import cv2
import numpy as np
import hashlib
import json
import os
import time
import base64

app = Flask(__name__)

# ---------------- FACE MODEL ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- LOG SYSTEM ----------------
LOG_FILE = "results_log.json"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump({}, f)

def load_logs():
    with open(LOG_FILE) as f:
        return json.load(f)

def save_logs(data):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f)


# ---------------- IMAGE DECODER (IMPORTANT FIX) ----------------
def decode_image(data_url):
    img_data = data_url.split(",")[1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


# ---------------- ANALYSIS ENGINE ----------------
def analyze(faces, oily, dry):
    logs = load_logs()

    acne = 0
    marks = 0
    dark = 0
    lip = 0
    pigmentation_flag = False

    f_oil_vals = []
    tex_vals = []

    for face in faces:

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        forehead = face[0:int(0.3*h), int(0.3*w):int(0.7*w)]
        cheek_l = face[int(0.4*h):int(0.7*h), 0:int(0.4*w)]
        cheek_r = face[int(0.4*h):int(0.7*h), int(0.6*w):w]
        chin = face[int(0.7*h):h, int(0.3*w):int(0.7*w)]

        fg = cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY)
        cl = cv2.cvtColor(cheek_l, cv2.COLOR_BGR2GRAY)
        cr = cv2.cvtColor(cheek_r, cv2.COLOR_BGR2GRAY)
        cg = cv2.cvtColor(chin, cv2.COLOR_BGR2GRAY)

        tex_vals.append(np.std(np.concatenate((cl.flatten(), cr.flatten()))))
        f_oil_vals.append(np.percentile(fg, 95))

        # -------- ACNE DETECTION --------
        for z in [forehead, cheek_l, cheek_r]:
            hsv = cv2.cvtColor(z, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, (0,160,150), (10,255,255))
            mask2 = cv2.inRange(hsv, (170,160,150), (180,255,255))
            red = mask1 + mask2

            cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if 80 < area < 200:
                    acne += 1

        # -------- MARKS --------
        for z in [forehead, cheek_l, cheek_r]:
            hsv = cv2.cvtColor(z, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (5,50,50), (25,180,220))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if 120 < cv2.contourArea(c) < 350:
                    marks += 1

        # -------- DARK CIRCLES --------
        l_eye = gray[int(0.45*h):int(0.6*h), int(0.2*w):int(0.4*w)]
        r_eye = gray[int(0.45*h):int(0.6*h), int(0.6*w):int(0.8*w)]
        cheek_ref = gray[int(0.5*h):int(0.7*h), int(0.3*w):int(0.7*w)]

        if (np.mean(l_eye) < np.mean(cheek_ref)-15) and (np.mean(r_eye) < np.mean(cheek_ref)-15):
            dark += 2

        # -------- LIP PIGMENTATION --------
        lips = gray[int(0.75*h):int(0.9*h), int(0.3*w):int(0.7*w)]
        face_ref = gray[int(0.4*h):int(0.7*h), int(0.3*w):int(0.7*w)]

        lips_hsv = cv2.cvtColor(face[int(0.75*h):int(0.9*h), int(0.3*w):int(0.7*w)], cv2.COLOR_BGR2HSV)

        if np.mean(lips) < np.mean(face_ref)-15 and np.mean(lips_hsv[:,:,1]) < 80:
            lip += 2

        # -------- PIGMENTATION --------
        if abs(np.mean(fg)-np.mean(cl)) > 28:
            pigmentation_flag = True

    f_oil = np.mean(f_oil_vals)
    tex = np.mean(tex_vals)

    # -------- SKIN TYPE --------
    if f_oil > 185 and tex < 18:
        skin = "Oily"
    elif f_oil > 170 and tex > 20:
        skin = "Combination"
    elif tex > 24:
        skin = "Dry"
    else:
        skin = "Normal"

    if oily == "Yes" and skin == "Normal":
        skin = "Combination"
    if dry == "Yes" and skin == "Normal":
        skin = "Dry"

    # -------- CONCERNS --------
    concerns = []
    exp = []

    if acne >= 4:
        concerns.append("ACNE 🔴")
        exp.append("Inflamed acne detected")
    if marks >= 5:
        concerns.append("ACNE MARKS 🟤")
        exp.append("Post-acne pigmentation")
    if dark >= 2:
        concerns.append("DARK CIRCLES 👁️")
        exp.append("Under-eye darkness detected")
    if lip >= 2:
        concerns.append("LIP PIGMENTATION 💄")
        exp.append("Lip discoloration detected")
    if pigmentation_flag:
        concerns.append("PIGMENTATION ⚫")
        exp.append("Uneven skin tone detected")

    # -------- SCORE --------
    score = 90
    score -= min(acne*2, 10)
    score -= min(marks*1.5, 10)
    score -= min(dark*2.5, 10)
    score -= min(lip*2.5, 10)
    if pigmentation_flag:
        score -= 5

    score = max(60, min(90, score))

    result = {
        "skin": skin,
        "score": int(score),
        "concerns": concerns,
        "exp": exp
    }

    logs[str(time.time())] = result
    save_logs(logs)

    return result


# ---------------- HTML UI ----------------
def build_html(res):
    skin = res["skin"]
    score = res["score"]
    concerns = res["concerns"]
    exp = res["exp"]

    products = {
        "Dry":["Hydrating Cleanser","Hyaluronic Serum","Ceramide Moisturizer","Cream Sunscreen"],
        "Oily":["Gel Cleanser","Niacinamide Serum","Gel Moisturizer","Matte Sunscreen"],
        "Combination":["Gentle Cleanser","Niacinamide Serum","Light Moisturizer","Non-greasy Sunscreen"],
        "Normal":["Gentle Cleanser","Vitamin C Serum","Light Moisturizer","Sunscreen"]
    }

    diet = {
        "Dry":["Drink water","Healthy fats","Omega-3 foods","Avoid caffeine"],
        "Oily":["Avoid sugar","Eat veggies","Zinc foods","Green tea"],
        "Combination":["Balanced diet","Fruits","Hydration","Less junk"],
        "Normal":["Water","Balanced meals","Fruits","Avoid junk"]
    }

    roast = {
        "Dry":"YOUR SKIN IS SCREAMING FOR WATER 💀",
        "Oily":"TOO MUCH OIL MODE 🛢️",
        "Combination":"CAN'T DECIDE SKIN TYPE 😵‍💫",
        "Normal":"SKIN IS PERFECT 😎"
    }

    color = "lime" if score>=80 else "orange" if score>=70 else "red"

    html = f"<h2>SKIN: {skin}</h2>"
    html += f"<h2>SCORE: {score}/90</h2>"
    html += f"<div style='background:#333;height:20px;'><div style='width:{score}%;background:{color};height:100%'></div></div>"

    html += "<h3>CONCERNS</h3><ul>"
    html += "".join(f"<li>{c}</li>" for c in concerns)
    html += "</ul>"

    html += "<h3>EXPLANATION</h3><ul>"
    html += "".join(f"<li>{e}</li>" for e in exp)
    html += "</ul>"

    html += "<h3>PRODUCTS</h3>"
    html += "<br>".join(products[skin])

    html += "<h3>DIET</h3><ul>"
    html += "".join(f"<li>{d}</li>" for d in diet[skin])
    html += "</ul>"

    html += f"<p>{roast[skin]}</p>"

    return html


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>Skin AI</title>
<style>
body{margin:0;background:#111;color:white;font-family:Arial}
.container{display:flex;height:100vh}
.left{width:60%;display:flex;flex-direction:column;align-items:center;justify-content:center}
.right{width:40%;padding:20px;overflow:auto;background:#222}
video{width:90%;border-radius:10px}
button{padding:10px 20px;margin-top:10px}
</style>
</head>

<body>
<h2 style="text-align:center">SKIN AI (PHONE CAMERA)</h2>

<div class="container">

<div class="left">
<video id="video" autoplay playsinline></video>
<canvas id="canvas" style="display:none"></canvas>

<label>Oily after hours?</label>
<select id="oily"><option>No</option><option>Yes</option></select>

<label>Dry patches?</label>
<select id="dry"><option>No</option><option>Yes</option></select>

<button onclick="scan()">SCAN</button>
<p id="status"></p>
</div>

<div class="right" id="result"><h3>RESULT</h3></div>

</div>

<script>
const video=document.getElementById("video");

navigator.mediaDevices.getUserMedia({video:true})
.then(stream=>video.srcObject=stream);

function scan(){
const canvas=document.getElementById("canvas");
const ctx=canvas.getContext("2d");

canvas.width=video.videoWidth;
canvas.height=video.videoHeight;

ctx.drawImage(video,0,0);

let img=canvas.toDataURL("image/jpeg");

document.getElementById("status").innerText="Analyzing...";

fetch("/capture",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({
image:img,
oily:document.getElementById("oily").value,
dry:document.getElementById("dry").value
})
})
.then(r=>r.json())
.then(d=>{
document.getElementById("result").innerHTML=d.html;
document.getElementById("status").innerText="Done";
});
}
</script>

</body>
</html>
""")


@app.route("/capture", methods=["POST"])
def capture():
    data = request.json

    frame = decode_image(data["image"])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"html":"NO FACE DETECTED"})

    x,y,w,h = max(faces, key=lambda f:f[2]*f[3])
    face = frame[y:y+h, x:x+w]

    res = analyze([face], data["oily"], data["dry"])

    return jsonify({"html": build_html(res)})


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False)
