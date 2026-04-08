# =========================
# SKIN AI (ULTIMATE PRO STABLE VERSION)
# =========================

from flask import Flask, render_template_string, jsonify, request
import cv2
import numpy as np
import base64
import hashlib
import time

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------- LOCK --------
last_face_hash = None
last_result = None

def get_face_hash(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray,(50,50))
    small = cv2.GaussianBlur(small,(5,5),0)
    return hashlib.md5(small.tobytes()).hexdigest()

# -------- ANALYSIS (UNCHANGED) --------
def analyze(faces, oily_input, dry_input):
    global last_face_hash, last_result

    face_hash = get_face_hash(faces[0])
    if last_face_hash and face_hash[:20] == last_face_hash[:20]:
        return last_result

    f_oil_vals, cheek_tex_vals, cheek_edge_vals, chin_oil_vals = [], [], [], []
    acne_count = 0
    marks_score = 0
    dark_votes, lip_votes = 0, 0

    for face in faces:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        h,w = gray.shape

        forehead = gray[0:int(0.3*h), int(0.3*w):int(0.7*w)]
        cheek_l = gray[int(0.4*h):int(0.7*h), 0:int(0.4*w)]
        cheek_r = gray[int(0.4*h):int(0.7*h), int(0.6*w):w]
        chin = gray[int(0.7*h):h, int(0.3*w):int(0.7*w)]

        cheeks = np.concatenate((cheek_l.flatten(), cheek_r.flatten()))

        f_oil_vals.append(np.percentile(forehead,95))
        cheek_tex_vals.append(np.std(cheeks))
        cheek_edge_vals.append(np.mean(cv2.Canny(cheeks.reshape(-1,1),50,150)))
        chin_oil_vals.append(np.percentile(chin,95))

        # ACNE
        red_mask = cv2.inRange(hsv,(0,120,120),(10,255,255)) + \
                   cv2.inRange(hsv,(170,120,120),(180,255,255))
        contours,_ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if 60 < area < 250:
                peri = cv2.arcLength(c,True)
                if peri == 0:
                    continue
                circ = 4*np.pi*(area/(peri*peri))
                if circ > 0.5:
                    acne_count += 1

        # MARKS
        dark_mask = cv2.inRange(gray, 0, 60)
        contours,_ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if 80 < cv2.contourArea(c) < 400:
                marks_score += 1

        # DARK CIRCLES
        eye = gray[int(0.45*h):int(0.65*h), int(0.3*w):int(0.7*w)]
        cheek_ref = gray[int(0.4*h):int(0.7*h), int(0.2*w):int(0.8*w)]
        if np.mean(eye) < np.mean(cheek_ref) - 15:
            dark_votes += 1

        # LIP
        lips = gray[int(0.75*h):h, int(0.3*w):int(0.7*w)]
        if np.mean(lips) < np.mean(gray) - 18:
            lip_votes += 1

    f_oil = np.mean(f_oil_vals)
    cheek_tex = np.mean(cheek_tex_vals)
    cheek_edges = np.mean(cheek_edge_vals)
    chin_oil = np.mean(chin_oil_vals)

    if f_oil > 185 and cheek_edges < 20:
        skin = "Oily"
    elif f_oil > 170 and cheek_edges > 22:
        skin = "Combination"
    elif cheek_edges > 25:
        skin = "Dry"
    else:
        skin = "Normal"

    if oily_input == "Yes" and skin == "Normal":
        skin = "Combination"
    if dry_input == "Yes" and skin == "Normal":
        skin = "Dry"

    concerns = []
    explanations = []

    if acne_count >= 4:
        concerns.append("ACNE 🔴")
        explanations.append("Inflamed acne clusters detected")

    if marks_score >= 5:
        concerns.append("ACNE MARKS 🟤")
        explanations.append("Post-acne scars detected")

    if abs(f_oil - chin_oil) + cheek_tex > 70:
        concerns.append("PIGMENTATION ⚫")
        explanations.append("Uneven tone detected")

    if dark_votes > len(faces)//2:
        concerns.append("DARK CIRCLES 👁️")
        explanations.append("Under-eye darker than cheeks")

    if lip_votes > len(faces)//2:
        concerns.append("LIP PIGMENTATION 💄")
        explanations.append("Lip darker than skin")

    score = int(75 - (cheek_tex/5 + cheek_edges/6))
    score = max(60, min(80, score))

    result = (skin, score, concerns, explanations)

    last_face_hash = face_hash
    last_result = result

    return result


# -------- UI --------
@app.route('/')
def index():
    return render_template_string("""
<html>
<body style="background:#0f2027;color:white;text-align:center;font-family:Arial;">

<h1>✨ SKIN AI ANALYZER ✨</h1>

<video id="video" autoplay playsinline width="90%"></video>
<canvas id="canvas" style="display:none;"></canvas>

<br><br>

<label>Oily after 2-3 hrs?</label>
<select id="oil"><option>No</option><option>Yes</option></select>

<label>Dry patches?</label>
<select id="dry"><option>No</option><option>Yes</option></select>

<br><br>

<button onclick="scan()">SCAN</button>
<p id="status"></p>

<div id="result"></div>

<script>
const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => { video.srcObject = stream; });

function capture(){
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video,0,0);
    return canvas.toDataURL("image/jpeg");
}

function scan(){
let s=document.getElementById("status");

let frames=[];
let count=0;

let interval=setInterval(()=>{
frames.push(capture());
count++;

s.innerHTML="📸 Capturing "+count+"/5";

if(count==5){
clearInterval(interval);

fetch('/capture',{
method:"POST",
headers:{'Content-Type':'application/json'},
body:JSON.stringify({
frames:frames,
oily:document.getElementById("oil").value,
dry:document.getElementById("dry").value
})
})
.then(r=>r.json())
.then(d=>{
document.getElementById("result").innerHTML=d.html;
s.innerHTML="🔒 RESULT LOCKED ✅";
});
}
},200);
}
</script>

</body>
</html>
""")


# -------- CAPTURE --------
@app.route('/capture', methods=['POST'])
def capture():
    data = request.json

    faces = []

    for img in data["frames"]:
        image_data = img.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = face_cascade.detectMultiScale(gray,1.3,5)

        if len(det)>0:
            x,y,w,h = max(det, key=lambda f:f[2]*f[3])
            faces.append(frame[y:y+h, x:x+w])

    if len(faces)==0:
        return jsonify({"html":"⚠️ NO FACE DETECTED"})

    skin,score,concerns,explanations = analyze(faces,data["oily"],data["dry"])

    products = {
        "Dry":["🧼 Hydrating Cleanser","💧 Hyaluronic Serum","🧴 Ceramide Moisturizer","☀️ Cream Sunscreen"],
        "Oily":["🧼 Gel Cleanser","💧 Niacinamide","🧴 Oil-free Moisturizer","☀️ Matte Sunscreen"],
        "Combination":["🧼 Gentle Cleanser","💧 Niacinamide","🧴 Light Moisturizer","☀️ Non-greasy Sunscreen"],
        "Normal":["🧼 Gentle Cleanser","💧 Vitamin C","🧴 Moisturizer","☀️ Sunscreen"]
    }

    diet = {
        "Dry":["💧 More water","🥑 Healthy fats","🥜 Omega-3","🚫 Less caffeine"],
        "Oily":["❌ No junk","🍎 Low GI foods","🌰 Zinc","🍵 Green tea"],
        "Combination":["🍽️ Balanced diet","🍓 Fruits","💦 Hydration","🚫 Less oil"],
        "Normal":["💧 Water","🥗 Clean food","🍽️ Balance","🚫 Less junk"]
    }

    roast = {
        "Dry":"💀 YOUR SKIN IS THIRSTY AF 💧",
        "Oily":"😭 OIL FACTORY RUNNING FULL TIME 🛢️",
        "Combination":"😵 YOUR SKIN IS CONFUSED BRO",
        "Normal":"😎 SKIN FLEX LEVEL MAX ✨"
    }

    html=f"<h2>{skin} SKIN</h2><h2>{score}/100</h2>"

    if concerns:
        html+="<ul>"+''.join(f"<li>{c}</li>" for c in concerns)+"</ul>"

    html+="<ul>"+''.join(f"<li>{e}</li>" for e in explanations)+"</ul>"

    html+="<h3>PRODUCTS</h3>"+'<br>'.join(products[skin])
    html+="<h3>DIET</h3><ul>"+''.join(f"<li>{d}</li>" for d in diet[skin])+"</ul>"

    if "LIP PIGMENTATION 💄" in concerns:
        html+="💄 Lip Balm<br>"
    if "DARK CIRCLES 👁️" in concerns:
        html+="👁️ Eye Cream<br>"

    html+=f"<p>{roast[skin]}</p>"

    return jsonify({"html":html})


if __name__ == "__main__":
    app.run(port=5001, debug=False)
