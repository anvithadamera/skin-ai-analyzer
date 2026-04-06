# skin_ai_final_complete_ultra_FINAL.py

from flask import Flask, render_template_string, Response, jsonify, request
import cv2
import numpy as np
import time
import base64

app = Flask(__name__)

# -------- CAMERA --------
# Removed cv2.VideoCapture, browser handles camera

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_face = None
scan_line = 0

# -------- FRAME --------
def decode_image(img_data):
    # Decode base64 image from browser
    img_bytes = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

# -------- DETECT --------
def detect_concerns(face):

    face = cv2.bilateralFilter(face, 9, 75, 75)

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3,3), np.uint8)

    acne_points = []
    marks_points = []

    red_mask = cv2.inRange(hsv,(0,150,120),(10,255,255)) + cv2.inRange(hsv,(170,150,120),(180,255,255))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    contours,_ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    acne = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 60 < area < 200:
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4*np.pi*(area/(peri*peri))
            if circ > 0.4:
                acne += 1
                x,y,w,h = cv2.boundingRect(c)
                acne_points.append((x+w//2, y+h//2))

    dark_mask = cv2.inRange(gray,0,50)

    contours,_ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marks = 0
    for c in contours:
        if 70 < cv2.contourArea(c) < 300:
            marks += 1
            x,y,wc,hc = cv2.boundingRect(c)
            marks_points.append((x+wc//2, y+hc//2))

    pigmentation = np.std(gray)

    return acne, marks, pigmentation, gray, acne_points, marks_points

# -------- ANALYZE --------
def analyze(face, oily, dry):

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    texture = np.std(gray)
    reflection = np.percentile(gray,95)
    brightness = np.mean(gray)

    acne, marks, pigmentation, gray, acne_pts, marks_pts = detect_concerns(face)

    h, w = gray.shape

    eye = gray[int(0.45*h):int(0.65*h), int(0.3*w):int(0.7*w)]
    cheek = gray[int(0.4*h):int(0.7*h), int(0.2*w):int(0.8*w)]

    dark_circles = False
    if eye.size>0 and cheek.size>0:
        if np.mean(eye) < np.mean(cheek) - 12:
            dark_circles = True

    lips = gray[int(0.75*h):h, int(0.3*w):int(0.7*w)]
    lip_pig = False
    if lips.size>0 and np.mean(lips) < brightness - 15:
        lip_pig = True

    # ✅ FIXED SKIN TYPES
    if oily == "Yes" and dry == "Yes":
        skin = "Combination"
    elif oily == "No" and dry == "No" and acne < 3 and marks < 3 and pigmentation < 10:
        skin = "Normal"
    else:
        if reflection > 200:
            skin = "Oily"
        elif texture > 55:
            skin = "Dry"
        else:
            skin = "Combination"

    if texture > 65 and pigmentation > 15:
        skin = "Sensitive"

    concerns = []

    if acne > 5:
        concerns.append("ACNE 🔴")
    if marks > 3:
        concerns.append("ACNE MARKS 🟤")
    if pigmentation > 12:
        concerns.append("PIGMENTATION ⚫")
    if dark_circles:
        concerns.append("DARK CIRCLES 👁️")
    if lip_pig:
        concerns.append("LIP PIGMENTATION 💄")

    score = 80
    if "ACNE 🔴" in concerns: score -= 15
    if "ACNE MARKS 🟤" in concerns: score -= 10
    if "PIGMENTATION ⚫" in concerns: score -= 10
    if "DARK CIRCLES 👁️" in concerns: score -= 5
    if "LIP PIGMENTATION 💄" in concerns: score -= 5
    if texture < 45: score += 5
    if 120 < brightness < 170: score += 5
    score = max(40, min(score, 95))

    return skin, score, concerns, acne_pts, marks_pts, dark_circles, lip_pig

# -------- UI --------
@app.route('/')
def index():
    return render_template_string("""
<html>
<head>
<style>
body{margin:0;background:#0f2027;color:white;font-family:Arial;}
.container{display:flex;height:100vh;}
.left{width:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px;}
.right{width:50%;padding:30px;background:rgba(255,255,255,0.05);overflow:auto;}
h2,h3{text-decoration:underline;}
button{padding:12px 25px;border-radius:20px;background:#00c6ff;color:white;border:none;}
</style>
</head>

<body>

<h1 style="text-align:center;">✨ SKIN AI ANALYZER ✨</h1>

<div class="container">

<div class="left">
<video id="video" width="90%" autoplay playsinline></video><br>

<label>Does your skin get oily after 2-3 hrs?</label>
<select id="oil"><option>No</option><option>Yes</option></select>

<label>Do you have dry patches?</label>
<select id="dry"><option>No</option><option>Yes</option></select>

<button onclick="scan()">SCAN</button>
<p id="status"></p>
</div>

<div class="right" id="result">
<h2>RESULT</h2>
</div>

</div>

<script>
const video = document.getElementById('video');
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => { video.srcObject = stream; })
.catch(err => { alert('Cannot access camera: ' + err); });

function scan(){
let s=document.getElementById("status");
s.innerHTML="3...";
setTimeout(()=>{s.innerHTML="2...";},700);
setTimeout(()=>{s.innerHTML="1...";},1400);

setTimeout(()=>{
s.innerHTML="📸 Capturing...";

let canvas=document.createElement('canvas');
canvas.width=video.videoWidth;
canvas.height=video.videoHeight;
canvas.getContext('2d').drawImage(video,0,0);

fetch('/capture',{
method:"POST",
headers:{'Content-Type':'application/json'},
body:JSON.stringify({
oily:document.getElementById("oil").value,
dry:document.getElementById("dry").value,
image:canvas.toDataURL('image/jpeg')
})
})
.then(r=>r.json())
.then(d=>{
s.innerHTML="🧠 Analyzing...";
setTimeout(()=>{
document.getElementById("result").innerHTML=d.html;
s.innerHTML="✅ Done";
},1200);
});
},2100);
}
</script>

</body>
</html>
""")

# -------- CAPTURE --------
@app.route('/capture', methods=['POST'])
def capture():
    global current_face

    data = request.json

    if "image" in data:
        frame = decode_image(data["image"])
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
        if len(faces)>0:
            x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
            current_face = frame[y:y+h, x:x+w]
        else:
            current_face = None

    if current_face is None:
        return jsonify({"html":"⚠️ NO FACE DETECTED"})

    skin, score, concerns, _, _, _, _ = analyze(current_face, data["oily"], data["dry"])

    products = {
        "Dry":[
            "🧼 Hydrating Cleanser",
            "💧 Hyaluronic/Glycerin Serum",
            "🧴 Ceramide Moisturizer (Episoft AC / Cetaphil)",
            "☀️ Cream Sunscreen (UV Doux / Cetaphil)"
        ],
        "Oily":[
            "🧼 Gel/Foamy Cleanser",
            "💧 Niacinamide Serum",
            "🧴 Lightweight Gel Moisturizer (Sebamed)",
            "☀️ Matte Sunscreen (Acne UV)"
        ],
        "Combination":[
            "🧼 Gentle Cleanser",
            "💧 Niacinamide/Hydration based",
            "🧴 Lightweight Moisturizer (Episoft)",
            "☀️ Non-greasy Sunscreen (UV Doux)"
        ],
        "Normal":[
            "🧼 Gentle Cleanser",
            "💧 Vitamin C/Niacinamide",
            "🧴 Light Moisturizer",
            "☀️ Sunscreen (UV Doux/Cetaphil)"
        ],
        "Sensitive":[
            "🧼 Gentle Fragrance-Free Cleanser",
            "💧 Aloe/Panthenol Serum",
            "🧴 Barrier Repair Moisturizer",
            "☀️ Mineral Sunscreen"
        ]
    }

    diet = {
        "Dry":[
            "💧 Drink at least 2-3 liters of water daily",
            "🥑 Eat healthy fats (avocado, nuts, seeds, ghee)",
            "🥜 Include omega-3 foods (flax seeds, walnuts)",
            "🚫 Avoid excess caffeine and alcohol"
        ],
        "Oily":[
            "❌ Avoid sugary, fried, junk food",
            "🍎 Eat low-glycemic foods (veggies, whole grains)",
            "🌰 Include zinc-rich foods (pumpkin seeds, nuts)",
            "🍵 Drink green tea or detox drinks"
        ],
        "Combination":[
            "🍽️ Balanced diet (carbs + protein + healthy fats)",
            "🍓 Eat fresh fruits and veggies",
            "💦 Stay well hydrated",
            "🚫 Limit processed and oily foods"
        ],
        "Normal":[
            "💧 Drink 2-3 liters of water",
            "🥗 Eat vitamin-rich foods",
            "🍽️ Maintain balanced diet",
            "🚫 Avoid excess junk or sugar"
        ],
        "Sensitive":[
            "🌿 Eat anti-inflammatory foods (turmeric, ginger)",
            "🥛 Include probiotics (curd, yogurt)",
            "🚫 Avoid spicy processed foods",
            "🍓 Eat antioxidant-rich foods (berries, leafy greens)"
        ]
    }

    roasts = {
        "Dry":"💀 YOUR SKIN IS SO DRY, IT'S LITERALLY SCREAMING FOR HYDRATION 💧",
        "Oily":"😭 YOUR SKIN LOVES TO BE EXTRA, A LITTLE TOO EXTRA OILY 🛢️",
        "Combination":"😵‍💫 YOUR SKIN CAN'T DECIDE WHAT IT WANTS",
        "Normal":"😎 YOUR SKIN IS SO UNPROBLEMATIC, IT'S HONESTLY HELLACIOUS ✨",
        "Sensitive":"⚡ YOUR SKIN REACTS FASTER THAN YOU IN ARGUMENTS 😭"
    }

    html=f"<h2>SKIN TYPE: {skin}</h2>"
    html+=f"<h2>SKIN SCORE: {score}/100</h2>"
    html+=f"<div style='background:#333;height:20px;border-radius:10px;'><div style='width:{score}%;background:lime;height:100%;'></div></div>"

    if concerns:
        html+="<h3>SKIN CONCERNS</h3><ul>"
        html+=''.join(f"<li>{c}</li>" for c in concerns)
        html+="</ul>"

    html+="<h3>PRODUCT RECOMMENDATIONS</h3>"
    for p in products[skin]:
        html+=p+"<br>"

    if "LIP PIGMENTATION 💄" in concerns:
        html+="💄 Brightening SPF50 PA++++ Lip Balm<br>"
    if "DARK CIRCLES 👁️" in concerns:
        html+="👁️ Caffeine / Retinol Eye Cream<br>"

    html+="<h3>DIET RECOMMENDATIONS</h3><ul>"
    html+=''.join(f"<li>{d}</li>" for d in diet[skin])
    html+="</ul>"

    html+=f"<p>{roasts[skin]}</p>"

    return jsonify({"html":html})

# Removed /video_feed route; browser handles camera now

if __name__=="__main__":
    app.run(debug=True)
