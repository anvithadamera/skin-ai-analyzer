# skin_ai_final_web_full.py

from flask import Flask, render_template_string, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)

# -------- ANALYSIS --------
def analyze(face, oily, dry):

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    texture = np.std(gray)
    brightness = np.mean(gray)
    reflection = np.percentile(gray,95)

    # --- DETECTION ---
    acne = np.sum(gray < 60) // 500
    marks = np.sum(gray < 80) // 800
    pigmentation = np.std(gray)

    h, w = gray.shape

    # DARK CIRCLES
    eye = gray[int(0.45*h):int(0.65*h), int(0.3*w):int(0.7*w)]
    dark_circles = eye.size>0 and np.mean(eye) < brightness - 12

    # LIP PIGMENTATION
    lips = gray[int(0.75*h):h, int(0.3*w):int(0.7*w)]
    lip_pig = lips.size>0 and np.mean(lips) < brightness - 15

    # SKIN TYPE
    if reflection > 210:
        skin = "Oily"
    elif texture > 55:
        skin = "Dry"
    elif abs(reflection - texture) < 20:
        skin = "Normal"
    else:
        skin = "Combination"

    if oily == "Yes" and skin != "Dry":
        skin = "Oily"
    if dry == "Yes" and skin != "Oily":
        skin = "Dry"

    # CONCERNS
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

    # SCORE (YOUR LOGIC)
    score = 80
    if "ACNE 🔴" in concerns: score -= 15
    if "ACNE MARKS 🟤" in concerns: score -= 10
    if "PIGMENTATION ⚫" in concerns: score -= 10
    if "DARK CIRCLES 👁️" in concerns: score -= 5
    if "LIP PIGMENTATION 💄" in concerns: score -= 5
    if texture < 45: score += 5
    if 120 < brightness < 170: score += 5
    score = max(40, min(score, 95))

    return skin, score, concerns


# -------- UI --------
@app.route('/')
def index():
    return render_template_string("""

<html>
<head>
<style>
body{
background:linear-gradient(135deg,#1e1e2f,#0f2027);
color:white;font-family:Arial;
}
.container{display:flex;height:90vh;}
.left{width:50%;padding:20px;text-align:center;}
.right{width:50%;padding:20px;overflow:auto;}

video{
width:90%;
border-radius:15px;
}

.card{
background:rgba(255,255,255,0.05);
padding:15px;
border-radius:15px;
}

button{
background:linear-gradient(45deg,#00c6ff,#0072ff);
border:none;padding:10px 20px;border-radius:20px;color:white;
}
</style>
</head>

<body>

<h1 style="text-align:center;">✨ SKIN AI ANALYZER ✨</h1>

<div class="container">

<div class="left card">
<video id="video" autoplay playsinline></video>
<canvas id="canvas" style="display:none;"></canvas><br><br>

<label>Does your skin get oily after 2-3 hrs?</label><br>
<select id="oil"><option>No</option><option>Yes</option></select><br><br>

<label>Do you have dry patches?</label><br>
<select id="dry"><option>No</option><option>Yes</option></select><br><br>

<button onclick="scan()">SCAN</button>
<p id="status"></p>

</div>

<div class="right card" id="result">
<h2>RESULT</h2>
</div>

</div>

<script>

const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
})
.catch(err => alert("Camera permission needed"));

function scan(){

let s=document.getElementById("status");

s.innerHTML="3...";
setTimeout(()=>{s.innerHTML="2...";},700);
setTimeout(()=>{s.innerHTML="1...";},1400);

setTimeout(()=>{

let canvas=document.getElementById("canvas");
let ctx=canvas.getContext("2d");

canvas.width=video.videoWidth;
canvas.height=video.videoHeight;

ctx.drawImage(video,0,0);

let img=canvas.toDataURL("image/jpeg");

fetch('/analyze',{
method:"POST",
headers:{'Content-Type':'application/json'},
body:JSON.stringify({
image:img,
oily:document.getElementById("oil").value,
dry:document.getElementById("dry").value
})
})
.then(r=>r.json())
.then(d=>{
document.getElementById("result").innerHTML=d.html;
s.innerHTML="✅ Done";
});

},2100);
}

</script>

</body>
</html>

""")

# -------- CAPTURE --------
@app.route('/analyze', methods=['POST'])
def capture():

    data = request.json

    img_data = data["image"].split(',')[1]
    img = base64.b64decode(img_data)

    npimg = np.frombuffer(img, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    skin, score, concerns = analyze(frame, data["oily"], data["dry"])

    # PRODUCTS
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
        ]
    }

    # DIET
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
        ]
    }

    # ROASTS
    roasts = {
        "Dry":"💀 YOUR SKIN IS SO DRY, IT'S LITERALLY SCREAMING FOR HYDRATION 💧",
        "Oily":"😭 YOUR SKIN LOVES TO BE EXTRA, A LITTLE TOO EXTRA OILY 🛢️",
        "Combination":"😵‍💫 YOUR SKIN CAN'T DECIDE WHAT IT WANTS",
        "Normal":"😎 YOUR SKIN IS SO UNPROBLEMATIC, IT'S HONESTLY HELLACIOUS ✨"
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


if __name__=="__main__":
    app.run()
