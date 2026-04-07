# skin_ai_dermatology_FULL_FINAL.py

from flask import Flask, render_template_string, Response, jsonify, request
import cv2
import numpy as np
import time

app = Flask(__name__)

# -------- CAMERA --------
camera = cv2.VideoCapture(0)  # Default camera for cross-device access

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_face = None
scan_line = 0

# -------- FRAME --------
def get_frame():
    ret, frame = camera.read()
    if not ret:
        time.sleep(0.1)
        ret, frame = camera.read()

    # 🔥 REMOVE BEAUTIFICATION (sharpen image)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    frame = cv2.filter2D(frame, -1, kernel)

    return frame

# -------- DETECT --------
def detect_concerns(face):

    face = cv2.bilateralFilter(face, 9, 75, 75)

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    kernel = np.ones((3,3), np.uint8)

    acne_points = []
    marks_points = []

    # STRICT FACE REGION (avoid lips/edges)
    roi_mask = np.zeros_like(gray)
    cv2.rectangle(roi_mask, (int(0.15*w), int(0.1*h)), (int(0.85*w), int(0.75*h)), 255, -1)

    # 🔴 ACNE DETECTION
    red_mask = cv2.inRange(hsv,(0,150,120),(10,255,255)) + cv2.inRange(hsv,(170,150,120),(180,255,255))
    red_mask = cv2.bitwise_and(red_mask, roi_mask)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    contours,_ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    acne = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 80 < area < 250:
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4*np.pi*(area/(peri*peri))
            if circ > 0.5:
                acne += 1
                x,y,wc,hc = cv2.boundingRect(c)
                acne_points.append((x+wc//2, y+hc//2))

    # 🌑 GRID BASED PIGMENTATION
    grid_size = 6
    patch_vals = []

    for i in range(grid_size):
        for j in range(grid_size):
            patch = gray[int(i*h/grid_size):int((i+1)*h/grid_size),
                         int(j*w/grid_size):int((j+1)*w/grid_size)]
            if patch.size > 0:
                patch_vals.append(np.mean(patch))

    pigmentation = np.std(patch_vals)

    # 🟤 MARKS
    dark_mask = cv2.inRange(gray,0,50)
    contours,_ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marks = 0
    for c in contours:
        if 80 < cv2.contourArea(c) < 300:
            marks += 1
            x,y,wc,hc = cv2.boundingRect(c)
            marks_points.append((x+wc//2, y+hc//2))

    return acne, marks, pigmentation, gray, acne_points, marks_points

# -------- ANALYZE --------
def analyze(face, oily, dry):

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    texture = np.std(gray)
    brightness = np.mean(gray)

    # LIGHTING CHECK
    lighting_msg = None
    if brightness < 70:
        lighting_msg = "⚠️ Low lighting detected"
    elif brightness > 200:
        lighting_msg = "⚠️ Overexposed lighting detected"

    # DRY SKIN (edges)
    edges = cv2.Canny(gray,50,150)
    dryness = np.mean(edges)

    # OIL (T-ZONE)
    h, w = gray.shape
    tzone = gray[int(0.1*h):int(0.4*h), int(0.3*w):int(0.7*w)]
    reflection = np.percentile(tzone,95)

    acne, marks, pigmentation, gray, acne_pts, marks_pts = detect_concerns(face)

    # DARK CIRCLES
    eye = gray[int(0.45*h):int(0.65*h), int(0.3*w):int(0.7*w)]
    cheek = gray[int(0.4*h):int(0.7*h), int(0.2*w):int(0.8*w)]

    dark_circles = False
    if eye.size>0 and cheek.size>0:
        if np.mean(eye) < np.mean(cheek) - 12:
            dark_circles = True

    # LIP PIGMENTATION
    lips = gray[int(0.75*h):h, int(0.3*w):int(0.7*w)]
    lip_pig = False
    if lips.size>0 and np.mean(lips) < brightness - 15:
        lip_pig = True

    # SKIN TYPE LOGIC FIXED
    if oily == "Yes" and dry == "Yes":
        skin = "Combination"
    elif reflection > 200:
        skin = "Oily"
    elif dryness > 20:
        skin = "Dry"
    else:
        skin = "Normal"

    if texture > 65 and pigmentation > 15:
        skin = "Sensitive"

    # CONCERNS
    concerns = []
    explanations = []

    if acne > 5:
        concerns.append("ACNE 🔴")
        explanations.append("Inflammatory acne lesions detected via red pixel clustering")

    if marks > 3:
        concerns.append("ACNE MARKS 🟤")
        explanations.append("Post-inflammatory hyperpigmentation detected")

    if pigmentation > 12:
        concerns.append("PIGMENTATION ⚫")
        explanations.append("Uneven melanin distribution observed")

    if dark_circles:
        concerns.append("DARK CIRCLES 👁️")
        explanations.append("Periorbital darkening detected")

    if lip_pig:
        concerns.append("LIP PIGMENTATION 💄")
        explanations.append("Lip tone darker than baseline")

    # SCORE
    score = 100
    score -= acne * 2
    score -= marks * 2
    score -= int(pigmentation)
    score -= int(dryness/5)

    score = max(40, min(score, 95))

    # CONFIDENCE
    confidence = min(95, int((len(acne_pts)+len(marks_pts))/10*100))

    return skin, score, concerns, acne_pts, marks_pts, dark_circles, lip_pig, explanations, confidence, lighting_msg

# -------- VIDEO --------
def generate_frames():
    global current_face, scan_line

    while True:
        frame = get_frame()
        frame = cv2.flip(frame,1)
        display = frame.copy()

        h_frame, w_frame, _ = frame.shape

        box_w = int(w_frame * 0.6)
        box_h = int(h_frame * 0.75)

        x1 = int((w_frame - box_w) / 2)
        y1 = int((h_frame - box_h) / 2)

        x2 = x1 + box_w
        y2 = y1 + box_h

        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if len(faces)>0:
            x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
            current_face = roi[y:y+h, x:x+w]

            skin, score, concerns, acne_pts, marks_pts, dark_circles, lip_pig, explanations, confidence, lighting_msg = analyze(current_face,"No","No")

            # DRAW ACNE
            for (px,py) in acne_pts:
                cv2.circle(display,(x1+x+px,y1+y+py),6,(0,0,255),2)

            # MARKS
            for (px,py) in marks_pts:
                cv2.circle(display,(x1+x+px,y1+y+py),6,(0,165,255),2)

            # ANIME SCAN LINE
            scan_line = (scan_line + 10) % (y2-y1)
            cv2.line(display,(x1,y1+scan_line),(x2,y1+scan_line),(255,255,0),2)

        cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),3)

        _,buffer=cv2.imencode('.jpg',display)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')

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
<img src="/video_feed" width="90%"><br>

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
function scan(){
let s=document.getElementById("status");
s.innerHTML="3...";
setTimeout(()=>{s.innerHTML="2...";},700);
setTimeout(()=>{s.innerHTML="1...";},1400);

setTimeout(()=>{
s.innerHTML="📸 Capturing...";

fetch('/capture',{
method:"POST",
headers:{'Content-Type':'application/json'},
body:JSON.stringify({
oily:document.getElementById("oil").value,
dry:document.getElementById("dry").value
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

    if current_face is None:
        return jsonify({"html":"⚠️ NO FACE DETECTED"})

    data = request.json

    skin, score, concerns, _, _, _, _, explanations, confidence, lighting_msg = analyze(current_face, data["oily"], data["dry"])

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
    html+=f"<h3>CONFIDENCE: {confidence}%</h3>"

    if lighting_msg:
        html+=f"<p>{lighting_msg}</p>"

    html+=f"<div style='background:#333;height:20px;border-radius:10px;'><div style='width:{score}%;background:lime;height:100%;'></div></div>"

    if concerns:
        html+="<h3>SKIN CONCERNS</h3><ul>"
        html+=''.join(f"<li>{c}</li>" for c in concerns)
        html+="</ul>"

    if explanations:
        html+="<h3>AI ANALYSIS</h3><ul>"
        html+=''.join(f"<li>{e}</li>" for e in explanations)
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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
