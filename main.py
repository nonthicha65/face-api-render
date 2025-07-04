from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile
import requests
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # อนุญาตให้ Flutter เรียก API นี้ได้

@app.route('/verify-face', methods=['POST'])
def verify_face():
    try:
        user_id = request.form.get('user_id')
        uploaded_file = request.files['image']
        print("📥 เริ่มรับคำขอ")

        # 🔍 ดึง URL ภาพจาก Firestore (เปลี่ยนเป็น Firestore จริงใน production)
        # ในที่นี้จำลองเป็น dictionary แทน
        registered_faces = {
            "HR001": {
                "face_url": "https://firebasestorage.googleapis.com/v0/b/project-gps-c1f58.firebasestorage.app/o/faces%2FHR001.jpg?alt=media&token=424d76a5-8eef-4802-bce7-7489517f2622"
            },
            "HR002": {
                "face_url": "https://firebasestorage.googleapis.com/v0/b/project-gps-c1f58.firebasestorage.app/o/faces%2FHR002.jpg?alt=media&token=abc123"
            },
        }

        face_url = registered_faces.get(user_id, {}).get("face_url")
        print(f"🧾 user_id: {user_id}, ชื่อไฟล์: {uploaded_file.filename}")
        print(f"📦 face_url: {face_url}")

        if not face_url:
            return jsonify({'error': 'face_url not found for user'}), 400

        # ✅ โหลดรูปภาพที่ลงทะเบียน
        ori_img = requests.get(face_url).content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_ori:
            temp_ori.write(ori_img)
            ori_path = temp_ori.name

        # ✅ บันทึกภาพที่เพิ่งอัปโหลด
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_new:
            uploaded_file.save(temp_new.name)
            new_path = temp_new.name

        print("🔍 เปรียบเทียบใบหน้า...")

        result = DeepFace.verify(
            img1_path=ori_path,
            img2_path=new_path,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv"
        )

        print("📊 ผลลัพธ์:", result)

        # ✅ ลบไฟล์ temp เมื่อใช้เสร็จ
        os.remove(ori_path)
        os.remove(new_path)

        return jsonify({
            'verified': result["verified"],
            'distance': result["distance"],
            'threshold': result["threshold"],
            'model': result["model"],
            'detector_backend': result["detector_backend"],
            'similarity_metric': result["similarity_metric"],
            'facial_areas': result.get("facial_areas", {}),
            'time': result.get("time", 0)
        })

    except Exception as e:
        print(f"❗เกิดข้อผิดพลาด: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
