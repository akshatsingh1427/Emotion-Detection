import cv2
import numpy as np
import tensorflow as tf
import time


try:
    model = tf.keras.models.load_model('emotion_model.h5')
    print("✅ Model loaded successfully!")
except:
    print("❌ Error: Model not found. Please run train.py first")
    exit()

emotion_data = {
    'Angry': {'label': 'Angry', 'color': (0, 0, 255), 'display': '😠 Angry', 'short': 'A'},      # Red
    'Disgust': {'label': 'Disgust', 'color': (0, 128, 0), 'display': '🤢 Disgust', 'short': 'D'},  # Green
    'Fear': {'label': 'Fear', 'color': (128, 0, 128), 'display': '😨 Fear', 'short': 'F'},        # Purple
    'Happy': {'label': 'Happy', 'color': (0, 255, 255), 'display': '😊 Happy', 'short': 'H'},     # Yellow
    'Sad': {'label': 'Sad', 'color': (255, 0, 255), 'display': '😢 Sad', 'short': 'S'},          # Pink
    'Surprise': {'label': 'Surprise', 'color': (0, 165, 255), 'display': '😲 Surprise', 'short': 'SU'}, # Orange
    'Neutral': {'label': 'Neutral', 'color': (255, 0, 0), 'display': '😐 Neutral', 'short': 'N'}  # Blue
}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam")
    exit()

# Set window to full screen
cv2.namedWindow('Emotion Detector', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Emotion Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n" + "="*50)
print("🎭 EMOTION DETECTION SYSTEM")
print("="*50)
print("Controls:")
print("  ESC or q - Exit")
print("  f - Toggle fullscreen")
print("  s - Take screenshot")
print("  d - Toggle detection")
print("="*50)

# Variables
fps_time = 0
frame_count = 0
fps = 0
emotion_history = []
detection_enabled = True

# Store last predictions
last_predictions = np.zeros(7)

def draw_corner_ui(frame, faces_detected, dominant_emotion, emotion_probs=None):
    """Draw all UI elements in corners"""
    h, w = frame.shape[:2]
    
    # ========== TOP-LEFT CORNER: Info Panel ==========
    tl_x, tl_y = 20, 20
    panel_w, panel_h = 300, 180
    
    # Semi-transparent background for top-left panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (tl_x, tl_y), (tl_x + panel_w, tl_y + panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Title
    cv2.putText(frame, "EMOTION DETECTOR", (tl_x + 10, tl_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (tl_x + 10, tl_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Face count
    face_status = f"Faces: {faces_detected}" if detection_enabled else "Detection OFF"
    face_color = (0, 255, 0) if faces_detected > 0 else (0, 0, 255) if detection_enabled else (100, 100, 100)
    cv2.putText(frame, face_status, (tl_x + 10, tl_y + 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
    
    # Dominant emotion
    if dominant_emotion:
        dom_info = emotion_data[dominant_emotion]
        cv2.putText(frame, f"Dominant: {dom_info['display']}", (tl_x + 10, tl_y + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dom_info['color'], 2)
    
    # Detection status
    status_color = (0, 255, 0) if detection_enabled else (100, 100, 100)
    cv2.putText(frame, f"Detection: {'ON' if detection_enabled else 'OFF'}", 
               (tl_x + 10, tl_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # ========== TOP-RIGHT CORNER: Quick Emotion Bar ==========
    tr_x, tr_y = w - 320, 20
    bar_w, bar_h = 300, 50
    
    # Background for emotion bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (tr_x, tr_y), (tr_x + bar_w, tr_y + bar_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "CURRENT EMOTION:", (tr_x + 10, tr_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if emotion_probs is not None and len(emotion_probs) > 0:
        # Show top emotion
        top_idx = np.argmax(emotion_probs)
        emotion_names = list(emotion_data.keys())
        top_emotion = emotion_names[top_idx]
        top_color = emotion_data[top_emotion]['color']
        top_prob = emotion_probs[top_idx]
        
        cv2.putText(frame, f"{emotion_data[top_emotion]['display']} {top_prob:.1%}", 
                   (tr_x + 10, tr_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, top_color, 2)
    
    # ========== BOTTOM-LEFT CORNER: Emotion Legend ==========
    bl_x, bl_y = 20, h - 180
    legend_w, legend_h = 250, 160
    
    # Background for legend
    overlay = frame.copy()
    cv2.rectangle(overlay, (bl_x, bl_y), (bl_x + legend_w, bl_y + legend_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "EMOTION LEGEND:", (bl_x + 10, bl_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw emotion squares
    y_offset = bl_y + 50
    for i, (emotion, info) in enumerate(emotion_data.items()):
        # Color square
        cv2.rectangle(frame, (bl_x + 10, y_offset - 15), (bl_x + 30, y_offset + 5), info['color'], -1)
        # Emotion name
        cv2.putText(frame, info['display'], (bl_x + 40, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 30
    
    # ========== BOTTOM-RIGHT CORNER: Probabilities ==========
    br_x, br_y = w - 350, h - 180
    prob_w, prob_h = 330, 160
    
    # Background for probabilities
    overlay = frame.copy()
    cv2.rectangle(overlay, (br_x, br_y), (br_x + prob_w, br_y + prob_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "EMOTION PROBABILITIES:", (br_x + 10, br_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if emotion_probs is not None and len(emotion_probs) > 0:
        emotion_names = list(emotion_data.keys())
        y_offset = br_y + 50
        
        # Sort emotions by probability (highest first)
        sorted_indices = np.argsort(emotion_probs)[::-1]
        
        for idx in sorted_indices[:4]:  # Show top 4 only
            emotion = emotion_names[idx]
            prob = emotion_probs[idx]
            color = emotion_data[emotion]['color']
            
            # Emotion name
            cv2.putText(frame, f"{emotion[:8]:8}", (br_x + 10, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Probability bar
            bar_length = int(prob * 100)
            cv2.rectangle(frame, (br_x + 100, y_offset - 8),
                         (br_x + 100 + bar_length, y_offset + 8), color, -1)
            
            # Percentage
            cv2.putText(frame, f"{prob:.1%}", (br_x + 220, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 30
    
    return frame

def draw_face_info(frame, x, y, w, h, emotion, confidence, color):
    """Draw minimal info around face"""
    # Simple rectangle around face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Small label at top of face
    cv2.rectangle(frame, (x, max(0, y-25)), (x+80, y), color, -1)
    cv2.putText(frame, f"{emotion[:3]} {confidence:.0%}", 
               (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Small dots for facial points (non-intrusive)
    cv2.circle(frame, (x + w//2, y + h//3), 3, color, -1)  # Between eyes
    cv2.circle(frame, (x + w//4, y + 2*h//3), 3, color, -1)  # Left mouth
    cv2.circle(frame, (x + 3*w//4, y + 2*h//3), 3, color, -1)  # Right mouth

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Calculate FPS
    current_time = time.time()
    if fps_time > 0:
        fps = 0.9 * fps + 0.1 * (1 / (current_time - fps_time))
    fps_time = current_time
    
    # Initialize variables for this frame
    current_emotion = None
    current_predictions = last_predictions.copy()
    faces_detected = 0
    
    if detection_enabled:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces_detected = len(faces)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 for model
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = np.reshape(face_normalized, (1, 48, 48, 1))
            
            # Predict emotion
            predictions = model.predict(face_input, verbose=0)[0]
            current_predictions = predictions
            
            # Get emotion
            emotion_idx = np.argmax(predictions)
            emotion_names = list(emotion_data.keys())
            detected_emotion = emotion_names[emotion_idx]
            confidence = predictions[emotion_idx]
            
            # Update emotion history
            emotion_history.append(detected_emotion)
            if len(emotion_history) > 20:
                emotion_history.pop(0)
            
            # Store current emotion
            current_emotion = detected_emotion
            
            # Get color for this emotion
            color = emotion_data[detected_emotion]['color']
            
            # Draw minimal face info
            draw_face_info(frame, x, y, w, h, detected_emotion, confidence, color)
    
    # Calculate dominant emotion from history
    dominant_emotion = None
    if emotion_history:
        dominant_emotion = max(set(emotion_history), key=emotion_history.count)
    
    # Draw all UI in corners
    frame = draw_corner_ui(frame, faces_detected, dominant_emotion, current_predictions)
    
    # Show frame
    cv2.imshow('Emotion Detector', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Exit on 'q' or ESC
    if key == ord('q') or key == 27:
        print("\n🛑 Closing emotion detection...")
        break
    
    # Toggle full screen on 'f'
    elif key == ord('f'):
        current_prop = cv2.getWindowProperty('Emotion Detector', cv2.WND_PROP_FULLSCREEN)
        if current_prop == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty('Emotion Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("📺 Switched to windowed mode")
        else:
            cv2.setWindowProperty('Emotion Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("📺 Switched to full screen mode")
    
    # Take screenshot on 's'
    elif key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved as {filename}")
    
    # Toggle face detection on 'd'
    elif key == ord('d'):
        detection_enabled = not detection_enabled
        status = "ON" if detection_enabled else "OFF"
        print(f"👁️ Face detection: {status}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✅ Program terminated successfully!")
