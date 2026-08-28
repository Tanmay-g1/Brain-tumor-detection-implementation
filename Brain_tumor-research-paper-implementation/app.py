"""
Step 15: Multi-Class Brain Tumor Classification Web Interface
Flask app for interactive predictions and XAI visualizations
Run: python app.py
Then open: http://127.0.0.1:5000
"""

import os
import io
import numpy as np
import cv2
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
MODEL_PATH = 'models/brain_tumor_multiclass.h5'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
NUM_CLASSES = 4
LAST_CONV = 'conv5_last'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    model = None

# ── Helper Functions ──────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def img_to_base64(img_array):
    """Convert numpy image to base64 string."""
    if isinstance(img_array, tf.Tensor):
        img_array = img_array.numpy()
    
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        if img_array.max() <= 1:
            img_array = np.uint8(img_array * 255)
        else:
            img_array = np.uint8(img_array)
    
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_str}"

def compute_gradcam(model, img_batch, class_idx):
    """Compute Grad-CAM overlay."""
    try:
        last_conv_layer = model.get_layer(LAST_CONV)
        
        feature_extractor = tf.keras.models.Model(
            inputs=model.inputs[0], outputs=last_conv_layer.output
        )
        
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        found = False
        for layer in model.layers:
            if found:
                x = layer(x)
            if layer.name == LAST_CONV:
                found = True
        classifier_model = tf.keras.models.Model(classifier_input, x)
        
        with tf.GradientTape() as tape:
            conv_outputs = feature_extractor(img_batch, training=False)
            tape.watch(conv_outputs)
            predictions = classifier_model(conv_outputs, training=False)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        
        h, w = IMG_SIZE
        hm_r = cv2.resize(heatmap, (w, h))
        colored = cv2.applyColorMap(np.uint8(255 * hm_r), cv2.COLORMAP_JET)
        orig_u8 = np.uint8(255 * img_batch[0])
        sup = cv2.addWeighted(orig_u8, 0.55, colored, 0.45, 0)
        
        return cv2.cvtColor(sup, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"GradCAM error: {e}")
        return None

def predict_fn_lime(images):
    """Predict probabilities for LIME."""
    return model.predict(images.astype(np.float32) / 255.0, verbose=0)

def compute_lime(img_u8, class_idx):
    """Compute LIME superpixel mask."""
    try:
        explainer = lime_image.LimeImageExplainer(random_state=42)
        exp = explainer.explain_instance(
            img_u8, predict_fn_lime, top_labels=NUM_CLASSES,
            hide_color=0, num_samples=300, random_seed=42
        )
        temp, mask = exp.get_image_and_mask(
            class_idx, positive_only=True, num_features=8, hide_rest=False
        )
        return mark_boundaries(temp / 255.0, mask, color=(1, 1, 0))
    except Exception as e:
        print(f"LIME error: {e}")
        return None

def compute_ig(model, img_batch, class_idx, num_steps=30):
    """Compute Integrated Gradients."""
    try:
        baseline = tf.zeros_like(img_batch)
        integrated_grads = None
        
        for step in range(num_steps):
            alpha = tf.constant(step / num_steps, dtype=tf.float32)
            interpolated = baseline + alpha * (img_batch - baseline)
            
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                predictions = model(interpolated, training=False)
            
            grads = tape.gradient(predictions[:, class_idx], interpolated)
            if integrated_grads is None:
                integrated_grads = grads
            else:
                integrated_grads += grads
        
        integrated_grads = (img_batch - baseline) * integrated_grads / num_steps
        saliency = tf.reduce_sum(tf.abs(integrated_grads[0]), axis=-1)
        saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
        return saliency.numpy()
    except Exception as e:
        print(f"IG error: {e}")
        return None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Upload image and get prediction + XAI visualizations."""
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Load and preprocess image
        img_pil = keras_image.load_img(io.BytesIO(file.read()), target_size=IMG_SIZE)
        img_01 = keras_image.img_to_array(img_pil) / 255.0
        img_u8 = np.uint8(img_01 * 255)
        img_batch = np.expand_dims(img_01, 0).astype(np.float32)
        img_batch_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        
        # Prediction
        pred = model.predict(img_batch, verbose=0)[0]
        pred_class_idx = np.argmax(pred)
        confidence = float(pred[pred_class_idx])
        
        # Prepare response
        result = {
            'predicted_class': CLASS_NAMES[pred_class_idx],
            'predicted_idx': int(pred_class_idx),
            'confidence': confidence,
            'all_predictions': {
                CLASS_NAMES[i]: float(pred[i])
                for i in range(NUM_CLASSES)
            },
            'original_image': img_to_base64(np.uint8(img_01 * 255))
        }
        
        # Generate XAI visualizations (if requested)
        xai_methods = request.form.get('xai_methods', 'gradcam,lime,ig').split(',')
        
        if 'gradcam' in xai_methods:
            gc = compute_gradcam(model, img_batch, pred_class_idx)
            if gc is not None:
                result['gradcam'] = img_to_base64(gc)
        
        if 'lime' in xai_methods:
            lime_img = compute_lime(img_u8, pred_class_idx)
            if lime_img is not None:
                result['lime'] = img_to_base64(np.uint8(lime_img * 255))
        
        if 'ig' in xai_methods:
            ig = compute_ig(model, img_batch_tf, pred_class_idx)
            if ig is not None:
                # Overlay on original
                fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
                ax.imshow(img_01, alpha=0.3)
                ax.imshow(ig, cmap='RdBu_r', vmin=0, vmax=1, alpha=0.8)
                ax.axis('off')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                img_str = base64.b64encode(buf.read()).decode()
                result['ig'] = f"data:image/png;base64,{img_str}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Multi-Class Brain Tumor Classification Web Interface")
    print("="*60)
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("\nFeatures:")
    print("  • Upload MRI images (JPG, PNG, GIF, BMP)")
    print("  • Get predictions with confidence scores")
    print("  • View Grad-CAM heatmaps")
    print("  • View LIME superpixel explanations")
    print("  • View Integrated Gradients saliency maps")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
