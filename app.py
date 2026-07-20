
import gradio as gr
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from Bio import Entrez, SeqIO
import pickle

# --- 1. Redefine focal_loss_fixed for model loading ---
def focal_loss_fixed(y_true, y_pred):
    gamma=2.0
    alpha=0.5
    y_true = tf.cast(y_true, tf.float32)
    bce = K.binary_crossentropy(y_true, y_pred)
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    focal_weight = alpha * K.pow(1.0 - p_t, gamma)
    return K.mean(focal_weight * bce)

# --- 2. Load the trained model and LabelEncoder ---
model = load_model("phage_lifecycle_model.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# --- 3. Define mapping and encode_data function ---
mapping = {"A":[1,0,0,0], "T":[0,1,0,0], "C":[0,0,1,0], "G":[0,0,0,1], "N":[0,0,0,0]}

def encode_data(frags):
    bases = ['A', 'T', 'C', 'G']
    kmers = [a+b+c for a in bases for b in bases for c in bases]
    kmer_map = {k: i for i, k in enumerate(kmers)}

    cnn_in = []
    for f in frags:
        encoded = [kmer_map.get(f[i:i+3], 0) for i in range(len(f)-2)]
        cnn_in.append(encoded)

    lstm_in = []
    for f in frags:
        arr = np.array(list(f))
        sig = []
        for i in range(0, len(f) - 500 + 1, 250):
          window = arr[i:i+500]
          g = np.count_nonzero(window == 'G')
          c = np.count_nonzero(window == 'C')
          valid_bases = np.count_nonzero((window == 'G') | (window == 'C') |
                                        (window == 'A') | (window == 'T'))
          content = (g + c) / valid_bases if valid_bases > 0 else 0
          skew = (g - c) / (g + c) if (g + c) > 0 else 0
          sig.append([content, skew])
        lstm_in.append(sig)

    return np.array(cnn_in)[..., np.newaxis], np.array(lstm_in)

# --- 4. Define fetch_sequence function ---
def fetch_sequence(accession):
    Entrez.email = "xyz@xyz.com"
    for attempt in range(3):
        try:
            handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            sequence = str(record.seq).upper()
            name = record.description.split(",")[0]
            return sequence, name
        except Exception as e:
            time.sleep(1)
    return None
# --- 5. Define predict_with_preview function ---
def predict_with_preview(accession_id):
    try:
        result = fetch_sequence(accession_id)
        if not result:
            return "Error", "0%", "NCBI Fetch Failed", "N/A", "N/A", None, "N/A"
        raw_seq, phage_name = result
        dna_preview = raw_seq[:200]
        full_len = len(raw_seq)
        all_probs = []
        window_size = 50000
        step_size = 25000

        for start in range(0, max(1, full_len - window_size + 1), step_size):
            chunk = raw_seq[start : start + window_size].ljust(window_size, "N")
            c_in, l_in = encode_data([chunk])
            prob = model.predict([c_in, l_in], verbose=0)[0][0]
            all_probs.append(prob)
        if len(all_probs) == 0:
            return "Error", "0%", "Sequence too short", "N/A", phage_name.replace(accession_id, "").strip(), "Insufficient data", None

        final_prob=np.mean(all_probs)

        label = le.inverse_transform([int(final_prob > 0.5)])[0]
        confidence = final_prob if final_prob > 0.5 else (1 - final_prob)
        confidence = confidence **0.6
        confidence = min(confidence, 0.95)
        img_path="lifecycle.png"
        if label.lower() == "virulent":
            explanation = "Virulent (lytic) phages infect and rapidly lyse bacterial cells, making them suitable candidates for phage therapy applications."
        else:
            explanation = "Temperate phages follow the lysogenic cycle and can integrate into host DNA, making them less suitable for therapy."

        return (f"{label}", f"{confidence:.2%}", f"Length: {full_len} bp", dna_preview, phage_name.replace(accession_id, "").strip(), explanation, img_path)

    except Exception as e:
        return "System Error", "0%", str(e), "N/A", "N/A", "N/A", None

# --- 6. Define the Gradio Interface ---
demo = gr.Interface(
    fn=predict_with_preview,
    inputs=gr.Textbox(label="Enter Phage Accession (e.g., NC_000866.4)"), description="Enter a phage accession ID to predict lifecycle and view biological insights",
    outputs=[
        gr.Textbox(label="Predicted Lifecycle"),
        gr.Textbox(label="Confidence Score"),
        gr.Textbox(label="Genome Info"),
        gr.Textbox(label="DNA Preview (First 200bp)"),
        gr.Textbox(label="Phage Name"),              # ✅ NEW
        gr.Textbox(label="Biological Explanation"),  # ✅ NEW
        gr.Image(label="Lytic vs Lysogenic Lifecycle (Phage Therapy Context)")        # ✅ NEW
    ],
    title="Phage Lifecycle Prediction through Hybrid Deep Learning Approach"
)

# --- 7. Launch the Gradio app (optional for direct Colab run) ---
demo.launch() # This is commented out because it's not needed for app.py deployment itself
