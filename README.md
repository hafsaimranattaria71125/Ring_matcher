# Ring_matcher
This project implements a ring matching system capable of identifying similar rings from a catalog, even when the query image includes a hand.It still has errors and needs improvement. It leverages computer vision models like YOLO for object detection, Segment Anything Model (SAM) for precise segmentation, CLIP for embedding and similarity search, and Real-ESRGAN for image enhancement.

✨ Features
Hand-Agnostic Ring Extraction: Utilizes YOLO and SAM to accurately segment and extract the ring from images, even if it's being worn on a hand.
Rotation-Invariant Matching: Employs multiple rotations during embedding to ensure accurate similarity matching regardless of the ring's orientation in the query image.
Two-Stage Similarity Search: Identifies both 'Exact' structural matches and 'Very Similar' design matches from a catalog of rings.
Image Upscaling (Optional): Integrates Real-ESRGAN for enhancing the quality of extracted rings, potentially improving matching accuracy for low-resolution inputs.
Catalog Indexing: Efficiently indexes a collection of ring images to create a searchable database of embeddings.
🛠️ Technologies Used
Python: Main programming language.
PyTorch: Deep learning framework.
OpenCV: Computer Vision tasks.
PIL (Pillow): Image processing.
NumPy: Numerical operations.
segment-anything (SAM): For precise image segmentation.
CLIP: For generating image embeddings.
Real-ESRGAN: For super-resolution image enhancement.
Ultralytics YOLO: For object detection (e.g., detecting rings on hands).
scikit-learn: For cosine similarity calculations.
matplotlib: For visualization of results.
🚀 Setup and Usage
1. Environment Setup
First, the necessary libraries and models need to be installed and downloaded. This is handled by the initial cells in the notebook.

Installations: torch, torchvision, segment-anything, basicsr, gfpgan, facexlib, Real-ESRGAN, CLIP, ultralytics.
Model Downloads: SAM (Segment Anything Model) weights and Real-ESRGAN weights are downloaded.
2. Load Models
The project loads several pre-trained models:

SAM: For segmenting the ring from the background/hand.
CLIP: To generate embeddings (numerical representations) of the rings for similarity comparison.
Real-ESRGAN: (Optional) For upscaling images.
YOLO: For initial detection of rings within an image, especially on a hand.
3. Upload Catalog Images
Upload your catalog of ring images to populate the database. The system will process these images and create embeddings for each ring, which are then used for matching.

4. Query Image Upload and Matching
Upload a single query image. This can be a professional product shot or a photo of a ring being worn on a hand.

How it works:
Hand Detection & Ring Extraction: If a hand is detected, the system intelligently removes it using a combination of YOLO and SAM, isolating just the ring.
Rotation-Invariant Embedding: The extracted ring image is rotated and multiple embeddings are generated to ensure robustness against orientation differences.
Similarity Search: The query ring's embeddings are compared against the catalog embeddings using cosine similarity.
Two-Stage Matching: The best matches are identified, categorizing them as 'Exact', 'Very Similar', or 'Similar' based on a confidence score.
Visualization: The original query image, the extracted ring, and the top matching rings from the catalog are displayed.
💡 Key Functions
get_sam_mask(image_bgr): Uses SAM to generate a mask for the main object in an image.
process_for_clip(image_bgr, mask=None): Preprocesses an image for CLIP, optionally applying a mask.
get_embedding(pil_img): Generates a CLIP embedding for a given PIL image.
extracted_actual_ring(image_path, predictor, realesrgan_model=None, yolo_model=None): The core function for robust ring extraction, handling hand removal and segmentation.
is_clean_catalog_image(pil_img): Checks if an image is a clean product shot or if it contains a hand.
rotate_image_pil(img, angle): Rotates a PIL image.
rotation_invariant_matching(query_embs, db_embs, top_k=5, threshold=0.65): Performs rotation-invariant cosine similarity matching
