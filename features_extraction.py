import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops

def glcm_feature_extract(image_path):
    image = io.imread(image_path)
    image = (image * 255).astype(np.uint8) # Normalisasi Gambar untuk GLCM

    # Menghitung GLCM
    distances = [1]  # Jarak antar piksel
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Sudut arah (0°, 45°, 90°, 135°)
    glcm = graycomatrix(image, distances, angles, levels= 256, symmetric= False, normed= True)

    # Menghitung fitur tekstur
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'ASM': graycoprops(glcm, 'ASM').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean()
    }

    return list(features.values())