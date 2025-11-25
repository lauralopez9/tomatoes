from ultralytics import YOLO

# Cargar el modelo guardado
model = YOLO('modelos_entrenados/SegmentacionYolo.pt')

# Usar para predecir
results = model('C:/Users/ll529/Downloads/uu.jpg')
results[0].show()