// Configuración
const API_URL = 'https://aplicaionwebtomatoes-czb9b4h2b0fjasfr.canadacentral-01.azurewebsites.net';
let selectedImage = null;
let stream = null;

// Elementos del DOM
const fileInput = document.getElementById('fileInput');
const selectFileBtn = document.getElementById('selectFileBtn');
const cameraBtn = document.getElementById('cameraBtn');
const uploadArea = document.getElementById('uploadArea');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeImgBtn = document.getElementById('removeImgBtn');
const processBtn = document.getElementById('processBtn');
const resultsSection = document.getElementById('resultsSection');
const segmentationResults = document.getElementById('segmentationResults');
const classificationResults = document.getElementById('classificationResults');
const cameraModal = document.getElementById('cameraModal');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const cancelCameraBtn = document.getElementById('cancelCameraBtn');

// Event listeners
selectFileBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
cameraBtn.addEventListener('click', openCamera);
removeImgBtn.addEventListener('click', removeImage);
processBtn.addEventListener('click', processImage);
closeCameraBtn.addEventListener('click', closeCamera);
cancelCameraBtn.addEventListener('click', closeCamera);
captureBtn.addEventListener('click', capturePhoto);

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    if (!selectedImage) {
        fileInput.click();
    }
});

// Funciones
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Por favor selecciona un archivo de imagen');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        selectedImage = e.target.result;
        previewImg.src = selectedImage;
        imagePreview.style.display = 'block';
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedImage = null;
    previewImg.src = '';
    imagePreview.style.display = 'none';
    processBtn.disabled = true;
    resultsSection.style.display = 'none';
    fileInput.value = '';
}

async function openCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment', // Cámara trasera en móviles
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        video.srcObject = stream;
        cameraModal.style.display = 'flex';
    } catch (error) {
        alert('Error al acceder a la cámara: ' + error.message);
        console.error('Error accessing camera:', error);
    }
}

function closeCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraModal.style.display = 'none';
    video.srcObject = null;
}

function capturePhoto() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    selectedImage = canvas.toDataURL('image/jpeg', 0.9);
    previewImg.src = selectedImage;
    imagePreview.style.display = 'block';
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
    
    closeCamera();
}

async function processImage() {
    if (!selectedImage) return;

    const modelo = document.querySelector('input[name="modelo"]:checked').value;
    const btnText = processBtn.querySelector('.btn-text');
    const btnLoader = processBtn.querySelector('.btn-loader');
    
    // Mostrar loading
    processBtn.disabled = true;
    btnText.textContent = 'Procesando...';
    btnLoader.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        const endpoint = modelo === 'segmentacion' ? '/api/segmentacion' : '/api/clasificacion';
        
        const response = await fetch(`${API_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                imagen: selectedImage
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error al procesar la imagen');
        }

        // Mostrar resultados
        if (modelo === 'segmentacion') {
            showSegmentationResults(data);
        } else {
            showClassificationResults(data);
        }

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Error:', error);
    } finally {
        // Restaurar botón
        processBtn.disabled = false;
        btnText.textContent = 'Procesar imagen';
        btnLoader.style.display = 'none';
    }
}

function showSegmentationResults(data) {
    segmentationResults.style.display = 'block';
    classificationResults.style.display = 'none';

    // Mostrar imagen con resultados
    const resultImg = document.getElementById('segmentationResultImg');
    resultImg.src = data.imagen_resultado;

    // Mostrar lista de detecciones
    const detectionsList = document.getElementById('detectionsList');
    detectionsList.innerHTML = '';

    if (data.detecciones.length === 0) {
        detectionsList.innerHTML = '<p style="text-align: center; color: var(--gray-dark);">No se detectaron tomates en la imagen</p>';
        return;
    }

    data.detecciones.forEach(det => {
        const item = document.createElement('div');
        item.className = 'detection-item';
        
        const claseColors = {
            'damaged': '#FFB3BA',
            'old': '#FFD3A5',
            'ripe': '#A8E6CF',
            'unripe': '#FFEAA7'
        };
        
        const color = claseColors[det.clase] || '#E8D5FF';
        
        item.style.borderLeftColor = color;
        item.innerHTML = `
            <h4>Tomate #${det.numero} - ${det.clase.charAt(0).toUpperCase() + det.clase.slice(1)}</h4>
            <p><strong>Confianza:</strong> ${(det.confianza * 100).toFixed(1)}%</p>
            <p><strong>Posición:</strong> (${det.bbox.x1.toFixed(0)}, ${det.bbox.y1.toFixed(0)}) a (${det.bbox.x2.toFixed(0)}, ${det.bbox.y2.toFixed(0)})</p>
        `;
        detectionsList.appendChild(item);
    });
}

function showClassificationResults(data) {
    classificationResults.style.display = 'block';
    segmentationResults.style.display = 'none';

    // Mostrar clase predicha
    const predictedClass = document.getElementById('predictedClass');
    predictedClass.textContent = data.clase_predicha;
    predictedClass.style.color = getClassColor(data.clase_predicha);

    // Mostrar confianza
    const predictionConfidence = document.getElementById('predictionConfidence');
    predictionConfidence.textContent = `${(data.confianza * 100).toFixed(1)}%`;
    predictionConfidence.style.color = getClassColor(data.clase_predicha);

    // Mostrar gráfico de probabilidades
    const probabilitiesChart = document.getElementById('probabilitiesChart');
    probabilitiesChart.innerHTML = '';

    const sortedProbs = Object.entries(data.probabilidades)
        .sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([clase, prob]) => {
        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        
        const label = document.createElement('div');
        label.className = 'probability-label';
        label.innerHTML = `
            <span>${clase}</span>
            <span>${(prob * 100).toFixed(1)}%</span>
        `;
        
        const fill = document.createElement('div');
        fill.className = 'probability-bar-fill';
        fill.style.width = `${prob * 100}%`;
        fill.style.background = `linear-gradient(90deg, ${getClassColor(clase)}80 0%, ${getClassColor(clase)} 100%)`;
        
        bar.appendChild(label);
        bar.appendChild(fill);
        probabilitiesChart.appendChild(bar);
    });
}

function getClassColor(clase) {
    const colors = {
        'Damaged': '#FF6B6B',
        'Old': '#FFA07A',
        'Ripe': '#51CF66',
        'Unripe': '#FFD93D'
    };
    return colors[clase] || '#9B6ED0';
}

// Verificar estado del backend al cargar
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/api/health`);
        const data = await response.json();
        
        if (!data.segmentacion && !data.clasificacion) {
            alert('⚠️ Advertencia: Los modelos no están disponibles. Verifica que el backend esté corriendo y los modelos estén cargados.');
        }
    } catch (error) {
        alert('⚠️ No se pudo conectar con el backend. Asegúrate de que esté corriendo en http://localhost:5000');
        console.error('Error connecting to backend:', error);
    }
});

