document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const scanBtn = document.getElementById('scan-btn');
    const btnLoader = document.getElementById('btn-loader');
    const btnText = document.querySelector('.btn-text');
    
    const resultSection = document.getElementById('result-section');
    const resultTitle = document.getElementById('result-title');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');
    const warningText = document.getElementById('warning-text');

    let currentFile = null;

    // Drag and Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                showPreview(file);
            } else {
                alert("Please upload an image file.");
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            dropZoneContent.style.display = 'none';
            previewContainer.style.display = 'flex';
            scanBtn.disabled = false;
            // Hide previous results if any
            resultSection.style.display = 'none';
        }
    }

    resetBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        dropZoneContent.style.display = 'block';
        previewContainer.style.display = 'none';
        scanBtn.disabled = true;
        resultSection.style.display = 'none';
    });

    scanBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading state
        scanBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnLoader.style.display = 'inline-block';
        resultSection.style.display = 'none';

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                showResult(data);
            } else {
                alert("Error: " + (data.error || "Something went wrong"));
                resetBtnState();
            }
        } catch (err) {
            console.error(err);
            alert("Error connecting to the server.");
            resetBtnState();
        }
    });

    function showResult(data) {
        resetBtnState();
        
        resultSection.style.display = 'block';
        
        // Reset animation
        confidenceFill.style.width = '0%';
        confidenceFill.className = 'confidence-fill';
        resultTitle.className = '';

        const isReal = data.prediction === "Real Image";
        
        setTimeout(() => {
            resultTitle.textContent = data.prediction;
            resultTitle.classList.add(isReal ? 'is-real' : 'is-fake');
            
            confidenceFill.style.width = `${data.confidence}%`;
            confidenceFill.classList.add(isReal ? 'fill-real' : 'fill-fake');
            
            // Animate number
            animateValue(confidenceValue, 0, data.confidence, 1000);
            
            if (data.warning) {
                warningText.textContent = `Note: ${data.warning}`;
                warningText.style.display = 'block';
            } else {
                warningText.style.display = 'none';
            }
        }, 100);
    }

    function resetBtnState() {
        scanBtn.disabled = false;
        btnText.textContent = 'Analyze Image';
        btnLoader.style.display = 'none';
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = (progress * (end - start) + start).toFixed(2);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
