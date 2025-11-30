const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const undoBtn = document.getElementById('undo-btn');
const latexCode = document.getElementById('latex-code');
const katexOutput = document.getElementById('katex-output');
const copyBtn = document.getElementById('copy-btn');

// Resize canvas to fill container
function resizeCanvas() {
  // Save current content
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(canvas, 0, 0);

  // Resize
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  // Fill white background
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Restore content (scaled if needed, but for now just top-left)
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';
}

window.addEventListener('resize', resizeCanvas);
// Call once after load
setTimeout(resizeCanvas, 100);

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let timeoutId = null;

// History for undo
let history = [];
const MAX_HISTORY = 10;

function saveState() {
  if (history.length >= MAX_HISTORY) history.shift();
  history.push(canvas.toDataURL());
}

function draw(e) {
  if (!isDrawing) return;

  // Get correct coordinates
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();

  lastX = x;
  lastY = y;

  // Debounce prediction
  clearTimeout(timeoutId);
  timeoutId = setTimeout(predict, 1000); // 1 second delay
}

canvas.addEventListener('mousedown', (e) => {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  lastX = e.clientX - rect.left;
  lastY = e.clientY - rect.top;
  saveState();
});

canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

// Touch support
canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  lastX = touch.clientX - rect.left;
  lastY = touch.clientY - rect.top;
  saveState();
});

canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();

  lastX = x;
  lastY = y;

  clearTimeout(timeoutId);
  timeoutId = setTimeout(predict, 1000);
});

canvas.addEventListener('touchend', () => isDrawing = false);

clearBtn.addEventListener('click', () => {
  saveState();
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  latexCode.value = '';
  katexOutput.innerHTML = '';
});

undoBtn.addEventListener('click', () => {
  if (history.length > 0) {
    const imgData = history.pop();
    const img = new Image();
    img.src = imgData;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      // Trigger predict? Maybe not.
    };
  }
});

copyBtn.addEventListener('click', () => {
  latexCode.select();
  navigator.clipboard.writeText(latexCode.value);
});

async function predict() {
  // Convert canvas to blob
  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('file', blob, 'drawing.png');

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        const latex = data.latex;
        latexCode.value = latex;
        // MathJax rendering
        katexOutput.innerHTML = `\\[${latex}\\]`;
        if (window.MathJax) {
          MathJax.typesetPromise([katexOutput]).catch((err) => console.log(err));
        }
      } else {
        console.error('Server error');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  });
}
