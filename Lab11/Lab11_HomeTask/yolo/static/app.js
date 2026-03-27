const config = window.STUDIO_CONFIG;

const elements = {
  fileInput: document.getElementById("file-input"),
  uploadZone: document.getElementById("upload-zone"),
  task: document.getElementById("task"),
  runBtn: document.getElementById("run-btn"),
  inputPreviewShell: document.getElementById("input-preview-shell"),
  inputPreview: document.getElementById("input-preview"),
  inputFilename: document.getElementById("input-filename"),
  inputSize: document.getElementById("input-size"),
  errorMessage: document.getElementById("error-message"),
  statusChip: document.getElementById("status-chip"),
  resultsEmpty: document.getElementById("results-empty"),
  resultsContent: document.getElementById("results-content"),
  originalImage: document.getElementById("original-image"),
  annotatedImage: document.getElementById("annotated-image"),
  summaryGrid: document.getElementById("summary-grid"),
  predictionList: document.getElementById("prediction-list"),
  rawJson: document.getElementById("raw-json"),
  downloadImage: document.getElementById("download-image"),
  downloadJson: document.getElementById("download-json"),
  copyJson: document.getElementById("copy-json"),
};

let selectedFile = null;
let latestPayload = null;

const setError = (message = "") => {
  if (!message) {
    elements.errorMessage.classList.add("hidden");
    elements.errorMessage.textContent = "";
    return;
  }

  elements.errorMessage.textContent = message;
  elements.errorMessage.classList.remove("hidden");
};

const setStatus = (message) => {
  elements.statusChip.textContent = message;
};

const formatFileSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

const previewFile = (file) => {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (event) => {
    elements.inputPreview.src = event.target.result;
    elements.originalImage.src = event.target.result;
    elements.inputPreviewShell.classList.remove("hidden");
    elements.inputFilename.textContent = file.name;
    elements.inputSize.textContent = formatFileSize(file.size);
    elements.runBtn.disabled = false;
    setError();
    setStatus("Ready to run");
  };
  reader.readAsDataURL(file);
};

const renderSummary = (payload) => {
  const rows = [
    ["Task", payload.task],
    ["Headline", payload.summary.headline],
    ["Inference Time", `${payload.summary.inference_time_ms} ms`],
    ["Predictions", String(payload.summary.prediction_count)],
    ["Image Size", `${payload.input.width} x ${payload.input.height}`],
    ["File Type", payload.input.content_type],
  ];

  elements.summaryGrid.innerHTML = rows
    .map(
      ([label, value]) => `
        <article class="summary-item">
          <span class="prediction-meta">${label}</span>
          <strong>${value}</strong>
        </article>
      `,
    )
    .join("");
};

const renderPredictions = (payload) => {
  if (!payload.predictions.length) {
    elements.predictionList.innerHTML = `
      <article class="prediction-item">
        <h4>No predictions</h4>
        <div class="prediction-meta">The selected task did not return any predictions for this image.</div>
      </article>
    `;
    return;
  }

  elements.predictionList.innerHTML = payload.predictions
    .slice(0, 12)
    .map((prediction, index) => {
      const title = prediction.class_name || `Prediction ${index + 1}`;
      return `
        <article class="prediction-item">
          <h4>${title}</h4>
          <div class="prediction-meta">${JSON.stringify(prediction, null, 2)}</div>
        </article>
      `;
    })
    .join("");
};

const renderPayload = (payload) => {
  latestPayload = payload;
  elements.resultsEmpty.classList.add("hidden");
  elements.resultsContent.classList.remove("hidden");
  elements.annotatedImage.src = payload.artifacts.annotated_image_url;
  elements.downloadImage.href = payload.artifacts.annotated_image_url;
  elements.downloadJson.href = payload.artifacts.result_json_url;
  renderSummary(payload);
  renderPredictions(payload);
  elements.rawJson.textContent = JSON.stringify(payload, null, 2);
};

const handleRun = async () => {
  if (!selectedFile) {
    setError("Choose an image before running inference.");
    return;
  }

  const formData = new FormData();
  formData.append("task", elements.task.value);
  formData.append("file", selectedFile);

  elements.runBtn.disabled = true;
  setStatus("Running inference...");
  setError();

  try {
    const response = await fetch(config.inferUrl, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok || payload.status === "error") {
      throw new Error(payload.error || "Inference failed.");
    }

    renderPayload(payload);
    setStatus("Inference complete");
  } catch (error) {
    setError(error.message);
    setStatus("Request failed");
  } finally {
    elements.runBtn.disabled = false;
  }
};

["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
  elements.uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.uploadZone.addEventListener(eventName, () => elements.uploadZone.classList.add("dragover"));
});

["dragleave", "drop"].forEach((eventName) => {
  elements.uploadZone.addEventListener(eventName, () => elements.uploadZone.classList.remove("dragover"));
});

elements.uploadZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) previewFile(file);
});

elements.fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) previewFile(file);
});

elements.runBtn.addEventListener("click", handleRun);

elements.copyJson.addEventListener("click", async () => {
  if (!latestPayload) return;
  await navigator.clipboard.writeText(JSON.stringify(latestPayload, null, 2));
  setStatus("JSON copied to clipboard");
});
