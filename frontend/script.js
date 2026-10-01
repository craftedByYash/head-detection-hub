// State Management
let currentMode = "single"; // "single" | "compare"
let uploadedVideo = null;    // Video metadata from /api/upload
let activeJobId = null;
let progressPollTimer = null;
let syncPlaybackActive = false;

// DOM Elements
const systemDeviceBadge = document.getElementById("systemDeviceBadge");
const deviceDot = document.getElementById("deviceDot");
const deviceLabel = document.getElementById("deviceLabel");

const dropZone = document.getElementById("dropZone");
const videoFileInput = document.getElementById("videoFileInput");
const dropZoneContent = document.getElementById("dropZoneContent");
const uploadSpinner = document.getElementById("uploadSpinner");
const uploadedVideoInfo = document.getElementById("uploadedVideoInfo");

let selectedModelName = null;
const modelSelect = document.getElementById("modelSelect");
const modelSelectA = document.getElementById("modelSelectA");
const modelSelectB = document.getElementById("modelSelectB");
const singleModelSelector = document.getElementById("singleModelSelector");
const compareModelSelector = document.getElementById("compareModelSelector");
const modelOptionButtons = document.getElementById("modelOptionButtons");
const modelCountLabel = document.getElementById("modelCountLabel");
const modelUploadInput = document.getElementById("modelUploadInput");


const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const iouSlider = document.getElementById("iouSlider");
const iouValue = document.getElementById("iouValue");
const deviceSelect = document.getElementById("deviceSelect");
const frameLimitSelect = document.getElementById("frameLimitSelect");


const runBtn = document.getElementById("runBtn");
const runBtnText = document.getElementById("runBtnText");
const runBtnIcon = document.getElementById("runBtnIcon");

const progressCard = document.getElementById("progressCard");
const progressStatusHeader = document.getElementById("progressStatusHeader");
const progressStatusMessage = document.getElementById("progressStatusMessage");
const progressPercent = document.getElementById("progressPercent");
const progressBarFill = document.getElementById("progressBarFill");
const liveFrame = document.getElementById("liveFrame");
const liveElapsed = document.getElementById("liveElapsed");
const liveFps = document.getElementById("liveFps");

const singleResultCard = document.getElementById("singleResultCard");
const processedVideoPlayer = document.getElementById("processedVideoPlayer");
const downloadResultBtn = document.getElementById("downloadResultBtn");
const downloadResultBtnLarge = document.getElementById("downloadResultBtnLarge");


const compareResultCard = document.getElementById("compareResultCard");
const videoPlayerA = document.getElementById("videoPlayerA");
const videoPlayerB = document.getElementById("videoPlayerB");
const compareTitleA = document.getElementById("compareTitleA");
const compareTitleB = document.getElementById("compareTitleB");
const downloadBtnA = document.getElementById("downloadBtnA");
const downloadBtnB = document.getElementById("downloadBtnB");
const syncPlayBtn = document.getElementById("syncPlayBtn");

const comparisonTableBody = document.getElementById("comparisonTableBody");
const errorAlert = document.getElementById("errorAlert");
const errorTitle = document.getElementById("errorTitle");
const errorMessage = document.getElementById("errorMessage");
const errorTip = document.getElementById("errorTip");

// Initialize application on page load
document.addEventListener("DOMContentLoaded", () => {
  fetchSystemInfo();
  loadModels();
  initDragAndDrop();
  setupSyncVideoEvents();
  loadHistory();
});

// Mode Switching (Single vs Side-by-Side Comparison)
function switchMode(mode) {
  currentMode = mode;
  document.getElementById("tabSingle").classList.toggle("active", mode === "single");
  document.getElementById("tabCompare").classList.toggle("active", mode === "compare");

  if (mode === "single") {
    singleModelSelector.classList.remove("hidden");
    compareModelSelector.classList.add("hidden");
    document.getElementById("modelSectionTitle").textContent = "2. Select Model & Parameters";
    runBtnText.textContent = "Run Head Detection";
    runBtnIcon.textContent = "🚀";
  } else {
    singleModelSelector.classList.add("hidden");
    compareModelSelector.classList.remove("hidden");
    document.getElementById("modelSectionTitle").textContent = "2. Select Models for Comparison";
    runBtnText.textContent = "Compare Models Side-by-Side";
    runBtnIcon.textContent = "⚖️";
  }
}

// Fetch System Acceleration Details
async function fetchSystemInfo() {
  try {
    const res = await fetch("/api/system");
    const data = await res.json();
    if (data.cuda_available) {
      deviceDot.className = "status-indicator cuda";
      deviceLabel.textContent = `Device: CUDA GPU (${data.gpu_name})`;
    } else {
      deviceDot.className = "status-indicator cpu";
      deviceLabel.textContent = `Device: CPU (CUDA unavailable)`;
    }
  } catch (err) {
    deviceLabel.textContent = "Device: System Ready";
  }
}

// Load Available .pt Models from /api/models
async function loadModels(preferredSelection = null) {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const models = data.models || [];

    if (modelCountLabel) {
      modelCountLabel.textContent = `${models.length} model${models.length === 1 ? "" : "s"} found`;
    }

    modelSelect.innerHTML = "";
    modelSelectA.innerHTML = "";
    modelSelectB.innerHTML = "";
    if (modelOptionButtons) {
      modelOptionButtons.innerHTML = "";
    }

    if (models.length === 0) {
      const emptyOpt = document.createElement("option");
      emptyOpt.value = "";
      emptyOpt.disabled = true;
      emptyOpt.selected = true;
      emptyOpt.textContent = "No .pt models found in models/ folder";
      modelSelect.appendChild(emptyOpt);
      modelSelectA.appendChild(emptyOpt.cloneNode(true));
      modelSelectB.appendChild(emptyOpt.cloneNode(true));

      if (modelOptionButtons) {
        modelOptionButtons.innerHTML = `
          <div class="text-muted text-center py-3" style="font-size:0.85rem;">
            No <code>.pt</code> models found in <code>models/</code>.<br>
            Click <strong>📁 + Add .pt Model</strong> above or place models into the <code>models/</code> folder.
          </div>
        `;
      }
      showErrorAlert(
        "No Models Detected",
        "No .pt model weight files were found in the 'models/' directory.",
        "Please click '📁 + Add .pt Model' or place your trained head-detection .pt files inside the models/ folder."
      );
      return;
    }

    // Determine active model to select
    let targetSelection = preferredSelection;
    if (!targetSelection || !models.some(m => m.filename === targetSelection)) {
      targetSelection = selectedModelName && models.some(m => m.filename === selectedModelName)
        ? selectedModelName
        : models[0].filename;
    }
    selectedModelName = targetSelection;

    // Build Option Buttons & Dropdown Options
    models.forEach((m, idx) => {
      // 1. Dropdown options
      const opt = document.createElement("option");
      opt.value = m.filename;
      opt.textContent = `${m.filename} (${m.size_mb} MB)`;
      modelSelect.appendChild(opt);

      const optA = opt.cloneNode(true);
      const optB = opt.cloneNode(true);
      modelSelectA.appendChild(optA);
      modelSelectB.appendChild(optB);

      if (idx === 0) {
        modelSelectA.value = m.filename;
      }
      if (idx === 1 || (idx === 0 && models.length === 1)) {
        modelSelectB.value = m.filename;
      }

      // 2. Interactive Option Button Tile
      if (modelOptionButtons) {
        const isSelected = m.filename === selectedModelName;
        const card = document.createElement("div");
        card.className = "model-option-card" + (isSelected ? " selected" : "");
        card.dataset.modelName = m.filename;
        card.setAttribute("role", "radio");
        card.setAttribute("aria-checked", isSelected ? "true" : "false");
        card.onclick = () => selectModel(m.filename);

        card.innerHTML = `
          <div class="model-opt-left">
            <span class="model-radio-circle"></span>
            <span class="model-opt-name">${m.filename}</span>
          </div>
          <span class="model-opt-size">${m.size_mb} MB</span>
        `;
        modelOptionButtons.appendChild(card);
      }
    });

    // Synchronize select element
    modelSelect.value = selectedModelName;
    hideErrorAlert();
  } catch (err) {
    showErrorAlert("Model Discovery Error", "Failed to retrieve models from backend: " + err.message);
  }
}

// Select a model and update both option buttons and dropdown
function selectModel(modelFilename) {
  selectedModelName = modelFilename;

  if (modelSelect) {
    modelSelect.value = modelFilename;
  }

  if (modelOptionButtons) {
    const cards = modelOptionButtons.querySelectorAll(".model-option-card");
    cards.forEach(card => {
      const match = card.dataset.modelName === modelFilename;
      card.classList.toggle("selected", match);
      card.setAttribute("aria-checked", match ? "true" : "false");
    });
  }
}

function onModelDropdownChange(value) {
  selectModel(value);
}

// Upload a .pt model from disk directly into models/ folder
async function handleModelUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".pt")) {
    showErrorAlert("Invalid File", "Only .pt model weight files are supported.");
    event.target.value = "";
    return;
  }

  const formData = new FormData();
  formData.append("model_file", file);

  try {
    const res = await fetch("/api/models/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to upload model.");
    }

    const data = await res.json();
    event.target.value = "";
    // Reload models and automatically select the newly uploaded model
    await loadModels(data.filename);
  } catch (err) {
    showErrorAlert("Model Upload Error", err.message);
    event.target.value = "";
  }
}


// Drag & Drop Video Upload Handlers
function initDragAndDrop() {
  ["dragenter", "dragover"].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  videoFileInput.addEventListener("change", e => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });
}

// Upload Selected Video File
async function handleFileSelected(file) {
  const allowed = [".mp4", ".avi", ".mov", ".mkv"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showErrorAlert(
      "Invalid File Type",
      `The file '${file.name}' is not a supported video format.`,
      `Please upload a video ending in: ${allowed.join(", ")}`
    );
    return;
  }

  hideErrorAlert();
  dropZoneContent.classList.add("hidden");
  uploadSpinner.classList.remove("hidden");

  const formData = new FormData();
  formData.append("video", file);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Upload failed");
    }

    uploadedVideo = await res.json();

    // Update UI with video metadata
    document.getElementById("metaVideoName").textContent = uploadedVideo.original_name;
    document.getElementById("metaResolution").textContent = uploadedVideo.resolution;
    document.getElementById("metaFps").textContent = uploadedVideo.fps;
    document.getElementById("metaDuration").textContent = uploadedVideo.duration;
    document.getElementById("metaFrames").textContent = uploadedVideo.total_frames;

    dropZone.classList.add("hidden");
    uploadedVideoInfo.classList.remove("hidden");

    // Load comparison table history for this video
    loadHistory(uploadedVideo.filename);

  } catch (err) {
    showErrorAlert("Upload Error", err.message, "Ensure the video is uncorrupted and readable by OpenCV.");
    resetVideoUpload();
  } finally {
    uploadSpinner.classList.add("hidden");
    dropZoneContent.classList.remove("hidden");
  }
}

function resetVideoUpload() {
  uploadedVideo = null;
  videoFileInput.value = "";
  uploadedVideoInfo.classList.add("hidden");
  dropZone.classList.remove("hidden");
}

// Parameter Slider Display Updates
function updateConfDisplay(val) {
  confValue.textContent = parseFloat(val).toFixed(2);
}

function updateIouDisplay(val) {
  iouValue.textContent = parseFloat(val).toFixed(2);
}

// Run Head Detection (Single or Side-by-Side)
async function handleRunDetection() {
  if (!uploadedVideo) {
    showErrorAlert("No Video Uploaded", "Please upload a drone video before running detection.");
    return;
  }

  hideErrorAlert();
  const conf = parseFloat(confSlider.value);
  const iou = parseFloat(iouSlider.value);
  const device = deviceSelect.value;

  if (currentMode === "single") {
    const model = selectedModelName || modelSelect.value;
    if (!model) {
      showErrorAlert("No Model Selected", "Please select a model from the option buttons or dropdown.");
      return;
    }
    await startSingleDetection(uploadedVideo.filename, model, conf, iou, device);
  } else {
    const modelA = modelSelectA.value;
    const modelB = modelSelectB.value;
    if (!modelA || !modelB) {
      showErrorAlert("Model Selection Incomplete", "Please select both Model A and Model B.");
      return;
    }
    await startComparisonDetection(uploadedVideo.filename, modelA, modelB, conf, iou, device);
  }
}

// Initiate Single Model Detection Job
async function startSingleDetection(videoFilename, modelName, conf, iou, device) {
  setProcessingUI(true, `Running head detection with ${modelName}...`);

  try {
    const maxFrames = frameLimitSelect ? parseInt(frameLimitSelect.value) || 0 : 0;
    const formData = new FormData();
    formData.append("video", videoFilename);
    formData.append("model", modelName);
    formData.append("confidence", conf);
    formData.append("iou", iou);
    formData.append("device", device);
    formData.append("max_frames", maxFrames);

    const res = await fetch("/api/detect", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to start detection job.");
    }

    const data = await res.json();
    activeJobId = data.job_id;
    startProgressPolling(activeJobId);

  } catch (err) {
    setProcessingUI(false);
    showErrorAlert("Job Dispatch Error", err.message);
  }
}

// Initiate Side-by-Side Comparison Job
async function startComparisonDetection(videoFilename, modelA, modelB, conf, iou, device) {
  setProcessingUI(true, `Comparing ${modelA} vs ${modelB}...`);

  try {
    const maxFrames = frameLimitSelect ? parseInt(frameLimitSelect.value) || 0 : 0;
    const formData = new FormData();
    formData.append("video", videoFilename);
    formData.append("model_a", modelA);
    formData.append("model_b", modelB);
    formData.append("confidence", conf);
    formData.append("iou", iou);
    formData.append("device", device);
    formData.append("max_frames", maxFrames);


    const res = await fetch("/api/compare", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to start comparison job.");
    }

    const data = await res.json();
    activeJobId = data.job_id;
    startProgressPolling(activeJobId);

  } catch (err) {
    setProcessingUI(false);
    showErrorAlert("Job Dispatch Error", err.message);
  }
}

// Poll Progress Endpoint (/api/progress/{job_id})
function startProgressPolling(jobId) {
  if (progressPollTimer) clearInterval(progressPollTimer);

  progressPollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/progress/${jobId}`);
      if (!res.ok) {
        throw new Error("Unable to retrieve job progress.");
      }

      const job = await res.json();
      updateProgressDisplay(job);

      if (job.status === "completed") {
        clearInterval(progressPollTimer);
        setProcessingUI(false);
        handleJobCompleted(job);
      } else if (job.status === "failed") {
        clearInterval(progressPollTimer);
        setProcessingUI(false);
        showErrorAlert(
          "Detection Failed",
          job.error || "An unexpected error occurred during model inference.",
          "Please verify that the model is a valid Ultralytics YOLO model and fits within available memory."
        );
      } else if (job.status === "cancelled") {
        clearInterval(progressPollTimer);
        setProcessingUI(false);
        progressStatusHeader.textContent = "Detection Cancelled";
        progressStatusMessage.textContent = "Job was aborted by user.";
      }

    } catch (err) {
      clearInterval(progressPollTimer);
      setProcessingUI(false);
      showErrorAlert("Progress Error", err.message);
    }
  }, 450);
}

// Update Live Progress UI Elements
function updateProgressDisplay(job) {
  const pct = Math.min(100, Math.max(0, job.progress || 0));
  progressBarFill.style.width = `${pct}%`;
  progressPercent.textContent = `${pct}%`;

  if (job.message) {
    progressStatusMessage.textContent = job.message;
  }

  if (job.total_frames > 0) {
    liveFrame.textContent = `${job.current_frame} / ${job.total_frames}`;
  } else {
    liveFrame.textContent = `${job.current_frame} frames`;
  }

  liveElapsed.textContent = `${job.elapsed_seconds || 0}s`;
  liveFps.textContent = `${job.inference_fps || 0} FPS`;
}

// Cancel Current Job
async function cancelCurrentJob() {
  if (!activeJobId) return;
  try {
    await fetch(`/api/cancel/${activeJobId}`, { method: "POST" });
    progressStatusMessage.textContent = "Cancelling processing...";
  } catch (err) {
    console.error("Cancel failed:", err);
  }
}

// Handle Job Completion & Display Outputs
function handleJobCompleted(job) {
  if (job.mode === "single") {
    displaySingleResult(job.result);
  } else {
    displayComparisonResult(job.result);
  }

  // Refresh comparison table
  if (uploadedVideo) {
    loadHistory(uploadedVideo.filename);
  } else {
    loadHistory();
  }
}

// Render Single Model Evaluation Results
function displaySingleResult(stats) {
  singleResultCard.classList.remove("hidden");
  compareResultCard.classList.add("hidden");

  // Load video into HTML5 player with cache-busting timestamp
  const videoSrc = `${stats.output_url}?t=${Date.now()}`;
  processedVideoPlayer.src = videoSrc;
  processedVideoPlayer.load();

  downloadResultBtn.href = stats.download_url;
  downloadResultBtn.setAttribute("download", stats.output_filename);

  if (downloadResultBtnLarge) {
    downloadResultBtnLarge.href = stats.download_url;
    downloadResultBtnLarge.setAttribute("download", stats.output_filename);
  }

  // Populate all 13 stats in the statistics panel
  document.getElementById("statsModelBadge").textContent = stats.model;
  document.getElementById("statModel").textContent = stats.model;
  document.getElementById("statVideo").textContent = stats.video;
  document.getElementById("statResolution").textContent = stats.resolution;
  document.getElementById("statFps").textContent = stats.fps;
  document.getElementById("statDuration").textContent = stats.duration;
  document.getElementById("statTotalFrames").textContent = stats.total_frames;
  document.getElementById("statFramesProcessed").textContent = stats.frames_processed;
  document.getElementById("statTotalDetections").textContent = stats.total_head_detections;
  document.getElementById("statAvgHeads").textContent = stats.avg_heads_per_frame;
  document.getElementById("statMaxHeads").textContent = stats.max_heads_per_frame;
  document.getElementById("statAvgConfidence").textContent = stats.avg_confidence;
  document.getElementById("statProcessingTime").textContent = stats.processing_time;
  document.getElementById("statInferenceFps").textContent = `${stats.inference_fps} FPS`;

  // Autoplay video so user immediately sees head detections
  processedVideoPlayer.play().catch(() => {
    // If browser blocks unmuted autoplay, mute and play
    processedVideoPlayer.muted = true;
    processedVideoPlayer.play().catch(e => console.log("Autoplay notice:", e));
  });

  // Smoothly scroll to results
  singleResultCard.scrollIntoView({ behavior: "smooth" });
}

function setPlaybackSpeed(speed, btnElement) {
  if (processedVideoPlayer) {
    processedVideoPlayer.playbackRate = speed;
  }
  if (btnElement && btnElement.parentElement) {
    btnElement.parentElement.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
    btnElement.classList.add("active");
  }
}


// Render Side-by-Side Comparison Results
function displayComparisonResult(results) {
  compareResultCard.classList.remove("hidden");
  singleResultCard.classList.add("hidden");

  const a = results.model_a;
  const b = results.model_b;

  // Video A
  compareTitleA.textContent = a.model;
  videoPlayerA.src = `${a.output_url}?t=${Date.now()}`;
  videoPlayerA.load();
  downloadBtnA.href = a.download_url;
  downloadBtnA.setAttribute("download", a.output_filename);

  document.getElementById("compareDetectionsA").textContent = a.total_head_detections;
  document.getElementById("compareConfA").textContent = a.avg_confidence;
  document.getElementById("compareAvgHeadsA").textContent = a.avg_heads_per_frame;
  document.getElementById("compareMaxHeadsA").textContent = a.max_heads_per_frame;
  document.getElementById("compareFpsA").textContent = `${a.inference_fps} FPS`;

  // Video B
  compareTitleB.textContent = b.model;
  videoPlayerB.src = `${b.output_url}?t=${Date.now()}`;
  videoPlayerB.load();
  downloadBtnB.href = b.download_url;
  downloadBtnB.setAttribute("download", b.output_filename);

  document.getElementById("compareDetectionsB").textContent = b.total_head_detections;
  document.getElementById("compareConfB").textContent = b.avg_confidence;
  document.getElementById("compareAvgHeadsB").textContent = b.avg_heads_per_frame;
  document.getElementById("compareMaxHeadsB").textContent = b.max_heads_per_frame;
  document.getElementById("compareFpsB").textContent = `${b.inference_fps} FPS`;

  // Scroll to compare section
  compareResultCard.scrollIntoView({ behavior: "smooth" });
}

// Synchronized Video Playback for Comparison Mode
function setupSyncVideoEvents() {
  let isSeeking = false;

  videoPlayerA.addEventListener("seeked", () => {
    if (isSeeking) return;
    isSeeking = true;
    if (Math.abs(videoPlayerB.currentTime - videoPlayerA.currentTime) > 0.08) {
      videoPlayerB.currentTime = videoPlayerA.currentTime;
    }
    setTimeout(() => { isSeeking = false; }, 80);
  });

  videoPlayerB.addEventListener("seeked", () => {
    if (isSeeking) return;
    isSeeking = true;
    if (Math.abs(videoPlayerA.currentTime - videoPlayerB.currentTime) > 0.08) {
      videoPlayerA.currentTime = videoPlayerB.currentTime;
    }
    setTimeout(() => { isSeeking = false; }, 80);
  });
}

function toggleSyncPlayback() {
  if (videoPlayerA.paused || videoPlayerB.paused) {
    videoPlayerA.play();
    videoPlayerB.play();
    syncPlayBtn.textContent = "⏸ Pause Both";
  } else {
    videoPlayerA.pause();
    videoPlayerB.pause();
    syncPlayBtn.textContent = "▶ Play Both in Sync";
  }
}

// Fetch & Populate Evaluation History Table
async function loadHistory(videoFilter = null) {
  try {
    const url = videoFilter ? `/api/history?video=${encodeURIComponent(videoFilter)}` : "/api/history";
    const res = await fetch(url);
    const data = await res.json();
    const history = data.history || [];

    comparisonTableBody.innerHTML = "";

    if (history.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td colspan="8" class="text-center text-muted empty-state">
          No evaluation runs recorded yet. Upload a drone video and click <strong>Run Head Detection</strong> to compare models.
        </td>
      `;
      comparisonTableBody.appendChild(row);
      return;
    }

    history.forEach(item => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${item.model}</strong></td>
        <td>${item.avg_confidence}</td>
        <td>${item.avg_heads_per_frame}</td>
        <td>${item.max_heads_per_frame || "-"}</td>
        <td>${item.total_detections}</td>
        <td>${item.inference_fps} FPS</td>
        <td>${item.processing_time}</td>
        <td>
          <a href="${item.output_url}" target="_blank" class="btn btn-xs btn-outline">View</a>
          <a href="${item.download_url}" download class="btn btn-xs btn-outline">⬇</a>
        </td>
      `;
      comparisonTableBody.appendChild(tr);
    });

  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

async function clearEvaluationHistory() {
  if (!confirm("Are you sure you want to clear the comparison table history?")) return;
  try {
    await fetch("/api/history", { method: "DELETE" });
    loadHistory();
  } catch (err) {
    showErrorAlert("History Error", "Failed to clear history.");
  }
}

// UI State Toggles
function setProcessingUI(isProcessing, message = "") {
  runBtn.disabled = isProcessing;
  if (isProcessing) {
    progressCard.classList.remove("hidden");
    progressBarFill.style.width = "0%";
    progressPercent.textContent = "0%";
    progressStatusHeader.textContent = "Processing video...";
    progressStatusMessage.textContent = message || "Initializing YOLO model...";
    liveFrame.textContent = "-";
    liveElapsed.textContent = "0.0s";
    liveFps.textContent = "-";
    runBtnIcon.textContent = "⏳";
  } else {
    runBtnIcon.textContent = currentMode === "single" ? "🚀" : "⚖️";
    activeJobId = null;
  }
}

function showErrorAlert(title, message, tip = "") {
  errorTitle.textContent = title;
  errorMessage.textContent = message;
  errorTip.textContent = tip;
  errorAlert.classList.remove("hidden");
  errorAlert.scrollIntoView({ behavior: "smooth" });
}

function hideErrorAlert() {
  errorAlert.classList.add("hidden");
}
