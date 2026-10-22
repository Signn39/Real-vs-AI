const form = document.getElementById("upload-form");
const fileInput = document.getElementById("image-input");
const uploadSection = document.querySelector(".upload-section");
const placeholder = document.getElementById("results-placeholder");
const aiLikelihood = document.getElementById("ai-likelihood");
const realLikelihood = document.getElementById("real-likelihood");

function formatPercentage(value) {
    return `${Number(value).toFixed(2)}%`;
}

function showError(message) {
    placeholder.querySelector("h3").textContent = "Analysis failed";
    placeholder.querySelector("p").textContent = message;
    placeholder.hidden = false;
}

function showResult(result) {
    placeholder.querySelector("h3").textContent = result.result;
    placeholder.querySelector("p").textContent =
        `Confidence: ${formatPercentage(result.confidence)}`;
    placeholder.hidden = false;
    aiLikelihood.textContent = formatPercentage(result.probabilities.ai_generated);
    realLikelihood.textContent = formatPercentage(result.probabilities.real);
}

async function analyzeImage(file) {
    const formData = new FormData();
    formData.append("image", file);
    uploadSection.classList.add("is-loading");

    try {
        const response = await fetch(form.action, {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error?.message || "The image could not be analyzed.");
        }
        showResult(payload);
    } catch (error) {
        showError(error.message);
    } finally {
        uploadSection.classList.remove("is-loading");
    }
}

fileInput.addEventListener("change", () => {
    const [file] = fileInput.files;
    if (file) {
        analyzeImage(file);
    }
});

form.addEventListener("submit", (event) => event.preventDefault());

["dragenter", "dragover"].forEach((eventName) => {
    uploadSection.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadSection.classList.add("is-dragging");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    uploadSection.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadSection.classList.remove("is-dragging");
    });
});

uploadSection.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (file) {
        analyzeImage(file);
    }
});