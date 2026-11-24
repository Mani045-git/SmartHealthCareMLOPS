const API_URL = "https://sm-7c0b7282b23f4f69be3df82920c18120.ecs.ap-south-1.on.aws/predict";

// -------------------------------------------------
// Predict Health Risk
// -------------------------------------------------

async function predictRisk() {

    const button = document.querySelector(
        'button[onclick="predictRisk()"]'
    );

    const container = document.getElementById("risk-container");

    const originalButtonHTML = button ? button.innerHTML : "";

    const numericFields = [
        "age",
        "height_cm",
        "weight_kg",
        "bmi",
        "heart_rate_bpm",
        "systolic_bp",
        "diastolic_bp",
        "spo2_percent",
        "temperature_f",
        "glucose_mg_dl",
        "cholesterol_mg_dl",
        "hemoglobin_g_dl",
        "creatinine_mg_dl",
        "sleep_hours"
    ];

    const categoricalFields = [
        "gender",
        "physical_activity",
        "smoking",
        "alcohol",
        "stress_level"
    ];

    const data = {};

    // Validate numerical fields
    for (const field of numericFields) {

        const input = document.getElementById(field);

        if (!input || input.value.trim() === "") {
            alert(`Please enter ${formatName(field)}.`);
            input?.focus();
            return;
        }

        const value = Number(input.value);

        if (!Number.isFinite(value)) {
            alert(`Please enter a valid number for ${formatName(field)}.`);
            input.focus();
            return;
        }

        data[field] = value;
    }

    // Validate categorical fields
    for (const field of categoricalFields) {

        const input = document.getElementById(field);

        if (!input || input.value.trim() === "") {
            alert(`Please select ${formatName(field)}.`);
            input?.focus();
            return;
        }

        data[field] = input.value;
    }

    try {

        if (button) {
            button.disabled = true;
            button.innerHTML = "Analyzing...";
        }

        if (container) {
            container.innerHTML = "<p class='placeholder'>Analyzing health risk...</p>";
        }

        // Send data to FastAPI
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {

            const errorText = await response.text();

            throw new Error(
                `API Error ${response.status}: ${errorText}`
            );
        }

        const result = await response.json();

        if (
            !result.predictions ||
            Object.keys(result.predictions).length !== 9
        ) {
            throw new Error("Expected 9 predictions from API.");
        }

        // Save predictions and patient information
        localStorage.setItem(
            "healthPredictions",
            JSON.stringify(result.predictions)
        );

        localStorage.setItem(
            "healthPatientData",
            JSON.stringify(data)
        );

        // Open Results Page
        window.location.href = "results.html";

    } catch (error) {

        console.error("Prediction Error:", error);

        if (container) {
            container.innerHTML = `
                <p class="error-message">
                    Unable to get predictions. Please check the API server.
                </p>
            `;
        }

        alert(
            "Unable to connect to Smart Healthcare API.\n\n" +
            error.message
        );

        if (button) {
            button.innerHTML = originalButtonHTML;
        }

    } finally {

        if (button) {
            button.disabled = false;
        }
    }
}


// -------------------------------------------------
// Format Names
// -------------------------------------------------

function formatName(name) {

    return name
        .replaceAll("_", " ")
        .replace(/\b\w/g, char => char.toUpperCase());
}