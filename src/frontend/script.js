const API_URL =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
        ? "http://127.0.0.1:9000"
        : "/api";

const LAMBDA_API_URL =
    "https://q16uqkqopf.execute-api.us-east-1.amazonaws.com/smartgrid-prediction-trigger";

const FEATURE_NAMES = [
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "p1",
    "p2",
    "p3",
    "p4",
    "g1",
    "g2",
    "g3",
    "g4"
];


// ============================================================
// GET INPUT VALUES
// ============================================================

function getCurrentFeatures() {

    const features = {};

    FEATURE_NAMES.forEach(name => {

        const element =
            document.getElementById(name);

        if (element) {
            features[name] =
                Number(element.value);
        }
    });

    return features;
}


// ============================================================
// API REQUEST
// ============================================================

async function predictFeatures(features) {

    const response = await fetch(
        `${API_URL}/predict`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(features)
        }
    );

    if (!response.ok) {

        const errorText =
            await response.text();

        throw new Error(
            errorText ||
            `HTTP error ${response.status}`
        );
    }

    return await response.json();
}


// ============================================================
// ============================================================
// SEND CURRENT VALUES TO AWS LAMBDA
// ============================================================

async function sendToAWSLambda(features) {

    try {

        const response = await fetch(
            LAMBDA_API_URL,
            {
                method: "POST",

                // text/plain avoids unnecessary CORS preflight
                headers: {
                    "Content-Type": "text/plain"
                },

                body: JSON.stringify(features)
            }
        );

        const result = await response.json();

        console.log("AWS Lambda result:", result);

        return result;

    } catch (error) {

        console.error(
            "AWS Lambda error:",
            error
        );

        return null;
    }
}


// MAIN PREDICTION
// ============================================================

async function runPrediction() {

    try {

        const features =
            getCurrentFeatures();

        const result =
            await predictFeatures(features);

        if (!result.success) {

            throw new Error(
                result.error ||
                "Prediction failed."
            );
        }

        displayPrediction(result);

        displayAdvancedResults(result);

        // Send the SAME current website values to AWS Lambda
        sendToAWSLambda(features);

    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        alert(
            "Prediction failed:\n" +
            error.message
        );
    }
}


// ============================================================
// DISPLAY MAIN MODEL RESULTS
// ============================================================

function displayPrediction(result) {

    const gnn =
        result.gnn || {};

    const rf =
        result.random_forest || {};


    // GNN prediction

    const gnnPrediction =
        document.getElementById(
            "gnn-prediction"
        );

    if (gnnPrediction) {

        gnnPrediction.textContent =
            gnn.prediction || "Unknown";
    }


    // GNN reliability

    const gnnReliability =
        document.getElementById(
            "gnn-reliability"
        );

    if (gnnReliability) {

        gnnReliability.textContent =
            `${gnn.reliability_score ?? 0}%`;
    }


    // GNN stable

    const gnnStable =
        document.getElementById(
            "gnn-stable"
        );

    if (gnnStable) {

        gnnStable.textContent =
            `${gnn.stable_probability ?? 0}%`;
    }


    // GNN unstable

    const gnnUnstable =
        document.getElementById(
            "gnn-unstable"
        );

    if (gnnUnstable) {

        gnnUnstable.textContent =
            `${gnn.unstable_probability ?? 0}%`;
    }


    // RF prediction

    const rfPrediction =
        document.getElementById(
            "rf-prediction"
        );

    if (rfPrediction) {

        rfPrediction.textContent =
            rf.prediction || "Unknown";
    }


    // RF stable

    const rfStable =
        document.getElementById(
            "rf-stable"
        );

    if (rfStable) {

        rfStable.textContent =
            `${rf.stable_probability ?? 0}%`;
    }


    // RF unstable

    const rfUnstable =
        document.getElementById(
            "rf-unstable"
        );

    if (rfUnstable) {

        rfUnstable.textContent =
            `${rf.unstable_probability ?? 0}%`;
    }


    // Status labels

    const gnnStatus =
        document.getElementById(
            "gnn-status"
        );

    if (gnnStatus) {

        gnnStatus.textContent =
            "Prediction Complete";
    }


    const rfStatus =
        document.getElementById(
            "rf-status"
        );

    if (rfStatus) {

        rfStatus.textContent =
            "Prediction Complete";
    }
}


// ============================================================
// ADVANCED RESULTS
// ============================================================

function displayAdvancedResults(result) {

    const gnn =
        result.gnn || {};

    const rf =
        result.random_forest || {};


    // Risk

    const riskElement =
        document.getElementById(
            "risk-level"
        );

    if (riskElement) {

        riskElement.textContent =
            gnn.risk_level || "UNKNOWN";
    }


    // Model agreement

    const agreementElement =
        document.getElementById(
            "model-agreement"
        );

    if (agreementElement) {

        agreementElement.textContent =
            result.model_agreement ||
            "UNKNOWN";
    }


    // Maintenance

    const maintenanceElement =
        document.getElementById(
            "maintenance-recommendation"
        );

    if (maintenanceElement) {

        maintenanceElement.textContent =
            gnn.maintenance_recommendation ||
            "Continue monitoring.";
    }


    // Feature importance

    displayFeatureImportance(
        rf.feature_importance || []
    );
}


// ============================================================
// FEATURE IMPORTANCE
// ============================================================

function displayFeatureImportance(features) {

    const container =
        document.getElementById(
            "feature-importance"
        );

    if (!container) {
        return;
    }


    if (
        !features ||
        features.length === 0
    ) {

        container.innerHTML =
            "<p>Feature influence unavailable.</p>";

        return;
    }


    container.innerHTML = "";


    features
        .slice(0, 6)
        .forEach(item => {

            const row =
                document.createElement("div");

            row.className =
                "importance-item";


            const percentage =
                (
                    Number(item.importance) * 100
                ).toFixed(1);


            row.innerHTML = `

                <div class="importance-header">

                    <strong>
                        ${item.feature}
                    </strong>

                    <span>
                        ${percentage}%
                    </span>

                </div>

                <div class="importance-bar">

                    <div
                        class="importance-fill"
                        style="
                            width:${Math.min(
                                Number(percentage) * 2,
                                100
                            )}%;
                        "
                    ></div>

                </div>
            `;


            container.appendChild(row);
        });
}


// ============================================================
// FAULT SCENARIO SIMULATOR
// ============================================================

async function runScenario(scenario) {

    try {

        const features =
            getCurrentFeatures();


        // Normal

        if (scenario === "normal") {
            // Keep current values.
        }


        // High load

        else if (
            scenario === "high-load"
        ) {

            features.p1 *= 1.5;
            features.p2 *= 1.5;
            features.p3 *= 1.5;
            features.p4 *= 1.5;
        }


        // Generator stress

        else if (
            scenario === "generator-stress"
        ) {

            features.g1 *= 1.4;
            features.g2 *= 1.4;
            features.g3 *= 1.4;
            features.g4 *= 1.4;
        }


        // Transmission stress

        else if (
            scenario === "transmission-stress"
        ) {

            features.tau1 *= 1.3;
            features.tau2 *= 1.3;
            features.tau3 *= 1.3;
            features.tau4 *= 1.3;
        }


        // Combined fault

        else if (
            scenario === "combined-fault"
        ) {

            features.p1 *= 1.5;
            features.p2 *= 1.5;

            features.g1 *= 1.4;
            features.g2 *= 1.4;

            features.tau3 *= 1.3;
            features.tau4 *= 1.3;
        }


        const result =
            await predictFeatures(features);


        const gnn =
            result.gnn || {};


        const container =
            document.getElementById(
                "scenario-result"
            );


        if (!container) {
            return;
        }


        container.innerHTML = `

            <div class="scenario-result-card">

                <h3>
                    ${scenario
                        .replaceAll("-", " ")
                        .toUpperCase()}
                </h3>

                <p>
                    Prediction:
                    <strong>
                        ${gnn.prediction}
                    </strong>
                </p>

                <p>
                    Reliability:
                    <strong>
                        ${gnn.reliability_score}%
                    </strong>
                </p>

                <p>
                    Risk:
                    <strong>
                        ${gnn.risk_level}
                    </strong>
                </p>

                <p>
                    Stable Probability:
                    ${gnn.stable_probability}%
                </p>

                <p>
                    Unstable Probability:
                    ${gnn.unstable_probability}%
                </p>

            </div>
        `;

    }

    catch (error) {

        console.error(
            "Scenario error:",
            error
        );

        alert(
            "Scenario simulation failed:\n" +
            error.message
        );
    }
}


// ============================================================
// WHAT-IF ANALYSIS
// ============================================================

async function runWhatIf() {

    try {

        const parameterElement =
            document.getElementById("whatif-feature");

        const valueElement =
            document.getElementById("whatif-value");

        if (!parameterElement || !valueElement) {

            alert("What-If controls were not found.");

            return;
        }


        const parameter =
            parameterElement.value;

        const newValue =
            Number(valueElement.value);


        if (!Number.isFinite(newValue)) {

            alert(
                "Please enter a valid numeric value."
            );

            return;
        }


        // ----------------------------------------------------
        // GET ORIGINAL VALUES
        // ----------------------------------------------------

        const originalFeatures =
            getCurrentFeatures();


        const originalValue =
            originalFeatures[parameter];


        // ----------------------------------------------------
        // PREDICTION BEFORE CHANGE
        // ----------------------------------------------------

        const before =
            await predictFeatures(
                originalFeatures
            );


        // ----------------------------------------------------
        // CREATE MODIFIED COPY
        // ----------------------------------------------------

        const modifiedFeatures =
            {
                ...originalFeatures
            };


        modifiedFeatures[parameter] =
            newValue;


        // ----------------------------------------------------
        // PREDICTION AFTER CHANGE
        // ----------------------------------------------------

        const after =
            await predictFeatures(
                modifiedFeatures
            );


        // ----------------------------------------------------
        // GNN VALUES
        // ----------------------------------------------------

        const beforeGnn =
            before.gnn || {};

        const afterGnn =
            after.gnn || {};


        // ----------------------------------------------------
        // RANDOM FOREST VALUES
        // ----------------------------------------------------

        const beforeRf =
            before.random_forest || {};

        const afterRf =
            after.random_forest || {};


        const beforeGnnReliability =
            Number(
                beforeGnn.reliability_score || 0
            );


        const afterGnnReliability =
            Number(
                afterGnn.reliability_score || 0
            );


        const beforeRfReliability =
            Number(
                beforeRf.stable_probability || 0
            );


        const afterRfReliability =
            Number(
                afterRf.stable_probability || 0
            );


        const gnnChange =
            afterGnnReliability -
            beforeGnnReliability;


        const rfChange =
            afterRfReliability -
            beforeRfReliability;


        // ----------------------------------------------------
        // UPDATE ACTUAL INPUT BOX
        // ----------------------------------------------------

        const inputElement =
            document.getElementById(
                parameter
            );


        if (inputElement) {

            inputElement.value =
                newValue;
        }


        // ----------------------------------------------------
        // DISPLAY RESULT
        // ----------------------------------------------------

        const container =
            document.getElementById(
                "whatif-result"
            );


        if (!container) {
            return;
        }


        const gnnChangeText =
            gnnChange > 0
                ? `+${gnnChange.toFixed(2)}%`
                : `${gnnChange.toFixed(2)}%`;


        const rfChangeText =
            rfChange > 0
                ? `+${rfChange.toFixed(2)}%`
                : `${rfChange.toFixed(2)}%`;


        container.innerHTML = `

            <div class="whatif-result-card">

                <h3>
                    What-If Analysis Result
                </h3>


                <p>

                    Parameter:

                    <strong>
                        ${parameter}
                    </strong>

                </p>


                <p>

                    Original Value:

                    <strong>
                        ${originalValue}
                    </strong>

                </p>


                <p>

                    Modified Value:

                    <strong>
                        ${newValue}
                    </strong>

                </p>


                <hr>


                <h4>
                    GNN Analysis
                </h4>


                <p>

                    Prediction:

                    <strong>
                        ${beforeGnn.prediction}
                    </strong>

                    →

                    <strong>
                        ${afterGnn.prediction}
                    </strong>

                </p>


                <p>

                    Reliability:

                    <strong>
                        ${beforeGnnReliability.toFixed(2)}%
                    </strong>

                    →

                    <strong>
                        ${afterGnnReliability.toFixed(2)}%
                    </strong>

                </p>


                <p>

                    Reliability Change:

                    <strong>
                        ${gnnChangeText}
                    </strong>

                </p>


                <p>

                    Risk:

                    <strong>
                        ${beforeGnn.risk_level}
                    </strong>

                    →

                    <strong>
                        ${afterGnn.risk_level}
                    </strong>

                </p>


                <p>

                    Stable Probability:

                    ${beforeGnn.stable_probability}%

                    →

                    ${afterGnn.stable_probability}%

                </p>


                <p>

                    Unstable Probability:

                    ${beforeGnn.unstable_probability}%

                    →

                    ${afterGnn.unstable_probability}%

                </p>


                <hr>


                <h4>
                    Random Forest Analysis
                </h4>


                <p>

                    Prediction:

                    <strong>
                        ${beforeRf.prediction}
                    </strong>

                    →

                    <strong>
                        ${afterRf.prediction}
                    </strong>

                </p>


                <p>

                    Stable Probability:

                    <strong>
                        ${beforeRfReliability.toFixed(2)}%
                    </strong>

                    →

                    <strong>
                        ${afterRfReliability.toFixed(2)}%
                    </strong>

                </p>


                <p>

                    Probability Change:

                    <strong>
                        ${rfChangeText}
                    </strong>

                </p>


                <p>

                    Unstable Probability:

                    ${beforeRf.unstable_probability}%

                    →

                    ${afterRf.unstable_probability}%

                </p>


                <hr>


                <p>

                    <strong>
                        Parameter ${parameter}
                    </strong>

                    changed from

                    <strong>
                        ${originalValue}
                    </strong>

                    to

                    <strong>
                        ${newValue}
                    </strong>.

                </p>


                <p>

                    The modified value has been applied
                    to the input field above.

                </p>

            </div>

        `;


    }

    catch (error) {

        console.error(
            "What-If error:",
            error
        );


        alert(
            "What-If analysis failed:\n" +
            error.message
        );
    }
}


// ============================================================
// REAL-TIME SIMULATION

// ============================================================

let liveInterval = null;

let liveRunning = false;


function toggleLiveSimulation() {

    if (liveRunning) {

        clearInterval(
            liveInterval
        );

        liveInterval = null;

        liveRunning = false;


        const button =
            document.getElementById(
                "live-button"
            );


        if (button) {

            button.textContent =
                "Start Live Monitoring";
        }


        const indicator =
            document.getElementById(
                "live-indicator"
            );


        if (indicator) {

            indicator.textContent =
                "● OFFLINE";
        }


        return;
    }


    liveRunning = true;


    const button =
        document.getElementById(
            "live-button"
        );


    if (button) {

        button.textContent =
            "Stop Live Monitoring";
    }


    const indicator =
        document.getElementById(
            "live-indicator"
        );


    if (indicator) {

        indicator.textContent =
            "● LIVE";
    }


    runLivePrediction();


    liveInterval =
        setInterval(
            runLivePrediction,
            5000
        );
}


// ============================================================
// LIVE PREDICTION
// ============================================================

async function runLivePrediction() {

    try {

        const features =
            getCurrentFeatures();


        FEATURE_NAMES.forEach(name => {

            const current =
                Number(
                    features[name]
                );


            const variation =
                (
                    Math.random() - 0.5
                ) * 0.10;


            features[name] =
                current +
                current * variation;
        });


        const result =
            await predictFeatures(
                features
            );


        const gnn =
            result.gnn || {};


        // Display values

        const valuesContainer =
            document.getElementById(
                "live-grid-values"
            );


        if (valuesContainer) {

            valuesContainer.innerHTML = `

                <div class="live-values-grid">

                    <div>
                        tau1:
                        ${features.tau1.toFixed(3)}
                    </div>

                    <div>
                        tau2:
                        ${features.tau2.toFixed(3)}
                    </div>

                    <div>
                        tau3:
                        ${features.tau3.toFixed(3)}
                    </div>

                    <div>
                        tau4:
                        ${features.tau4.toFixed(3)}
                    </div>

                    <div>
                        p1:
                        ${features.p1.toFixed(3)}
                    </div>

                    <div>
                        p2:
                        ${features.p2.toFixed(3)}
                    </div>

                    <div>
                        p3:
                        ${features.p3.toFixed(3)}
                    </div>

                    <div>
                        p4:
                        ${features.p4.toFixed(3)}
                    </div>

                    <div>
                        g1:
                        ${features.g1.toFixed(3)}
                    </div>

                    <div>
                        g2:
                        ${features.g2.toFixed(3)}
                    </div>

                    <div>
                        g3:
                        ${features.g3.toFixed(3)}
                    </div>

                    <div>
                        g4:
                        ${features.g4.toFixed(3)}
                    </div>

                </div>
            `;
        }


        // Display prediction

        const predictionContainer =
            document.getElementById(
                "live-prediction"
            );


        if (predictionContainer) {

            predictionContainer.innerHTML = `

                <div class="live-prediction-card">

                    <h3>
                        ${gnn.prediction}
                    </h3>

                    <p>
                        Reliability:
                        <strong>
                            ${gnn.reliability_score}%
                        </strong>
                    </p>

                    <p>
                        Risk:
                        <strong>
                            ${gnn.risk_level}
                        </strong>
                    </p>

                    <p>
                        Stable:
                        ${gnn.stable_probability}%
                    </p>

                    <p>
                        Unstable:
                        ${gnn.unstable_probability}%
                    </p>

                </div>
            `;
        }

    }

    catch (error) {

        console.error(
            "Live simulation error:",
            error
        );
    }
}


// ============================================================
// PAGE LOAD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        console.log(
            "Smart Grid AI frontend loaded."
        );

        console.log(
            "API:",
            API_URL
        );
    }
);
