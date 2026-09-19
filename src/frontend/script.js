const API_URL = "http://127.0.0.1:9000";


const FEATURES = [
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


async function makePrediction() {

    const button =
        document.querySelector(".predict-button");

    button.textContent = "⏳ Running AI Prediction...";

    button.disabled = true;


    try {

        const input = {};


        // ================================================
        // READ ALL 12 INPUTS
        // ================================================

        for (const feature of FEATURES) {

            const element =
                document.getElementById(feature);

            const value =
                parseFloat(element.value);


            if (Number.isNaN(value)) {

                throw new Error(
                    `Please enter a valid value for ${feature}`
                );

            }


            input[feature] = value;

        }


        // ================================================
        // SEND REQUEST TO FLASK
        // ================================================

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(input)
            }
        );


        if (!response.ok) {

            throw new Error(
                `API error: ${response.status}`
            );

        }


        const result =
            await response.json();


        // ================================================
        // DISPLAY RESULTS
        // ================================================

        updateGNN(result.gnn);

        updateRandomForest(
            result.random_forest
        );


        // Scroll to results

        document
            .getElementById("results")
            .scrollIntoView({
                behavior: "smooth"
            });


    } catch (error) {

        console.error(error);

        alert(
            "Prediction failed.\n\n" +
            error.message
        );

    } finally {

        button.textContent =
            "⚡ Predict Grid Reliability";

        button.disabled = false;

    }

}


/* ========================================================
   UPDATE GNN
======================================================== */

function updateGNN(result) {

    const status =
        document.getElementById("gnn-status");


    status.textContent =
        result.prediction;


    status.className =
        "prediction-badge " +
        result.prediction.toLowerCase();


    document.getElementById(
        "gnn-reliability"
    ).textContent =
        `${result.reliability_score}%`;


    document.getElementById(
        "gnn-stable"
    ).textContent =
        `${result.stable_probability}%`;


    document.getElementById(
        "gnn-unstable"
    ).textContent =
        `${result.unstable_probability}%`;


    document.getElementById(
        "gnn-stable-bar"
    ).style.width =
        `${result.stable_probability}%`;


    document.getElementById(
        "gnn-unstable-bar"
    ).style.width =
        `${result.unstable_probability}%`;

}


/* ========================================================
   UPDATE RANDOM FOREST
======================================================== */

function updateRandomForest(result) {

    const status =
        document.getElementById("rf-status");


    status.textContent =
        result.prediction;


    status.className =
        "prediction-badge " +
        result.prediction.toLowerCase();


    document.getElementById(
        "rf-stable"
    ).textContent =
        `${result.stable_probability}%`;


    document.getElementById(
        "rf-unstable"
    ).textContent =
        `${result.unstable_probability}%`;


    document.getElementById(
        "rf-stable-bar"
    ).style.width =
        `${result.stable_probability}%`;


    document.getElementById(
        "rf-unstable-bar"
    ).style.width =
        `${result.unstable_probability}%`;

}
