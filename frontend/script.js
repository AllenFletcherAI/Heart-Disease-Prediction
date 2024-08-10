document.getElementById('predict-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    const submitButton = event.target.querySelector('button');

    // Disable the button to prevent multiple submissions
    submitButton.disabled = true;

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        const resultDiv = document.getElementById('result');
        
        if (result.prediction === 1) {
            resultDiv.innerHTML = `<p class="result-text danger">Prediction: High risk of heart disease</p>`;
        } else {
            resultDiv.innerHTML = `<p class="result-text success">Prediction: Low risk of heart disease</p>`;
        }
    } catch (error) {
        document.getElementById('result').innerHTML = `<p class="result-text error">Error: ${error.message}</p>`;
    } finally {
        // Re-enable the button after the request is complete
        submitButton.disabled = false;
    }
});

