document.getElementById("image-upload").addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById("file-name").textContent = file.name;
        document.getElementById("colorize-btn").disabled = false;
    }
});

document.getElementById("upload-form").addEventListener("submit", function (e) {
    e.preventDefault();

    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"]');
    formData.append('image', fileField.files[0]);

    // UI Updates
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results-section").classList.add("hidden");
    document.getElementById("colorize-btn").disabled = true;

    fetch('/colorize', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("loading").classList.add("hidden");
            document.getElementById("colorize-btn").disabled = false;

            if (data.error) {
                alert("Error: " + data.error);
            } else {
                // Update images
                document.getElementById("original-image").src = data.original;
                document.getElementById("colorized-image").src = data.colorized;
                document.getElementById("results-section").classList.remove("hidden");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById("loading").classList.add("hidden");
            document.getElementById("colorize-btn").disabled = false;
            alert("An error occurred during processing.");
        });
});
