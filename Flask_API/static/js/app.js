var images = {
    rgb: '',
    depth: ''
}

function handleFileUpload(event, storageKey, imageVariable) {
    // Get the file from the input event
    const file = event.target.files[0];

    // Create a new Image object
    const img = new Image();

    // Add an event listener for when the image is loaded
    img.addEventListener("load", () => {
        // Calculate the new width and height based on the original dimensions and a desired scale factor
        const newWidth = img.width * 0.3;
        const newHeight = img.height * 0.3;

        // Create a canvas element
        const canvas = document.createElement("canvas");

        // Set the canvas to the desired size
        canvas.width = newWidth;
        canvas.height = newHeight;

        // Draw the image on the canvas
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, newWidth, newHeight);

        // Get the data URL of the resized image
        const dataURL = canvas.toDataURL();

        // Convert the data URL to a base64 string
        const base64 = dataURL.split(",")[1];

        // Store the base64 string in the document storage
        localStorage.setItem(storageKey, base64);

        // Create an object URL for the file
        const objectURL = URL.createObjectURL(file);

        // Assign the object URL to the passed image variable
        images[imageVariable] = objectURL;
    });

    // Set the image's src to the selected file
    img.src = URL.createObjectURL(file);
}


function loadRGBFile(imageURI) {
    // handleFileUpload(event, "rgbFile", "rgb");
    document.getElementById("outputRGB").src = "data:image/jpg;base64," + localStorage.getItem('rgbFile');
}

function loadDepthFile(event) {
    handleFileUpload(event, "depthFile", "depth");
    document.getElementById("outputDepth").src = "data:image/jpeg;base64," + images.depth;
}
