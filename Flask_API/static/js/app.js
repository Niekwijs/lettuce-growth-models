var images = {
    rgb: '',
    depth: ''
}

// function loadPage(route) {
//     $("#root").load(route, function (statusTxt, jqXHR) {
//         if (statusTxt == "success") {
//             alert("New content loaded successfully!");
//         }
//         if (statusTxt == "error") {
//             alert("Error: " + jqXHR.status + " " + jqXHR.statusText);
//         }
//     });
// }

// $(document).ready(function () {
//     $("#root").load("../templates/home.html", function (statusTxt, jqXHR) {
//         if (statusTxt == "success") {
//             alert("New content loaded successfully!");
//         }
//         if (statusTxt == "error") {
//             alert("Error: " + jqXHR.status + " " + jqXHR.statusText);
//         }
//     });

//     $("#extraction-button").click(function () {
//         loadPage('../templates/extraction.html')
//     });

//     $("#home-button").click(function () {
//         loadPage('../templates/home.html')
//     });

//     $("#result-button").click(function () {
//         loadPage('../templates/result.html')
//     });
// });

//function loadRGBFile(event) {
//    images.rgb = URL.createObjectURL(event.target.files[0]);
//    document.getElementById('outputRGB').src = images.rgb;
//};
//
//function loadDepthFile(event) {
//    images.depth = URL.createObjectURL(event.target.files[0]);
//    document.getElementById('outputDepth').src = images.depth;
//};


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


function loadRGBFile(event) {
    handleFileUpload(event, "rgbFile", "rgb");
    document.getElementById("outputRGB").src = images.rgb;
}

function loadDepthFile(event) {
    handleFileUpload(event, "depthFile", "depth");
    document.getElementById("outputDepth").src = images.depth;
}

// function extractFeatures() {
//     $.post("/image", {
//         rgb: images.rgb,
//         depth: images.depth
//     })
//     // loadPage('../templates/extraction.html')
// }

// function showResult() {
//     loadPage('../templates/result.html')
// }