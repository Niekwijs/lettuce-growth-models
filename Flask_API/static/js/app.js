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

    // Create an object URL for the file
    const objectURL = URL.createObjectURL(file);

    // Assign the object URL to the passed image variable
    images[imageVariable] = objectURL;

    // Create a new FileReader
    const reader = new FileReader();

    // Clear out previous data in the document storage
    localStorage.removeItem(storageKey);

    // Add an event listener for when the file is loaded
    reader.addEventListener("load", () => {
        // Get the file"s data as a data URL
        const dataURL = reader.result;

        // Convert the data URL to a base64 string
        const base64 = dataURL.split(",")[1];

        // Store the base64 string in the document storage
        localStorage.setItem(storageKey, base64);
    });

    // Start reading the file
    reader.readAsDataURL(file);
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