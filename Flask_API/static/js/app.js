var images = {
    rgb: '',
    depth: ''
}

window.onload = (event) => {
    if (document.URL.indexOf("index") > -1) {
        alert("This is home");
    }

    if (localStorage.getItem('rgbFile') !== null && localStorage.getItem('depthFile') !== null) {
        document.getElementById("outputRGB").src = "data:image/jpg;base64," + localStorage.getItem('rgbFile');
        document.getElementById("outputDepth").src = "data:image/jpg;base64," + localStorage.getItem('depthFile');

        // let rgbRoot = document.getElementById("outputRGB");
        // let depthRoot = document.getElementById("outputDepth");

        // let rgbInput = document.getElementById('rgbInput')
        // let depthInput = document.getElementById('depthInput')

        // if (rgbRoot.src !== "" && depthRoot.src !== "") {
        //     fillInput('rgbFile', 'rgb.jpg', rgbInput)
        //     fillInput('depthFile', 'depth.jpg', depthInput)
        // }
        //     }
        // }

        // function fillInput(file, fileName, fileSrcTarget) {
        //     const dataTransfer = new DataTransfer();
        //     const newFile = new File([localStorage.getItem(file)], fileName, {
        //         type: 'image/jpg',
        //         lastModified: new Date(),
        //     });

        //     dataTransfer.items.add(newFile);

        //     fileSrcTarget.files = dataTransfer.files

        //     dataTransfer.clearData()
    }
}

function handleFileUpload(event, storageKey, imageVariable, imgOutput) {
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

        // Set html image src to uploaded file
        document.getElementById(imgOutput).src = "data:image/jpg;base64," + localStorage.getItem(storageKey);
    });

    // Set the image's src to the selected file
    img.src = URL.createObjectURL(file);
}
