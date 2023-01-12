var images = {
    rgb: '',
    depth: ''
}

function loadRGBFile(event) {
    images.rgb = URL.createObjectURL(event.target.files[0]);
    document.getElementById('outputRGB').src = images.rgb;
};

function loadDepthFile(event) {
    images.depth = URL.createObjectURL(event.target.files[0]);
    document.getElementById('outputDepth').src = images.depth;
};