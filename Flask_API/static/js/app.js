var images = {
    rgb: '',
    depth: ''
}

function loadPage(route) {
    $("#root").load(route, function (statusTxt, jqXHR) {
        if (statusTxt == "success") {
            alert("New content loaded successfully!");
        }
        if (statusTxt == "error") {
            alert("Error: " + jqXHR.status + " " + jqXHR.statusText);
        }
    });
}

$(document).ready(function () {
    $("#root").load("../templates/home.html", function (statusTxt, jqXHR) {
        if (statusTxt == "success") {
            alert("New content loaded successfully!");
        }
        if (statusTxt == "error") {
            alert("Error: " + jqXHR.status + " " + jqXHR.statusText);
        }
    });

    $("#extraction-button").click(function () {
        loadPage('../templates/extraction.html')
    });

    $("#home-button").click(function () {
        loadPage('../templates/home.html')
    });

    $("#result-button").click(function () {
        loadPage('../templates/result.html')
    });
});

function loadRGBFile(event) {
    images.rgb = URL.createObjectURL(event.target.files[0]);
    document.getElementById('outputRGB').src = images.rgb;
};

function loadDepthFile(event) {
    images.depth = URL.createObjectURL(event.target.files[0]);
    document.getElementById('outputDepth').src = images.depth;
};

function extractFeatures() {
    $.post("/image", {
        rgb: images.rgb,
        depth: images.depth
    })
    // loadPage('../templates/extraction.html')
}

function showResult() {
    loadPage('../templates/result.html')
}