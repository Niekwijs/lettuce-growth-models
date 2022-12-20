var image = ''

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

function loadFile(event) {
    image = URL.createObjectURL(event.target.files[0]);
    document.getElementById('output').src = image;
};

function extractFeatures() {
    loadPage('../templates/extraction.html')
}

function showResult() {
    loadPage('../templates/result.html')
}