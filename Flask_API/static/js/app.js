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
    $("#root").load("../templates/views/home.html", function (statusTxt, jqXHR) {
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

function checkExist(elementId) {
    setInterval(function () {
        if ($(elementId).length) {
            console.log("Exists!");
            clearInterval(checkExist);
            document.getElementById('elementId').src = image
        }
    }, 100);
}


function extractFeatures() {
    console.log(image)
    loadPage('../templates/extraction.html')
    checkExist('extractionImage')
}