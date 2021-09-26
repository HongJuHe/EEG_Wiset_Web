var map;
let service;
var directionsDisplay, directionsService;

var latitude, longitude;

const Gus = ["강동구", "중랑구", "노원구", "도봉구", "강북구",
"성북구", "동대문구", "광진구", "성동구", "종로구", "은평구", "서대문구",
"중구", "용산구", "송파구", "강남구", "서초구", "마포구", "강서구",
"영등포구", "양천구", "구로구", "동작구", "관악구", "금천구"];

var geo_options = {
    enableHighAccuracy: true,
    maximumAge        : 30000,
    timeout           : 27000
};

window.onload = getLocation;
function getLocation(){
    if(navigator.geolocation){
        navigator.geolocation.getCurrentPosition(locationSuccess, locationError, geo_options);
    }else{
        console.log("No Current Location");
    }
};
function locationSuccess(p){
    latitude = p.coords.latitude;
    longitude = p.coords.longitude;
    initialize();
}
function locationError(error){
    var errorTypes = {
        0 : "Error",
        1 : "Request Denied",
        2 : "No Current Location",
        3 : "Response Time Over"
    };
    var errorMsg = errorTypes[error.code];
}

function initialize() {
    directionsDisplay = new google.maps.DirectionsRenderer();
    directionsService = new google.maps.DirectionsService();

    var myLatLng = new google.maps.LatLng(latitude, longitude);
    var mapOptions = {
        zoom: 16,
        center: myLatLng
    };

    map = new google.maps.Map(document.getElementById('map-canvas'), mapOptions);

    const geocoder = new google.maps.Geocoder();
    const infowindow = new google.maps.InfoWindow();
    geocodeLatLng(geocoder, map, infowindow);

    service = new google.maps.places.PlacesService(map);
    directionsDisplay.setMap(map); // show map after load everything
}

function geocodeLatLng(geocoder, map, infowindow) {
    const latlng = {
        lat: latitude,
        lng: longitude,
    };

    geocoder.geocode({ location: latlng }).then((response) => {
        if (response.results[0]) {
            map.setZoom(16);

            const marker = new google.maps.Marker({
                position: latlng,
                map: map,
            });

            infowindow.setContent("현재 위치: " + response.results[0].formatted_address);
            infowindow.open(map, marker);

            setStatement(response.results[0].formatted_address);
            getStatement();
        } else {
            console.log("No results found");
        }
    }) .catch((e) => console.log("Geocoder failed due to: " + e));

}

function setStatement(sentence) {
    const mapInfo = document.getElementById("map-info");
    mapInfo.innerText = "현재 위치: " + sentence;
}

function getStatement() {
    var string = document.getElementById("map-info").innerText;
    var strArray = string.split(' ');
    var locationGu = strArray[4];

    const data = JSON.parse(JSON.stringify(Params));
    // console.log(typeof(data)) // => object
    // console.log(data) // => FULL OBJECT

    var description = "<hr>";
    const listInfo = document.getElementById("list-info");

    for (const key in Object.keys(data)) {
        const Gu = Object.keys(data)[key];

        if(Gu == locationGu) {
            const names = Object.values(data[Gu])[0];
            const scores = Object.values(data[Gu])[1];
            const links = Object.values(data[Gu])[2];
            const addresses = Object.values(data[Gu])[3];

            var maxLength = names.length;

            var order = Object.keys(scores);
            for(var i = 0; i < maxLength - 1; i++) {
                for(var j = i+1; j < maxLength; j++) {
                    if(scores[i] < scores[j]) {
                        var tmp;
                        tmp = scores[i];
                        scores[i] = scores[j];
                        scores[j] = tmp;

                        tmp = order[i];
                        order[i] = order[j];
                        order[j] = tmp;
                    }
                }
            }

            for(var i = 0; i < maxLength; i++) {
                var j = order[i];
                description += makeInformation(names[j], scores[i], links[j], addresses[j]);
            }

            listInfo.innerHTML = description;
        }
    }

}

function makeInformation(name, score, link, address) {
    var string = ''

    string += '<h4 onclick="geocodeAddress(' + "'" + name + "', '" + address + "'" + ')">';
    string += name + '</h4>';

    string += '<h6>';
    string += address + '</h6>';

    var starCount = 5;
    while(score >= 1) {
        string += '<i class="fas fa-star fa-2x"></i>';
        score -= 1;
        starCount--;
    }
    if(score >= 0.5) {
        string += '<i class="fas fa-star-half-alt fa-2x"></i>';
        starCount--;
    }
    while(starCount > 0) {
        string += '<i class="far fa-star fa-2x"></i>';
        starCount--;
    }

    string += '<br><br>';
    if(link == "없습니다") {
        string += '<a class="btn btn-xl btn-light me-4" href="#!">';
        string += '병원 홈페이지 준비중</a>';
    } else {
        string += '<a class="btn btn-dark btn-xl" href="';
        string += link + '" target="_blank">병원 홈페이지 바로가기</a>';
    }

    return string + '<hr color="navy">';
}

function geocodeAddress(name, address) {
    var geocoder = new google.maps.Geocoder();

    geocoder.geocode({'address': address + " " + name}, function(result, status) {
        console.log(result);
        console.log(status);

        if (status === 'OK') {
            map.setCenter(result[0].geometry.location);
            map.setZoom(18);

            var infowindow = new google.maps.InfoWindow({
                content: name + ", " + address,
            });
            var marker = new google.maps.Marker({
                map: map,
                position: result[0].geometry.location
            });
            marker.addListener("click", () => {
                infowindow.open({
                    map,
                    anchor: marker,
                    shouldFocus: true,
                });
            });
        } else {
            alert('Geocode Failed: ' + status);
        }
    });
}