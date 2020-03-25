// Add console.log to check to see if our code is working.
console.log("working");

// We create the tile layer that will be the background of our map.
let streets = L.tileLayer('https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}?access_token={accessToken}', {
attribution: 'Map data © <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery (c) <a href="https://www.mapbox.com/">Mapbox</a>',
	maxZoom: 18,
	accessToken: API_KEY
});

// We create the dark view tile layer that will be an option for our map.
let satelliteStreets = L.tileLayer('https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/tiles/{z}/{x}/{y}?access_token={accessToken}', {
attribution: 'Map data © <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery (c) <a href="https://www.mapbox.com/">Mapbox</a>',
	maxZoom: 18,
	accessToken: API_KEY
});

// Create a base layer that holds both maps.
let baseMaps = {
	"Streets": streets,
	"Satellite Streets": satelliteStreets
  };

// Create the map object with a center and zoom level.
let map = L.map("mapid", {
    center: [
      40.7, -94.5
    ],
    zoom: 4
  });

// Pass our map layers into our layers control and add the layers control to the map.
L.control.layers(baseMaps).addTo(map);


// Accessing the airport GeoJSON URL
let incidentData = "https://raw.githubusercontent.com/jennitian/gun-violence/jenn/static/js/incidents.json"

function styleInfo(feature) {
    return {
      opacity: 1,
      fillOpacity: .7,
      fillColor: "#cf112d",
      color: "#000000",
      radius: feature.properties.n_killed,
      stroke: true,
      weight: 0.5
    };
  }

function init() {

  
    d3.json(incidentData).then(function(data) {
        console.log(data);
        // Creating a GeoJSON layer with the retrieved data.
        L.geoJson(data, {
            // We turn each feature into a circleMarker on the map.
            pointToLayer: function(feature, latlng) {
                    console.log(data);
                    return L.circleMarker(latlng);
                },
            
            // We set the style for each circleMarker using our styleInfo function.
            style: styleInfo,
            onEachFeature: function(feature, layer) {
                layer.bindPopup("<h3> Number Killed: " + feature.properties.n_killed + "</h3> <hr><h3> Date: "
                + feature.properties.date + "</h3><hr><h3> Notes: "+ feature.properties.notes + "</h3>")
                }
            }).addTo(map);
    });
};

init();