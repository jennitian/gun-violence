// Add console.log to check to see if our code is working.
console.log("working");

// Create the map object with a center and zoom level.
// Create the map object with a center and zoom level.
let map = L.map("mapid", {
    center: [
      40.7, -94.5
    ],
    zoom: 4
  });

// We create the tile layer that will be the background of our map.
// We create the tile layer that will be the background of our map.
let streets = L.tileLayer('https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}?access_token={accessToken}', {
attribution: 'Map data © <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery (c) <a href="https://www.mapbox.com/">Mapbox</a>',
	maxZoom: 18,
	accessToken: API_KEY
});

// Then we add our 'graymap' tile layer to the map.
streets.addTo(map);

// Accessing the airport GeoJSON URL
let incidentData = "https://raw.githubusercontent.com/jennitian/gun-violence/jenn/Resources/incidents.json"

// Grabbing our GeoJSON data.
// Grabbing our GeoJSON data.
d3.json(incidentData).then(function(data) {
  console.log(data);
  let coordinates = data.lat_lng;
  Object.keys(coordinates).forEach(function (key){
    let object = coordinates[key]
    L.geoJson(object)
    .addTo(map);});


// Creating a GeoJSON layer with the retrieved data.

});
