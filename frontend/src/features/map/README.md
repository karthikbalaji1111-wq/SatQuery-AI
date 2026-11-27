# Map

This is the map on the website. It shows the satellite image in its correct place on top of a normal street map, along with the outline of the satellite scene and, for a comparison, the change map.

The backend works out where the image goes. It sends the four corners of the image in latitude and longitude, and the map just puts the picture there. The map doesn't do any coordinate maths of its own.

We use four corners rather than a simple box because a satellite image turned into map coordinates isn't a perfect rectangle. Over a city-sized area, treating it like one put the image up to about 144 m off.

The street map comes from OpenStreetMap's public tile server. It needs no key, which is good for a demo, but a bigger deployment should use a tile service that allows that amount of traffic. You can change it with VITE_BASEMAP_TILE_URL.
