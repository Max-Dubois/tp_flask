var map = L.map('map').setView([50.952, 1.881], 15);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        L.marker([50.952, 1.881]).addTo(map)
            .bindPopup('<b>Maxence Dubois</b><br>ULCO Calais.')
            .openPopup();