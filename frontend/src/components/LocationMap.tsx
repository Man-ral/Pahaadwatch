"use client";

import { useEffect } from "react";
import {
  CircleMarker,
  MapContainer,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";

type Point = { latitude: number; longitude: number };

type LocationMapProps = {
  center: Point;
  selected: Point;
  onSelect: (point: Point) => void;
};

function RecenterMap({ center }: { center: Point }) {
  const map = useMap();

  useEffect(() => {
    map.flyTo([center.latitude, center.longitude], Math.max(map.getZoom(), 10), {
      duration: 0.7,
    });
  }, [center.latitude, center.longitude, map]);

  return null;
}

function ClickToSelect({ onSelect }: { onSelect: LocationMapProps["onSelect"] }) {
  useMapEvents({
    click(event) {
      onSelect({
        latitude: Number(event.latlng.lat.toFixed(6)),
        longitude: Number(event.latlng.lng.toFixed(6)),
      });
    },
  });

  return null;
}

export default function LocationMap({ center, selected, onSelect }: LocationMapProps) {
  return (
    <MapContainer
      center={[center.latitude, center.longitude]}
      zoom={9}
      scrollWheelZoom
      className="location-map"
      preferCanvas
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        maxZoom={19}
      />
      <RecenterMap center={center} />
      <ClickToSelect onSelect={onSelect} />
      <CircleMarker
        center={[selected.latitude, selected.longitude]}
        radius={9}
        pathOptions={{
          color: "#ffffff",
          weight: 4,
          fillColor: "#237b57",
          fillOpacity: 1,
        }}
      />
      <CircleMarker
        center={[selected.latitude, selected.longitude]}
        radius={19}
        pathOptions={{
          color: "#237b57",
          weight: 1,
          opacity: 0.55,
          fillColor: "#237b57",
          fillOpacity: 0.12,
        }}
        interactive={false}
      />
    </MapContainer>
  );
}
