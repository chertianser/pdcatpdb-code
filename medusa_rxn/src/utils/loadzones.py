from pathlib import Path
import json


class LoadZones(object):
    def __init__(self, zone_setting_file: Path = None):
        if zone_setting_file is None:
            zone_setting_file = Path.cwd().parent.parent / 'data' / 'zones.json'
        with open(zone_setting_file, 'r') as f:
            zone_settings = json.load(f)

        self.zones = {}
        for key, item in zone_settings.items():
            zone = LoadZone(item)
            setattr(self, key, zone)
            # self.zones[key] = zone


class LoadZone:
    def __init__(self, zone_data):
        self.chemical = zone_data["content"]["chemical"]
        self._inert_atmosphere = zone_data["content"]["inert_atmosphere"]
        self.pump_port = zone_data["location"]["pump_port"]
        self.valve_port = zone_data["location"]["valve_port"]
        self.min_volume = zone_data["size"]["min_volume"]
        self.max_volume = zone_data["size"]["max_volume"]
        self._current_volume = zone_data["size"]["current_volume"]
        self.addable = zone_data["size"]["addable"]
        self.drawable = zone_data["size"]["drawable"]

    @property
    def inert_atmosphere(self) -> bool:
        return self._inert_atmosphere

    @inert_atmosphere.setter
    def inert_atmosphere(self, inert: bool):
        self._inert_atmosphere = inert

    @property
    def current_volume(self) -> float:
        return self._current_volume

    @current_volume.setter
    def current_volume(self, volume: float):
        self._current_volume = volume

    @property
    def capacity(self) -> float:
        return self.max_volume - self.current_volume

# zones = LoadZones()

