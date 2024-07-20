from typing import Dict, List, Optional, Tuple


class Vial(object):
    def __init__(self, min_volume: float, max_volume: float, current_volume: Optional[float] = None) -> None:
        """

        Args:
            min_volume: Minimum volume of the vial
            max_volume: Maximum volume of the vial
            current_volume: Current volume of the vial
        """
        self.min_volume: float = min_volume
        self.max_volume: float = max_volume
        if not current_volume:
            self.current_volume: float = min_volume
        else:
            self.current_volume: float = current_volume

    @property
    def usable_volume(self) -> float:
        """Volume of the vial that can be used to remove liquid."""
        return self.current_volume - self.min_volume

    @property
    def capacity(self) -> float:
        """Volume of the vial that can be used to add liquid."""
        return self.max_volume - self.current_volume

    def available(self, volume_to_check: float) -> bool:
        """
        Check if the vial has enough capacity to hold the volume to check.

        Args:
            volume_to_check: Volume to check
        """
        return self.min_volume <= self.current_volume + volume_to_check <= self.max_volume

    def add(self, volume_added: float) -> float:
        """
        Add volume to the vial.

        Args:
            volume_added: Volume to add
        """
        self.current_volume += volume_added
        return volume_added

    def remove(self, volume_removed: float) -> float:
        self.current_volume -= volume_removed
        return volume_removed


