# from matterlab_hotplates import IKAHotplate
from matterlab_pumps import JKemPump, TecanXCPump
# from matterlab_valves import JKem4in1Valve
from pathlib import Path
from typing import Union, Dict, Optional
import json
from math import ceil
import time
from datetime import datetime
# from ..utils.loadzones import LoadZones


class BaseProcedures(object):
    def __init__(self):
        self._load_ports_settings()
        self._initialize_pump()
        # self._initialize_valve()
        # self._initialize_hotplate()
        self._syringe_content = {"pump_port": "N2"}

    def _load_ports_settings(self):
        with open(Path.cwd().parent.parent / 'data' / 'sampling_ports_settings.json') as f:
            self.ports_settings = json.load(f)

    def _initialize_pump(self):
        self.pump = TecanXCPump(com_port='COM3',
                                address=0,
                                ports=self.ports_settings["pump_ports"],
                                syringe_volume=0.5e-3)

    # def _initialize_valve(self):
    #     self.valve = []
    #     for i in range(1, len(self.ports_settings["valve_ports"])+1):
    #         valve = JKem4in1Valve(com_port = 'COM11',
    #                               valve_num = i,
    #                               ports = self.ports_settings["valve_ports"][i-1]
    #                               )
    #         self.valve.append(valve)

    # def _initialize_hotplate(self):
    #     self.hotplate = IKAHotplate(com_port ='COM14')

    def _build_fluid_connection(self,
                                zone: Dict
                                ):
        pump_port = self.pump.switch_port(port = zone['pump_port'])
        print(f"Fluid connection built for \n"
              f"Pump port {zone['pump_port']} at {pump_port}\n")
        # if "valve_port" in zone:
        #     valve_port = self.valve[pump_port - 1].switch_port(port=zone['valve_port'])
        #     print(f"Fluid connection built for \n"
        #           f"Valve {pump_port - 1} port {zone['valve_port']} at {valve_port}")

    def transfer_compound(self,
                          target_zone: Dict,
                          quantity: float,
                          source_zone: Dict = None,
                          compound: Union[str, None] = None,
                          delay: int = 0,
                          specification: str = None,
                          **kwargs
                          ):
        if source_zone is None:
            source_zone = self.get_compound_location(compound)

        self._syringe_content: Dict = source_zone

        if specification == "gas":
            speed = 2.0
            self._build_fluid_connection(zone=source_zone)
            self.pump.draw(volume = quantity, speed= speed)
            time.sleep(delay)
            self._build_fluid_connection(zone = target_zone)
            self.pump.dispense_all(speed = speed)

        else:
            if "speed" in kwargs:
                speed = kwargs["speed"]
            else:
                speed = 0.5
            self._build_fluid_connection(zone=source_zone)
            self.pump.draw(volume=quantity, speed=speed)
            time.sleep(delay)
            self._build_fluid_connection(zone=target_zone)
            self.pump.dispense_all(speed=speed)

    # def get_compound_location(self, compound: Union[str, None]) -> list:
    #     """
    #     get the location of a compound
    #     :param compound:
    #     :return: list:[pump_port, Operational[valve_port]]
    #     """
    #     if compound is None:
    #         raise ValueError("Source zone and Compound both None, nothing to transfer.")
    #     # TODO add actual get location
    #     # return location

    # def schlenk_cycle(self,
    #                   zones: list[Dict],
    #                   num_cycles: int,
    #                   delay: int = 1,
    #                   **kwargs):
    #     for zone in zones:
    #         for i in range (0, num_cycles):
    #             self.transfer_compound(target_zone = {"pump_port":"waste"},
    #                                    quantity = 10.0,
    #                                    source_zone = zone,
    #                                    delay = delay,
    #                                    **kwargs
    #                                    )
    #             print(f"Schlenk cycle on zone {zone} cycle {i+1} done.")

    @property
    def syringe_content(self):
        return self._syringe_content

    def rinse_syringe(self,
                      source_zone: Dict,
                      quantity: float = 0.2,
                      iterations: int = 3,
                      delay: int = 2,
                      **kwargs):
        for i in (0, iterations):
            self.transfer_compound(#specification="liquid",
                                   target_zone={"pump_port": "waste"},
                                   quantity = quantity,
                                   source_zone=source_zone,
                                   delay = delay,
                                   **kwargs

                                   )
        print(f"Syringe has been rinsed with solution at zone {source_zone}.")

    def add_solution(self,
                     target_zone: Dict,
                     quantity: float,
                     source_zone: Dict,
                     purge_N2: int = 2,
                     delay: int = 2,
                     rinse: bool = True,
                     **kwargs
                     ):
        if rinse:
            if (self.syringe_content != source_zone) \
                    and (self.syringe_content != {"pump_port": "N2"}):
                self.rinse_syringe(source_zone)

        self.transfer_compound(target_zone = target_zone,
                               quantity=quantity,
                               source_zone=source_zone,
                               delay = delay,
                               **kwargs)
        print(f"{quantity} mL solution from zone {source_zone} transferred to {target_zone}.")

        for i in range(0, purge_N2):
            self.transfer_compound(specification='gas',
                                   target_zone=target_zone,
                                   quantity=0.5,
                                   source_zone={"pump_port": "N2"},
                                   )
            print(f"Tubing to {target_zone} flushed with 0.5 mL N2 for {i+1} times")

        self._syringe_content = source_zone

    def make_solution(self, zone: Dict,
                      quantity: float,
                      mix_iterations: int = 5,
                      delay: int = 2,
                      **kwargs):
        transfer_iterations = ceil(quantity / 10.0)
        quantity_per_iteration = quantity / transfer_iterations

        for j in range (0, transfer_iterations):
            self.add_solution(target_zone=zone,
                              quantity=quantity_per_iteration,
                              source_zone={"pump_port": "Dioxane"},
                              purge_N2=0,
                              delay = delay,
                              **kwargs
                              )
        print(f"{quantity} mL Dioxane added to stock solution zone {zone}.")

        self._syringe_content = zone

        for k in range(0, mix_iterations):
            self.add_solution(target_zone=zone,
                              quantity=10.0,
                              source_zone=zone,
                              purge_N2=2,
                              delay = delay,
                              **kwargs
                              )
        self.add_solution(target_zone=zone,
                          quantity=1,
                          source_zone=zone,
                          purge_N2=0,
                          **kwargs
                          )
        print(f"Solution at {zone} has been made with {quantity} mL of Dioxane.")

        self._syringe_content = zone

    def heat_and_stir(self,
                      reaction_time: float,
                      temp: float,
                      rpm: int,
                      **kwargs):
        self.hotplate.stir(stir_switch=True, rpm = int(rpm))
        self.hotplate.heat(heat_switch=True, temp=temp)
        print(f"Stir at {rpm} and heat at {temp} start.")
        time.sleep(reaction_time*3600)
        self.hotplate.stand_by()
        print("Heat and stir stopped.")


class Reaction(BaseProcedures):
    def __init__(self, num_points = 2):
        super().__init__()
        self.get_hplc_data_path()
        self.get_zones_from_setting()
        self.sample_number = 1
        self.num_points = num_points
        # self.reaction_to_waste()
        self.time_log={"0": time.time()}

    def get_zones_from_setting(self):
        print(Path.cwd())
        # exit()
        with open(Path.cwd().parent.parent / 'data' / 'hplc_sampling_zones.json') as f:
            self.zones = json.load(f)
            print(self.zones)

    def reaction_to_waste(self, rinse_iteration=3, rinse_quantity = 0.025):
        for i in range(0, rinse_iteration):
            self.add_solution(target_zone=self.zones["waste"],
                              quantity=rinse_quantity,
                              source_zone=self.zones["reaction"],
                              purge_N2=0,
                              delay=0,
                              rinse=0,
                              speed=0.1
                              )

    def reaction_to_hplc(self,
                         quantity: float = 0.025,
                         rinse_iteration: int = 3,
                         rinse_quantity: float = 0.025
                         ):
        for i in range(0, rinse_iteration):
            self.add_solution(target_zone=self.zones["waste"],
                              quantity = rinse_quantity,
                              source_zone=self.zones["reaction"],
                              purge_N2=0,
                              delay=0,
                              rinse=0,
                              speed=0.1
                              )
        self.add_solution(target_zone = self.zones["hplc"],
                          quantity = quantity,
                          source_zone = self.zones["reaction"],
                          purge_N2 = 1,
                          delay = 0,
                          rinse = False,
                          speed = 0.1
                          )
        self.time_log[f"{self.sample_number}"] = time.time() - self.time_log["0"]
        self.add_solution(target_zone=self.zones["waste"],
                          quantity=0.45,
                          source_zone=self.zones["acn"],
                          delay=0,
                          rinse=0,
                          speed=0.25,
                          purge_N2=0
                          )
        self.add_solution(target_zone = self.zones["hplc"],
                          quantity = 0.45,
                          source_zone = self.zones["acn"],
                          delay = 0,
                          rinse = 0,
                          speed = 0.25,
                          purge_N2 = 0
                          )
        self.add_solution(target_zone=self.zones["hplc"],
                          quantity=0.45,
                          source_zone=self.zones["acn"],
                          delay=0,
                          rinse=False,
                          speed=0.25,
                          purge_N2=1
                          )

    def hplc_to_waste(self, rinse_iterations: int = 3):
        for i in range(0, 3):
            self.add_solution(target_zone=self.zones["waste"],
                              quantity=0.5,
                              source_zone=self.zones["hplc"],
                              rinse=0,
                              purge_N2=0,
                              delay=1,
                              speed=0.5)
        for i in range(0, rinse_iterations):
            self.add_solution(target_zone=self.zones["hplc"],
                              quantity=0.4,
                              source_zone=self.zones["acn"],
                              rinse=0,
                              purge_N2=0,
                              delay=1,
                              speed=0.5)
            self.add_solution(target_zone=self.zones["waste"],
                              quantity=0.5,
                              source_zone=self.zones["hplc"],
                              rinse=0,
                              purge_N2=0,
                              delay=1,
                              speed=0.5)
        self.add_solution(target_zone=self.zones["waste"],
                          quantity=0.5,
                          source_zone=self.zones["hplc"],
                          rinse=0,
                          purge_N2=0,
                          delay=1,
                          speed=0.5)

    def get_hplc_data_path(self):
        folder_name = input("Please input the new folder name containing the .D folders\n")
        hplc_data_path = Path(r"C:\Users\aspur\OneDrive\Documents\AgilentChemStationData\1\Data\Han")/folder_name

        if hplc_data_path.exists():
            self.hplc_data_path = hplc_data_path
        else:
            raise ValueError("Wrong hplc data paht!")

    def watch_run_log(self, wait_time=30):
        """
        exit while True until self.hplc_data_path/{self.sample_number:03d}-* has been built
        sleep 5 s
        :return:
        """
        print(f"\nlooking for folder {self.sample_number}\n")
        while True:
            for d in self.hplc_data_path.iterdir():
                if f"{self.sample_number:03d}-" in str(d.name):
                    print(f"New injection folder found! Wait {wait_time} s for injection to complete")
                    time.sleep(wait_time)
                    return
                else:
                    time.sleep(1)

    def inject_samples(self):
        # print(self.sample_number, self.num_points)
        while self.sample_number <= self.num_points+2:
            self.watch_run_log()
            self.hplc_to_waste(rinse_iterations=3)
            if self.sample_number == 1:
                self.reaction_to_waste(rinse_iteration=3, rinse_quantity=0.03)
            self.reaction_to_hplc(quantity = 0.05, rinse_quantity=0.035, rinse_iteration=3)
            self.sample_number += 1
        self.hplc_to_waste(rinse_iterations=3)
        print(self.time_log)

if __name__ == "__main__":
    pdb = Reaction(num_points=10)
    pdb.inject_samples()