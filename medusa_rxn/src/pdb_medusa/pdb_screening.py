from matterlab_hotplates import IKAHotplate
from matterlab_pumps import JKemPump
from matterlab_valves import JKem4in1Valve
from pathlib import Path
from typing import Union, Dict, Optional
import json
from math import ceil
import time
from datetime import datetime


class BaseProcedures(object):
    def __init__(self):
        self._load_ports_settings()
        self._initialize_pump()
        self._initialize_valve()
        self._initialize_hotplate()
        self._syringe_content = {"pump_port": "N2"}

    def _load_ports_settings(self):
        with open(Path.cwd().parent.parent / 'data' / 'ports_settings.json') as f:
            self.ports_settings = json.load(f)

    def _initialize_pump(self):
        self.pump = JKemPump('COM13', 6, ports=self.ports_settings["pump_ports"])

    def _initialize_valve(self):
        self.valve = []
        for i in range(1, len(self.ports_settings["valve_ports"])+1):
            valve = JKem4in1Valve(com_port = 'COM11',
                                  valve_num = i,
                                  ports = self.ports_settings["valve_ports"][i-1]
                                  )
            self.valve.append(valve)

    def _initialize_hotplate(self):
        self.hotplate = IKAHotplate(com_port ='COM14')

    def _build_fluid_connection(self,
                                zone: Dict
                                ):
        pump_port = self.pump.switch_port(port = zone['pump_port'])
        print(f"Fluid connection built for \n"
              f"Pump port {zone['pump_port']} at {pump_port}\n")
        if "valve_port" in zone:
            valve_port = self.valve[pump_port - 1].switch_port(port=zone['valve_port'])
            print(f"Fluid connection built for \n"
                  f"Valve {pump_port - 1} port {zone['valve_port']} at {valve_port}")

    def transfer_compound(self,
                          specification: str,
                          target_zone: Dict,
                          quantity: float,
                          source_zone: Dict = None,
                          compound: Union[str, None] = None,
                          delay: int = 0,
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

    def schlenk_cycle(self,
                      zones: list[Dict],
                      num_cycles: int,
                      delay: int = 1,
                      **kwargs):
        for zone in zones:
            for i in range (0, num_cycles):
                self.transfer_compound(target_zone = {"pump_port":"waste"},
                                       quantity = 10.0,
                                       source_zone = zone,
                                       delay = delay,
                                       **kwargs
                                       )
                print(f"Schlenk cycle on zone {zone} cycle {i+1} done.")

    @property
    def syringe_content(self):
        return self._syringe_content

    def rinse_syringe(self,
                      source_zone: Dict,
                      quantity: float = 0.5,
                      iterations: int = 3,
                      delay: int = 2,
                      **kwargs):
        for i in (0, iterations):
            self.transfer_compound(
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
                                   quantity=1.0,
                                   source_zone={"pump_port": "N2"},
                                   )
            print(f"Tubing to {target_zone} flushed with 1 mL N2 for {i+1} times")

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
    def __init__(self):
        super().__init__()


    def generate_experiment_details(self, num_reactions: int, temp = 60, reaction_time = 6):
        self.available_reaction_zones = []
        total_volume = (1 + num_reactions) * 1.2
        for i in range(0, num_reactions):
            self.available_reaction_zones.append({"pump_port": f"reaction_group_{i//8}",
                                                  "valve_port": f"reaction_{i}"}
                                                 )
        self.reaction_steps = {
            "1": {
                "task": "schlenk_cycle",
                "parameters": {
                    "specification": "gas",
                    "zones":  self.available_reaction_zones,
                    "num_cycles": 5,
                    "delay": 1
                }
            },

            "2": {
                "task": "schlenk_cycle",
                "parameters": {
                    "specification": "gas",
                    "zones": [{"pump_port": "Bpin"}],
                    "num_cycles": 5,
                    "delay": 1
                }
              },
            "3":{
                "task": "rinse_syringe",
                "parameters": {
                    "specification": "liquid",
                    "source_zone": {"pump_port": "Dioxane"},
                    "quantity": 1,
                    "iteration": 3,
                    "delay": 1.0
                }
            },
            "4": {
                "task": "make_solution",
                "parameters": {
                    "specification": "liquid",
                    "zone": {"pump_port": "Bpin"},
                    "quantity": total_volume,
                    "mix_iterations": 10,
                    "delay": 2.0,
                    "speed": 0.3
                }
            }
        }
        for j in range(0, num_reactions):
            self.reaction_steps[f"{j*2+5}"] = {
                "task": "add_solution",
                "parameters": {
                    "specification": "liquid",
                    "target_zone": self.available_reaction_zones[j],
                    "quantity": 1.2,
                    "source_zone": {"pump_port": "Bpin"},
                    "purge_N2": 0,
                    "delay": 2.0,
                    "rinse": False
                }
            }
            self.reaction_steps[f"{j * 2 + 6}"] = {
                "task": "add_solution",
                "parameters": {
                    "specification": "liquid",
                    "target_zone": self.available_reaction_zones[j],
                    "quantity": 1.3,
                    "source_zone": {"pump_port": "Dioxane"},
                    "purge_N2": 2,
                    "delay": 2.0,
                    "rinse": False
                }
            }

        self.reaction_steps[f"{num_reactions*2+5}"] = {
            "task": "heat_and_stir",
            "parameters": {
                "specification": "none",
                "reaction_time": reaction_time,
                "temp": temp,
                "rpm": 800
            }
        }
        self.reaction_steps[f"{num_reactions * 2 + 6}"] = {
            "task": "add_solution",
            "parameters": {
                    "specification": "liquid",
                    "target_zone": {"pump_port": "Bpin"},
                    "quantity": 1,
                    "source_zone": {"pump_port": "Dioxane"},
                    "purge_N2": 2,
                    "delay": 2.0,
                    "rinse": True
                }
        }
        with open(Path.cwd().parent.parent/'data'/'experiment_plan.json', 'w') as f:
            json.dump(self.reaction_steps, f, indent=2)

    def generate_rinse_details(self, num_reactions: int):
        self.available_reaction_zones = []

        for i in range(0, num_reactions):
            self.available_reaction_zones.append({"pump_port": f"reaction_group_{i//8}",
                                                  "valve_port": f"reaction_{i}"}
                                                 )
        self.reaction_steps = {
            "1":{
                "task": "rinse_syringe",
                "parameters": {
                    "specification": "liquid",
                    "source_zone": {"pump_port": "Dioxane"},
                    "quantity": 1,
                    "iteration": 3,
                    "delay": 1.0
                }
            }
        }
        for j in range(0, num_reactions):
            self.reaction_steps[f"{j+2}"] = {
                "task": "add_solution",
                "parameters": {
                    "specification": "liquid",
                    "target_zone": self.available_reaction_zones[j],
                    "quantity": 1,
                    "source_zone": {"pump_port": "Dioxane"},
                    "purge_N2": 2,
                    "delay": 2.0,
                    "rinse": False
                }
            }

        self.reaction_steps[f"{num_reactions + 2}"] = {
            "task": "add_solution",
            "parameters": {
                    "specification": "liquid",
                    "target_zone": {"pump_port": "Bpin"},
                    "quantity": 1,
                    "source_zone": {"pump_port": "Dioxane"},
                    "purge_N2": 2,
                    "delay": 2.0,
                    "rinse": True
                }
        }
        with open(Path.cwd().parent.parent/'data'/'experiment_plan.json', 'w') as f:
            json.dump(self.reaction_steps, f, indent=2)
    def run_reaction(self, reaction: Dict):
        if reaction["task"] == "schlenk_cycle":
            self.schlenk_cycle(**reaction["parameters"])
            print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        elif reaction["task"] == "rinse_syringe":
            self.rinse_syringe(**reaction["parameters"])
            print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        elif reaction["task"] == "make_solution":
            self.make_solution(**reaction["parameters"])
            print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
            input("Press Enter to continue")
        elif reaction["task"] == "add_solution":
            self.add_solution(**reaction["parameters"])
            print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        elif reaction["task"] == "heat_and_stir":
            self.heat_and_stir(**reaction["parameters"])
            print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))

    def run_reactions(self, rxn_steps: Dict = None):
        if rxn_steps is None:
            rxn_steps = self.reaction_steps
        for i in range(0, len(rxn_steps)):
            self.run_reaction(rxn_steps[f"{i + 1}"])

