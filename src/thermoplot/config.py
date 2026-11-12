###########################################
# Imports
###########################################
import configparser
import numpy as np




class Config:
    def __init__(self, config_file: str) -> None:
        # Parse config file (.ini file) using configparser. Perform first input checks and store diagram type to self.
        self.config_file = config_file
        self.config_parser = configparser.ConfigParser()
        self.config_parser.optionxform = str  # Preserve case sensitivity of keys
        self.config_parser.read(config_file)
        if not self.config_parser.has_section("THERMOPLOT SETTINGS"):
            raise ValueError("Missing 'THERMOPLOT SETTINGS' section in config file")
        try:
            self.diagram_type = str(self.config_parser.get("THERMOPLOT SETTINGS", "diagram_type"))
        except:
            raise ValueError("Missing necessary thermoplot setting: diagram_type")

    def _parse_str_to_value(self, value: str) -> float | str | bool | list:
        """
        Helper function to parse values from the config file. It checks if the value is a list 
        (enclosed in square brackets), a boolean (true/false), or a float. If none of these, 
        it returns the value as a string.
        """
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            return [self._parse_str_to_value(item) for item in value[1:-1].split(",")]
        elif value.lower() in ["true", "false"]:
            return value.lower() == "true"
        else:
            try:
                if np.isclose(float(value), int(float(value))):
                    return int(float(value))
                return float(value)
            except ValueError:
                return value

    def get_thermoplot_settings(self) -> dict[str, float]:
        """
        Verify is all necessary settings are specified in the config file. If not, inform user
        on which are missing. If yes, extract all settings and store to self in dictionary format 
        for future extraction. 
        """
        necessary_thermoplot_settings = [
            "diagram_type",
            "fluid_name",
            "show_spinodal",
            "show_isolines",
            "show_critical_point",
            "show_critical_isoline",
            "n_pts",
            "latex_formatting"
        ]
        if self.diagram_type == "TS":
            necessary_thermoplot_settings.extend([
                "S_range",
                "T_range"
            ])
        elif self.diagram_type == "PH":
            necessary_thermoplot_settings.extend([
                "P_range",
                "H_range"
            ])
        if not all(req in self.config_parser.options("THERMOPLOT SETTINGS") for req in necessary_thermoplot_settings):
            missing_reqs = [req for req in necessary_thermoplot_settings if req not in self.config_parser.options("THERMOPLOT SETTINGS")]
            raise ValueError(f"Missing necessary thermoplot settings in config file: {missing_reqs}")
        self.thermoplot_settings = {key: self._parse_str_to_value(value) for key, value in self.config_parser.items("THERMOPLOT SETTINGS")}

        