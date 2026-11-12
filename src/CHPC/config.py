import configparser
import os


class ConfigError(ValueError):
    """Raised when a config value is missing or invalid."""


class Config:
    # cycle specifications stored in the cycle_config dictionary, with the same keys as in the config file. Which of 
    # them are necessary depends on the analysis type and the calculation method, see inputFileCheck. 
    _CYCLE_SPECIFICATIONS = (
        "T_h_in", "T_c_in", "ṁ_h", "cp_h", "ṁ_c", "cp_c", "η_turb", "η_compr", "η_shaft",
        "ΔT_pp_min_evap", "ΔT_pp_min_cond", "Q_out_req", "PR", "ΔT_pp_1", "ΔT_pp_3", "ΔT_pp_4", "ΔT_sh",
    )

    def __init__(self, configFilePath: str):
        if not os.path.exists(configFilePath):
            raise FileNotFoundError(
                f"Config file '{configFilePath}' not found. "
                f"Check path relative to cwd: {os.getcwd()}"
            )
        self.config_file = configFilePath
        self._parser = configparser.ConfigParser()
        self._parser.read(configFilePath, encoding="utf-8-sig")  # utf-8 is necessary for keys such as ṁ_h and η_turb

        # perform input file verification checks
        self.inputFileCheck()


    # ---------------------------------------------------------------------------------------------------------
    # input extraction helper functions, also verify whether input is of correct type or within acceptable set
    # ---------------------------------------------------------------------------------------------------------

    def _get_int(
        self, 
        section: str, 
        key: str, 
        positive: bool = False, 
        default: int | None = None, 
        inputOptions: list[int] | None = None
        ) -> int:
        try:
            value =  self._parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            else:
                raise ValueError(
                f"{self.config_file} [{section}]: missing key '{key}'"
                ) from e  
        try:
            value = int(value)
        except:
            raise TypeError("Config option value cannot be converted to integer")
        
        if inputOptions is not None and value not in inputOptions:
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be one of {inputOptions}, got '{value}'"
            )
        if positive and value < 0:
            raise ValueError(
                f"{self.config_file} [{section}]: '{key}' must be positive, got {value}"
            )
        return value


    def _get_float(
        self, 
        section: str, 
        key: str, 
        positive: bool = False, 
        default: float | None = None, 
        inputOptions: list[float] | None = None
        ) -> float:
        try:
            value =  self._parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            else:
                raise ValueError(
                f"{self.config_file} [{section}]: missing key '{key}'"
                ) from e  
        try:
            value = float(value)
        except:
            raise TypeError("Config option value cannot be converted to float")
        
        if inputOptions is not None and value not in inputOptions:
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be one of {inputOptions}, got '{value}'"
            )
        if positive and value < 0:
            raise ValueError(
                f"{self.config_file} [{section}]: '{key}' must be positive, got {value}"
            )
        return value


    def _get_str(
        self, 
        section: str, 
        key: str, 
        lower: bool = False,
        default: str | None = None, 
        inputOptions: list[str] | None = None
        ) -> str:
        try:
            value =  self._parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            else:
                raise ValueError(
                f"{self.config_file} [{section}]: missing key '{key}'"
                ) from e            
        try:
            value = str(value)
        except:
            raise TypeError("Config option value cannot be converted to string")
        if lower:
            value = value.lower()
        if inputOptions is not None and value not in inputOptions:
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be one of {inputOptions}, got '{value}'"
            )
        return value


    def _get_float_list(
        self, 
        section: str, 
        key: str, 
        length: int | None = None, 
        default: list[float] | None = None
        ) -> list[float]:
        try:
            value =  self._parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            else:
                raise ValueError(
                f"{self.config_file} [{section}]: missing key '{key}'"
                ) from e
        try:
            value = [float(item) for item in value.strip().strip("[]").split(",")]
        except:
            raise TypeError(f"{self.config_file} [{section}]: '{key}' cannot be converted to a list of floats, got '{value}'")
        if length is not None and len(value) != length:
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must contain {length} values, got {len(value)}"
            )
        return value



    # ------------------------------------------------------------
    # check whether required inputs are present
    # ------------------------------------------------------------
    def inputFileCheck(self) -> None:
        """
        Verification that the necessary sections and keys are present in the config file. The type and value of the 
        inputs are verified by the input parsing methods below, similar to pyshockflow. 

        Which inputs are necessary depends on two settings: 
        a) analysis_type: a single cycle (CHPC) requires the component efficiencies, whereas a COP vs efficiency 
           investigation (COP_vs_eff) requires the efficiency bounds and the number of efficiencies instead. 
        b) calculation_method: full_optimization and fixed_PR require the minimum pinch-point temperature differences 
           and the required heat output, fixed_PR also the pressure ratio. three_temp_fix (the approach of 
           thermodynamics_legacy_4) requires the four temperature differences ΔT_pp_1, ΔT_pp_3, ΔT_pp_4 and ΔT_sh instead.
        Inputs that are not used by the specified analysis type and calculation method are prohibited, such that the 
        user cannot be under the impression that they are taken into account. 
        """
        # helper functions
        # ================
        def check_required_sections(required_sections: dict[str, list[str]]) -> None:
            """
            Check that all required sections and keys, specified in required_sections,
            are present in the config file. required_sections is a dictionary of form
            {"section": ["key1", "key2", ...]}
            """
            for section, keys in required_sections.items():
                if not self._parser.has_section(section):
                    raise ConfigError(f"{self.config_file}: missing section '{section}'")
                for key in keys:
                    if not self._parser.has_option(section, key):
                        raise ConfigError(f"{self.config_file} [{section}]: missing necessary setting in config file: '{key}'")

        def check_prohibited_keys(prohibited_sections: dict[str, list[str]], reason: str) -> None:
            """
            Check that none of the prohibited keys (specified in prohibited_sections) are
            present in the config file. prohibited_sections has the same shape as
            required_sections: {"section": ["key1", "key2", ...]}.
            `reason` is appended to the error message to explain why the key is forbidden.
            """
            for section, keys in prohibited_sections.items():
                for key in keys:
                    if self._parser.has_option(section, key):
                        raise ConfigError(
                            f"{self.config_file} [{section}]: '{key}' must not be specified "
                            f"when {reason}."
                        )   

        # check for presence of required sections and keys
        # ================================================
        requiredSectionsNKeys = {
            "ANALYSIS TYPE": ["analysis_type"],
            "ANALYSIS SPECIFICATIONS": ["refrigerant", "T_h_in", "T_c_in", "ṁ_h", "cp_h", "ṁ_c", "cp_c"],
        }
        check_required_sections(requiredSectionsNKeys)

        # check conditionally required and conditionally prohibited inputs
        # ================================================================
        # format: ("setting", [trigger_values], {"section": ["key1", "key2", ...]}), where the keys in the dictionary
        # are required (or prohibited) if the setting carries one of the trigger values. 
        settings = {"analysis_type": self.analysisType(), "calculation_method": self.calculationMethod()}
        condRequired = [
            ("analysis_type", ["CHPC"],
                {"ANALYSIS SPECIFICATIONS": ["η_turb", "η_compr"]}),

            ("analysis_type", ["COP_vs_eff"],
                {"COP VS EFF SETTINGS": ["η_turb_bounds", "η_compr_bounds", "n_η"]}),

            ("calculation_method", ["full_optimization", "fixed_PR"],
                {"ANALYSIS SPECIFICATIONS": ["ΔT_pp_min_evap", "ΔT_pp_min_cond", "Q_out_req"]}),

            ("calculation_method", ["fixed_PR"],
                {"ANALYSIS SPECIFICATIONS": ["PR"]}),

            ("calculation_method", ["three_temp_fix"],
                {"ANALYSIS SPECIFICATIONS": ["ΔT_pp_1", "ΔT_pp_3", "ΔT_pp_4", "ΔT_sh"]}),
        ]
        for setting, trigger_values, required in condRequired:
            if settings[setting] in trigger_values:
                check_required_sections(required)

        condProhibited = [
            ("analysis_type", ["COP_vs_eff"],
                {"ANALYSIS SPECIFICATIONS": ["η_turb", "η_compr"]},
                "analysis_type = COP_vs_eff, for which the efficiencies follow from [COP VS EFF SETTINGS]"),

            ("analysis_type", ["CHPC"],
                {"COP VS EFF SETTINGS": ["η_turb_bounds", "η_compr_bounds", "n_η", "existing_complete_sweep_action"]},
                "analysis_type = CHPC"),

            ("calculation_method", ["full_optimization", "three_temp_fix"],
                {"ANALYSIS SPECIFICATIONS": ["PR"]},
                "calculation_method is not fixed_PR"),

            ("calculation_method", ["three_temp_fix"],
                {"ANALYSIS SPECIFICATIONS": ["ΔT_pp_min_evap", "ΔT_pp_min_cond", "Q_out_req"]},
                "calculation_method = three_temp_fix, which fixes ΔT_pp_1, ΔT_pp_3, ΔT_pp_4 and ΔT_sh instead"),

            ("calculation_method", ["full_optimization", "fixed_PR"],
                {"ANALYSIS SPECIFICATIONS": ["ΔT_pp_1", "ΔT_pp_3", "ΔT_pp_4", "ΔT_sh"]},
                "calculation_method is not three_temp_fix"),
        ]
        for setting, trigger_values, prohibited, reason in condProhibited:
            if settings[setting] in trigger_values:
                check_prohibited_keys(prohibited, reason)



    # ------------------------------------------------------------
    # input parsing methods
    # ------------------------------------------------------------
    # ANALYSIS TYPE
    def analysisType(self) -> str:
        return self._get_str("ANALYSIS TYPE", "analysis_type", inputOptions=["CHPC", "COP_vs_eff"])

    def calculationMethod(self) -> str:
        return self._get_str("ANALYSIS TYPE", "calculation_method", default="full_optimization", 
                             inputOptions=["full_optimization", "three_temp_fix", "fixed_PR"])

    # ANALYSIS SPECIFICATIONS
    def refrigerant(self) -> str:
        return self._get_str("ANALYSIS SPECIFICATIONS", "refrigerant")

    # NUMERICS
    def nPts(self) -> int:
        return self._get_int("NUMERICS", "n_pts", positive=True, default=150)

    # COP VS EFF SETTINGS
    def etaTurbBounds(self) -> list[float]:
        return self._get_float_list("COP VS EFF SETTINGS", "η_turb_bounds", length=2)

    def etaComprBounds(self) -> list[float]:
        return self._get_float_list("COP VS EFF SETTINGS", "η_compr_bounds", length=2)

    def nEta(self) -> int:
        return self._get_int("COP VS EFF SETTINGS", "n_η", positive=True)

    def existingCompleteSweepAction(self) -> str:
        return self._get_str("COP VS EFF SETTINGS", "existing_complete_sweep_action", lower=True, default="prompt", 
                             inputOptions=["prompt", "abort", "recompute"])



    # ------------------------------------------------------------
    # dictionaries used throughout CHPC
    # ------------------------------------------------------------
    def cycleConfig(self) -> dict:
        """
        Cycle specifications in the dictionary format used throughout CHPC, with the same keys as in the config file. 
        Only the specifications present in the config file are included, η_shaft is 1 if it is not specified. 
        """
        cycle_config = {"refrigerant": self.refrigerant(), "η_shaft": 1.0}
        for key in self._CYCLE_SPECIFICATIONS:
            if self._parser.has_option("ANALYSIS SPECIFICATIONS", key):
                cycle_config[key] = self._get_float("ANALYSIS SPECIFICATIONS", key, positive=True)
        return cycle_config

    def generalConfig(self) -> dict:
        """
        General settings in the dictionary format used throughout CHPC. 
        """
        general_config = {
            "analysis_type": self.analysisType(),
            "calculation_method": self.calculationMethod(),
            "n_pts": self.nPts(),
        }
        if general_config["analysis_type"] == "COP_vs_eff":
            general_config.update({
                "η_turb_bounds": self.etaTurbBounds(),
                "η_compr_bounds": self.etaComprBounds(),
                "n_η": self.nEta(),
                "existing_complete_sweep_action": self.existingCompleteSweepAction(),
            })
        return general_config
