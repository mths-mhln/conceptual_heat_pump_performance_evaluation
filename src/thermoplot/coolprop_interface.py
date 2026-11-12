import numpy as np
from CoolProp import AbstractState
import CoolProp.CoolProp as CP


class CoolPropAbstractState():
    """
    CoolProp AbstractState wrapper. allows user to use the familiar PropsSI syntax for CoolProp property extraction, while using the AbstractState under the hood for better performance. 
    The wrapper is necessary to allow vectorized evaluation of the AbstractState, which is not natively supported by CoolProp. Nan will be returned for points that are not valid for the 
    AbstractState (e.g. points outside the phase envelope). For more information on the AbstractState and its methods, see: https://coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html
    
    Methods
    -------
    PropsSI(prop, x_str, x, y_str, y)
        Extracts the specified property using the AbstractState. The input specification is automatically determined based on the x_str and y_str arguments, and the property is extracted using the 
        appropriate AbstractState method. For more information on the input specifications, see: https://coolprop.org/coolprop/wrappers/Python/html/index.html#input-specifications

        `prop` may also be a tuple of property strings, e.g. ("T", "P", "Dmass"). In that case, the AbstractState is
        updated only ONCE per (x, y) point, and all requested properties are extracted from that single update. This
        avoids the cost of repeatedly calling AS.update() for every property you want at the same state point. The
        return value is then a tuple of floats/arrays (one per requested property, in the order given), matching the
        order of `prop`.
    """
    # Lookup Tables. Replace old computational routines by a simple lookup, 
    # which allowed for less computations. 
    _MASS_PROPS = frozenset({"D", "U", "H", "S"})

    # Translation from CoolProp PropsSI method syntax to syntax 
    # accepted by CoolProp Abstractstate. See
    # https://coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html
    _TRANSLATOR = {
        "Umass": "umass",
        "Dmass": "rhomass",
        "Hmass": "hmass",
        "A": "speed_sound",
        "T": "T",
        "Q": "Q",
        "P": "p",
        "G": "gibbsmass",
        "Smass": "smass",
        "Cpmass": "cpmass",
        "Cvmass": "cvmass",
        "d(P)/d(D)|T": "drhomassdPcT",          
        "d(P)/d(D)|U": "dPdDmass_constUmass",   
        "d(P)/d(U)|D": "dPdUmass_constDmass",  
        "Phase": "phase",
        "V": "viscosity",
    }

    # Supported input pairs (both orders).  Value = (CP.xxx_INPUTS, reorder)
    # reorder=True means the user order is the reverse of the pair CoolProp accepts.
    _INPUT_SPEC = {
        "PT": (CP.PT_INPUTS, False),
        "TP": (CP.PT_INPUTS, True),
        "PUmass": (CP.PUmass_INPUTS, False),
        "UmassP": (CP.PUmass_INPUTS, True),
        "DmassP": (CP.DmassP_INPUTS, False),
        "PDmass": (CP.DmassP_INPUTS, True),
        "HmassP": (CP.HmassP_INPUTS, False),
        "PHmass": (CP.HmassP_INPUTS, True),
        "PQ": (CP.PQ_INPUTS, False),
        "QP": (CP.PQ_INPUTS, True),
        "DmassT": (CP.DmassT_INPUTS, False),
        "TDmass": (CP.DmassT_INPUTS, True),
        "DmassUmass": (CP.DmassUmass_INPUTS, False),
        "UmassDmass": (CP.DmassUmass_INPUTS, True),
        "DmassHmass": (CP.DmassHmass_INPUTS, False),
        "HmassDmass": (CP.DmassHmass_INPUTS, True),
        "DmassQ": (CP.DmassQ_INPUTS, False),
        "QDmass": (CP.DmassQ_INPUTS, True),
        "TUmass": (CP.TUmass_INPUTS, False),
        "UmassT": (CP.TUmass_INPUTS, True),
        "HmassT": (CP.HmassT_INPUTS, False),
        "THmass": (CP.HmassT_INPUTS, True),
        "QT": (CP.QT_INPUTS, False),
        "TQ": (CP.QT_INPUTS, True),
        "SmassT": (CP.SmassT_INPUTS, False),
        "TSmass": (CP.SmassT_INPUTS, True),
        "SmassUmass": (CP.SmassUmass_INPUTS, False),
        "UmassSmass": (CP.SmassUmass_INPUTS, True),
        "DmassSmass": (CP.DmassSmass_INPUTS, False),
        "SmassDmass": (CP.DmassSmass_INPUTS, True),
        "HmassSmass": (CP.HmassSmass_INPUTS, False),
        "SmassHmass": (CP.HmassSmass_INPUTS, True),
        "QSmass": (CP.QSmass_INPUTS, False),
        "SmassQ": (CP.QSmass_INPUTS, True),
        "PSmass": (CP.PSmass_INPUTS, False),
        "SmassP": (CP.PSmass_INPUTS, True),
    }

    def __init__(self, library, fluid_name):
        """
        Initializes the CoolPropAbstractState object with the specified library and fluid name. The library is typically "HEOS" for pure fluids, but can be adapted for mixtures and other libraries. 
        The name is the name of the fluid as recognized by CoolProp, e.g. "Water" or "R134a". 

        Attributes
        ----------
        Library: str
            Name of the backend library to use for extracting fluid thermodynamic properties
        Name: str
            Name of the fluid as recognized by CoolProp.
        """
        # extract properties necessary for initializing abstractstate
        if library == 'CoolProp':
            library = 'HEOS'

        # legacy code. I do not imagine myself putting a fluid name with [1] at the end, but it is in there, so i assume it can be called... 
        name = fluid_name
        if len(name) > 3 and name[-3:] == "[1]":
            name = name[:-3]

        # store fluid extraction information as class attributes
        self.FluidName = name
        self.Library = library
        self._abstract_state = None

        # initalize abstractstate:
        self._abstract_state = AbstractState(self.Library, name)

        # compute critical point properties for the fluid, to be used in the PropsSI method for points that are close to the critical point.
        Tcrit = self._abstract_state.T_critical()
        Dcrit = self._abstract_state.rhomass_critical()
        Pcrit = self._abstract_state.p_critical()
        self.critical_point_vals = (Tcrit, Dcrit, Pcrit)

    def _apply_mass_syntax(self, s: str) -> str:
        """
        Converts PropsSI syntax to AbstractState syntax. for properties that are typically mass-averaged, the subscript mass should be added behind it.
        """
        if s in self._MASS_PROPS:
            return s + "mass"
        return s

    def _apply_abstractstate_method_syntax(self, s: str) -> str:
        """
        Converts PropsSI syntax to AbstractState syntax. for properties that are typically mass-averaged, the subscript mass should be added behind it.
        """
        if s in self._TRANSLATOR:
            return self._TRANSLATOR.get(s)
        return s
               
    def _get_input_spec(self, x_str: str, y_str: str) -> tuple[CP.PQ_INPUTS, bool]:
        """
        Method to convert specified PropsSI input spec into a coolprop inputs object, required for updating the abstractstate thermodynamic state in update_and_get using the coolprop abstractstate
        update method. Note, not really CP.PQ_INPUTS, can be other pairs, but i had to give an example. 
        For all inputs, refer to "input_pairs" section of https://coolprop.org/_static/doxygen/html/namespace_cool_prop.html#aa1ce7c368d1058004293708038241850a648039a97f7392876038eaf56cf91e95

        Attributes
        ----------
        x_str: str
            String corresponding to the first input variable, e.g. "T" for temperature or "P" for pressure.
        y_str: str
            String corresponding to the second input variable, e.g. "T" for temperature or "P" for pressure.
        """
        key = x_str + y_str
        try:
            return self._INPUT_SPEC[key]
        except KeyError:
            raise ValueError(
                f"Unsupported input pair '{x_str}'+'{y_str}'. "
                f"Supported combinations: {sorted(self._INPUT_SPEC)}"
            )

    def _extract(self, AS: AbstractState, x_str_AS: str, x: float, y_str_AS: str, y: float, prop_AS: str, input_spec: CP.PQ_INPUTS, verbose: bool = False):
        """
        Vectorized method to update the AbstractState with the specified input specification and input variables, and return the specified output variable. 
        The method returns nan for points that are not valid for the AbstractState (e.g. points outside the phase envelope).

        Arguments
        ---------
        AS: AbstractState
            CoolProp AbstractState object to update and extract properties from.
        input_spec: int
            CoolProp input specification corresponding to the x_str and y_str variables, e.g. CP.PT_INPUTS for temperature and pressure inputs. This is determined in the get_input_spec method.
        x_str: str
            String corresponding to the first input variable, e.g. "T" for temperature or "P" for pressure.
        x: float
            Value of the first input variable, e.g. temperature or pressure.
        y_str: str
            String corresponding to the second input variable, e.g. "T" for temperature or "
        y: float
            Value of the second input variable, e.g. temperature or pressure.
        output: str
            String corresponding to the desired output variable, e.g. "T" for temperature or "P" for pressure. This is translated to the corresponding AbstractState method in the PropsSI method.
        reorder: bool
            Boolean indicating whether the input variables need to be reordered for the AbstractState update method as a specific order may be required to comply with CoolProp AbstractState syntax.
        
        Returns
        -------
        output: float | np.ndarray
            output of the desired variable, e.g. temperature or pressure. Will be a float for single point evaluation, or a numpy array for vectorized evaluation. 
            For points that are not valid for the AbstractState (e.g. points outside the phase envelope), nan will be returned.
        """     
        """Extract a property after a successful update."""           
        failed = CoolPropAbstractState._update_one(AS, x, y, input_spec, verbose)
        if failed:
            val = self._try_critical_recovery(
                AS, x_str_AS, x, y_str_AS, y, prop_AS
            )
        else:
            if prop_AS == "drhomassdPcT":
                val = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
            elif prop_AS == "dPdDmass_constUmass":
                val = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
            elif prop_AS == "dPdUmass_constDmass":
                val = AS.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
            else:
                val = getattr(AS, prop_AS)()  

        if prop_AS == "Q":
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
        return float(val)

    def _extract_multiple(self, AS: AbstractState, x_str_AS: str, x: float, y_str_AS: str, y: float, prop_AS_tuple: tuple[str], input_spec: CP.PQ_INPUTS, verbose: bool = False):
        """
        Same as _extract, but for a tuple of properties at the same (x, y) point. The AbstractState is updated only once, 
        after which all requested properties are read off that single update. 

        Returns
        -------
        output: tuple[float] | float
            One value per property in prop_AS_tuple, in the same order. nan for points that are not valid for the AbstractState.
            A single float if only one property is requested, as np.vectorize then expects a single output.
        """
        failed = CoolPropAbstractState._update_one(AS, x, y, input_spec, verbose)
        if failed:
            # the critical point recovery is property specific, hence fall back on _extract for every property.
            vals = [self._extract(AS, x_str_AS, x, y_str_AS, y, prop_AS, input_spec, verbose) for prop_AS in prop_AS_tuple]
        else:
            vals = []
            for prop_AS in prop_AS_tuple:
                if prop_AS == "drhomassdPcT":
                    val = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
                elif prop_AS == "dPdDmass_constUmass":
                    val = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
                elif prop_AS == "dPdUmass_constDmass":
                    val = AS.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
                else:
                    val = getattr(AS, prop_AS)()
                if prop_AS == "Q":
                    if val < 0.0:
                        val = 0.0
                    elif val > 1.0:
                        val = 1.0
                vals.append(float(val))
        return tuple(vals) if len(vals) > 1 else vals[0]

    def _update_one(AS: AbstractState, x: float, y: float, 
                    input_spec: CP.PQ_INPUTS, verbose: bool = False) -> bool:
        """
        Attempt a single update. Returns True if the update failed
        (caller should return nan or try critical recovery).
        Uses a tiny last-state cache to skip identical consecutive points.
        """
        try:
            AS.update(input_spec, x, y)
            return False
        except Exception as e:
            if verbose:
                print("Failed to update AbstractState:", e)
            return True

    def _try_critical_recovery(self, AS: AbstractState, x_str_AS: str, x: float,
                                   y_str_AS: str, y: float, out_key: str) -> float:
        """
        Called only after a normal update has failed.
        If both inputs are judged to be near the critical point, force the
        state to the critical point and return the requested property.
        Otherwise return nan.
        """
        def critical_input_value(prop_str_AS):
            Tcrit, Dcrit, Pcrit = self.critical_point_vals
            critical_values = {"T": Tcrit, "P": Pcrit, "Dmass": Dcrit}
            if prop_str_AS in critical_values:
                return critical_values[prop_str_AS]

            AS.update(CP.DmassT_INPUTS, Dcrit, Tcrit)
            Scrit = AS.smass()
            AS.update(CP.SmassT_INPUTS, Scrit, Tcrit)
            prop_method = self._apply_abstractstate_method_syntax(prop_str_AS)
            return getattr(AS, prop_method)()

        if (self._critical_value(AS, x_str_AS, x) and
                self._critical_value(AS, y_str_AS, y)):
            # Force state to critical point
            Tcrit, Dcrit, _ = self.critical_point_vals
            try:
                AS.update(CP.DmassT_INPUTS, Dcrit, Tcrit)
                S = AS.smass()
                AS.update(CP.SmassT_INPUTS, S, Tcrit)
                return getattr(AS, out_key)()
            except Exception:
                print("exception encountered for first critical point recovery attempt")
                return np.nan
        elif self._critical_value(AS, x_str_AS, x):
            # Pin the near-critical input and retain the other input value.
            try:
                x_critical = critical_input_value(x_str_AS)
                input_spec, _ = self._get_input_spec(x_str_AS, y_str_AS)
                AS.update(input_spec, x_critical, y)
                return getattr(AS, out_key)()
            except Exception:
                return np.nan
        elif self._critical_value(AS, y_str_AS, y):
            # Pin the near-critical input and retain the other input value.
            try:
                y_critical = critical_input_value(y_str_AS)
                input_spec, _ = self._get_input_spec(x_str_AS, y_str_AS)
                AS.update(input_spec, x, y_critical)
                return getattr(AS, out_key)()
            except Exception:
                return np.nan

        return np.nan

    def _critical_value(self, AS: AbstractState, prop_str_AS: str, prop_val: float) -> bool:
        """
        This method checks if the specified input value is close to the critical point, 
        and returns True if it is, and False otherwise.

        It does this by calculating the input values specified from the critical point
        T and S values (which were found, from limited testing) to always return a value. 

        Attributes
        ----------
        AS: AbstractState
            AbstractState object to extract critical point properties from.
        prop_str_AS: str
            String corresponding to the property variable, e.g. "T" for temperature or "P" for pressure.
            should comply with CoolProp AbstractState syntax, see self._PropsSI_syntax_to_AbstractState_syntax for translation from PropsSI syntax to AbstractState syntax.
        prop_val: float
            Value of the property variable, e.g. temperature or pressure.

        Returns
        -------
        bool
            True if the specified input pair is close to the critical point, False otherwise.
        """
        Tcrit, Dcrit, _ = self.critical_point_vals
        try:
            AS.update(CP.DmassT_INPUTS, Dcrit, Tcrit)
            S = AS.smass()
            AS.update(CP.SmassT_INPUTS, S, Tcrit)

            if prop_str_AS == "Q":
                Q_crit = AS.Q()
                if Q_crit < 0.0:
                    Q_crit = 0.0
                elif Q_crit > 1.0:
                    Q_crit = 1.0
            else:
                # Map the AS-style name to the key used by _extract / _getters
                translator = {
                    "Umass": "umass",
                    "Dmass": "rhomass",
                    "Hmass": "hmass",
                    "A": "speed_sound",
                    "T": "T",
                    "P": "p",
                    "Smass": "smass",
                    "Cpmass": "cpmass",
                    "Cvmass": "cvmass",
                    "d(P)/d(D)|T": "drhomassdPcT",
                    "d(P)/d(D)|U": "dPdDmass_constUmass",
                    "d(P)/d(U)|D": "dPdUmass_constDmass",
                    "Phase": "phase",
                    "V": "viscosity",
                }
                key = translator.get(prop_str_AS, prop_str_AS)
                prop_crit = getattr(AS, key)()

            return bool(np.isclose(prop_val, prop_crit, rtol=1e-5, atol=1e-5))
        except Exception:
            return False

    def PropsSI(self, prop: str | tuple[str], x_str: str = None, x: float | np.ndarray = None, y_str: str = None, y: float | np.ndarray = None, verbose: bool = False) -> float | np.ndarray:
        """
        Integral functionality, uses various methods to convert user input to an input spec accepted by AbstractState syntax, and extracts fluid thermodynamic property according to user specification. 
        using a CoolProp AbstractState syntax. 

        Attributes
        ----------
        prop: str | tuple[str]
            String corresponding to the desired output variable, e.g. "T" for temperature or "P" for pressure.
            May also be a tuple of such strings, e.g. ("T", "P", "Dmass"), to request several properties at the
            same (x, y) point. When a tuple is given, the AbstractState is updated only ONCE per point and every
            requested property is read off that single update, instead of updating once per property.
        x_str: str
            String corresponding to one of the input variables, e.g. "T" for temperature or "P" for pressure.
        x: float | np.ndarray
            Value of the first input variable, e.g. temperature or pressure. Can be a float for single point evaluation, or a numpy array for vectorized evaluation.
        y_str: str
            String corresponding to the other input variable, e.g. "T" for temperature or "P" for pressure.
        y: float | np.ndarray
            Value of the second input variable e.g. temperature or pressure. Can be a float for single point evaluation, or a numpy array for vectorized evaluation.
        
        Result
        ------
        output: float | np.ndarray | tuple[float | np.ndarray, ...]
            Value of the desired output variable, e.g. temperature or pressure. Will be a float for single point evaluation, or a numpy array for vectorized evaluation. 
            For points that are not valid for the AbstractState (e.g. points outside the phase envelope), nan will be returned.
            If `prop` was a tuple, a tuple of such values is returned instead, one per requested property, in the
            same order as `prop`.
        """
        # get previously instantiated abstractstate object
        AS = self._abstract_state

        x_str_AS = self._apply_mass_syntax(x_str)
        y_str_AS = self._apply_mass_syntax(y_str)
        
        # get input spec necessary for updating the AbstractState. 
        # if crit point is desired, the below does not work... so we need to check for that first.
        # this is simply checking 
        if x_str_AS is not None and y_str_AS is not None:
            input_spec, reorder = self._get_input_spec(x_str_AS, y_str_AS)
        else:
            reorder = False

        # if reorder, swap x and y
        if reorder:
            x, y = y, x
            x_str_AS, y_str_AS = y_str_AS, x_str_AS

        # if prop input is a tuple of props to be extracted at the same thdy points (x, y)...
        if isinstance(prop, tuple):
            # ... convert all inputs to AbstractState syntax. 
            prop_mass_tuple = tuple(self._apply_mass_syntax(p) for p in prop)
            prop_AS_tuple = tuple(self._apply_abstractstate_method_syntax(p) for p in prop_mass_tuple)
            
            for prop_AS in prop_AS_tuple:
                if prop_AS in ("Tcrit", "Pcrit", "Dcrit", "Tmax", "M", "Ttriple"):
                    if prop_AS == "Tcrit":
                        return self.critical_point_vals[0]
                    if prop_AS == "Dcrit":
                        return self.critical_point_vals[1]
                    if prop_AS == "Pcrit":
                        return self.critical_point_vals[2]
                    if prop_AS == "Tmax":
                        return AS.Tmax()
                    if prop_AS == "M":
                        return AS.molar_mass()
                    if prop_AS == "Ttriple":
                        return AS.Ttriple()

            # define a an np.vectorize object capable of looping over the x, y input, which updates the 
            # AbstractState only once per point and returns one output per requested property.  
            extractor = np.vectorize(
                self._extract_multiple,
                otypes=[float] * len(prop_AS_tuple),
                excluded={0, 1, 3, 5, 6, 7},  # everything except x (2) and y (4)
            )
            vals = extractor(AS, x_str_AS, x, y_str_AS, y, prop_AS_tuple, input_spec, verbose)
            return vals if len(prop_AS_tuple) > 1 else (vals,)

        else: 
            prop_mass = self._apply_mass_syntax(prop)
            prop_AS = self._apply_abstractstate_method_syntax(prop_mass)

            if prop_AS in ("Tcrit", "Pcrit", "Dcrit", "Tmax", "M", "Ttriple"):
                if prop_AS == "Tcrit":
                    return self.critical_point_vals[0]
                if prop_AS == "Dcrit":
                    return self.critical_point_vals[1]
                if prop_AS == "Pcrit":
                    return self.critical_point_vals[2]
                if prop_AS == "Tmax":
                    return AS.Tmax()
                if prop_AS == "M":
                    return AS.molar_mass()
                if prop_AS == "Ttriple":
                    return AS.Ttriple()
            # define a an np.vectorize object capable of looping over the x, y input.
            extractor = np.vectorize(
                self._extract,
                otypes=[float],
                excluded={0, 1, 3, 5, 6, 7, 8},  # everything except x (2) and y (3)
            )

            return extractor(AS, x_str_AS, x, y_str_AS, y, prop_AS, input_spec, verbose)


