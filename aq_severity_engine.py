"""
Air Quality Severity Classification Rule Engine
===============================================

This module implements a flexible, UI-editable rule engine for classifying
air quality severity based on multiple environmental parameters.

Architecture:
1. Severity levels are defined with thresholds for PM10, PM2.5, wind, humidity
2. Rules are stored in a JSON-friendly format for easy React UI integration
3. The engine evaluates conditions in order and returns the matching severity
4. Optional modifiers adjust severity based on time of day and zone sensitivity
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Literal
from datetime import datetime
import json


# ============================================================================
# SEVERITY LEVELS DEFINITION
# ============================================================================

@dataclass
class SeverityThresholds:
    """
    Defines the threshold ranges for a single severity level.
    All thresholds use min/max ranges for flexibility.
    
    Attributes:
        pm10_min: Minimum PM10 value (μg/m³)
        pm10_max: Maximum PM10 value (μg/m³), None = no upper limit
        pm25_min: Minimum PM2.5 value (μg/m³)
        pm25_max: Maximum PM2.5 value (μg/m³), None = no upper limit
        wind_min: Minimum wind speed (m/s)
        wind_max: Maximum wind speed (m/s)
        humidity_min: Minimum relative humidity (%)
        humidity_max: Maximum relative humidity (%)
    """
    pm10_min: float = 0
    pm10_max: Optional[float] = None
    pm25_min: float = 0
    pm25_max: Optional[float] = None
    wind_min: float = 0
    wind_max: Optional[float] = None
    humidity_min: float = 0
    humidity_max: Optional[float] = None


@dataclass
class SeverityLevel:
    """
    Complete definition of a severity level including thresholds and metadata.
    
    Attributes:
        level: Severity identifier (e.g., 'normal', 'moderate', 'critical')
        display_name: Human-readable name for UI
        priority: Evaluation order (lower = checked first)
        color: Hex color code for UI display
        thresholds: Environmental parameter thresholds
        description: User-facing description of conditions
    """
    level: str
    display_name: str
    priority: int
    color: str
    thresholds: SeverityThresholds
    description: str


# ============================================================================
# DEFAULT SEVERITY CONFIGURATION
# ============================================================================

# Standard severity levels based on common air quality indices
# These thresholds can be customized via the UI
DEFAULT_SEVERITY_LEVELS = [
    SeverityLevel(
        level="good",
        display_name="Good",
        priority=1,
        color="#00E400",
        thresholds=SeverityThresholds(
            pm10_max=50,
            pm25_max=12,
            wind_min=0,
            humidity_min=0
        ),
        description="Air quality is satisfactory, and air pollution poses little or no risk."
    ),
    SeverityLevel(
        level="moderate",
        display_name="Moderate",
        priority=2,
        color="#FFFF00",
        thresholds=SeverityThresholds(
            pm10_min=50,
            pm10_max=100,
            pm25_min=12,
            pm25_max=35.4,
            wind_min=0,
            humidity_min=0
        ),
        description="Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    ),
    SeverityLevel(
        level="unhealthy_sensitive",
        display_name="Unhealthy for Sensitive Groups",
        priority=3,
        color="#FF7E00",
        thresholds=SeverityThresholds(
            pm10_min=100,
            pm10_max=250,
            pm25_min=35.4,
            pm25_max=55.4,
            wind_min=0,
            humidity_min=0
        ),
        description="Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    ),
    SeverityLevel(
        level="unhealthy",
        display_name="Unhealthy",
        priority=4,
        color="#FF0000",
        thresholds=SeverityThresholds(
            pm10_min=250,
            pm10_max=350,
            pm25_min=55.4,
            pm25_max=150.4,
            wind_min=0,
            humidity_min=0
        ),
        description="Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    ),
    SeverityLevel(
        level="very_unhealthy",
        display_name="Very Unhealthy",
        priority=5,
        color="#8F3F97",
        thresholds=SeverityThresholds(
            pm10_min=350,
            pm10_max=430,
            pm25_min=150.4,
            pm25_max=250.4,
            wind_min=0,
            humidity_min=0
        ),
        description="Health alert: The risk of health effects is increased for everyone."
    ),
    SeverityLevel(
        level="hazardous",
        display_name="Hazardous",
        priority=6,
        color="#7E0023",
        thresholds=SeverityThresholds(
            pm10_min=430,
            pm25_min=250.4,
            wind_min=0,
            humidity_min=0
        ),
        description="Health warning of emergency conditions: everyone is more likely to be affected."
    ),
]


# ============================================================================
# OPTIONAL MODIFIERS
# ============================================================================

@dataclass
class TimeModifier:
    """
    Adjusts severity based on time of day.
    
    Example: Dust suppression activities might reduce severity during certain hours.
    
    Attributes:
        start_hour: Start of time window (24-hour format)
        end_hour: End of time window (24-hour format)
        severity_adjustment: Number of levels to adjust (-1 = less severe, +1 = more severe)
        reason: Explanation for the adjustment
    """
    start_hour: int
    end_hour: int
    severity_adjustment: int
    reason: str


@dataclass
class ZoneSensitivity:
    """
    Zone-specific sensitivity multipliers.
    
    Certain areas (hospitals, schools, residential) may require stricter thresholds.
    
    Attributes:
        zone_type: Type of zone (e.g., 'residential', 'hospital', 'industrial')
        threshold_multiplier: Multiplier for all thresholds (< 1.0 = stricter)
        description: Explanation of zone sensitivity
    """
    zone_type: str
    threshold_multiplier: float
    description: str


# Common zone sensitivities
DEFAULT_ZONE_SENSITIVITIES = {
    "residential": ZoneSensitivity(
        zone_type="residential",
        threshold_multiplier=0.9,
        description="10% stricter thresholds for residential areas"
    ),
    "hospital": ZoneSensitivity(
        zone_type="hospital",
        threshold_multiplier=0.8,
        description="20% stricter thresholds near healthcare facilities"
    ),
    "school": ZoneSensitivity(
        zone_type="school",
        threshold_multiplier=0.85,
        description="15% stricter thresholds near schools"
    ),
    "industrial": ZoneSensitivity(
        zone_type="industrial",
        threshold_multiplier=1.0,
        description="Standard thresholds for industrial zones"
    ),
}


# ============================================================================
# RULE ENGINE
# ============================================================================

class SeverityRuleEngine:
    """
    Main rule engine for air quality severity classification.
    
    This engine:
    1. Loads severity rules from configuration
    2. Evaluates environmental conditions against rules
    3. Applies optional modifiers (time, zone)
    4. Returns the appropriate severity level
    
    The engine is designed to be easily configurable via JSON,
    making it simple for a React UI to read and modify rules.
    """
    
    def __init__(self, 
                 severity_levels: Optional[List[SeverityLevel]] = None,
                 zone_sensitivities: Optional[Dict[str, ZoneSensitivity]] = None,
                 time_modifiers: Optional[List[TimeModifier]] = None):
        """
        Initialize the rule engine with severity levels and modifiers.
        
        Args:
            severity_levels: List of severity level definitions
            zone_sensitivities: Dictionary of zone-specific adjustments
            time_modifiers: List of time-based adjustments
        """
        self.severity_levels = severity_levels or DEFAULT_SEVERITY_LEVELS
        self.zone_sensitivities = zone_sensitivities or DEFAULT_ZONE_SENSITIVITIES
        self.time_modifiers = time_modifiers or []
        
        # Sort severity levels by priority for efficient evaluation
        self.severity_levels.sort(key=lambda x: x.priority)
    
    def classify(self,
                pm10: float,
                pm25: float,
                wind_speed: float,
                humidity: float,
                zone_type: Optional[str] = None,
                timestamp: Optional[datetime] = None) -> Dict:
        """
        Classify air quality severity based on environmental parameters.
        
        Algorithm:
        1. Apply zone sensitivity multiplier to thresholds if applicable
        2. Evaluate conditions against each severity level (in priority order)
        3. Apply time-based modifiers if applicable
        4. Return the matched severity level with metadata
        
        Args:
            pm10: PM10 concentration (μg/m³)
            pm25: PM2.5 concentration (μg/m³)
            wind_speed: Wind speed (m/s)
            humidity: Relative humidity (%)
            zone_type: Optional zone type for sensitivity adjustment
            timestamp: Optional timestamp for time-based modifiers
        
        Returns:
            Dictionary containing:
                - level: Severity level identifier
                - display_name: Human-readable name
                - color: Hex color code
                - description: Severity description
                - confidence: Classification confidence (0-1)
                - applied_modifiers: List of modifiers that were applied
                - threshold_exceeded: Which thresholds were exceeded
        """
        # Step 1: Apply zone sensitivity if specified
        zone_multiplier = 1.0
        applied_modifiers = []
        
        if zone_type and zone_type in self.zone_sensitivities:
            zone_sens = self.zone_sensitivities[zone_type]
            zone_multiplier = zone_sens.threshold_multiplier
            applied_modifiers.append({
                "type": "zone_sensitivity",
                "zone": zone_type,
                "multiplier": zone_multiplier,
                "description": zone_sens.description
            })
        
        # Step 2: Find matching severity level
        matched_level = None
        threshold_details = []
        
        for level in self.severity_levels:
            thresholds = level.thresholds
            
            # Apply zone multiplier to thresholds
            pm10_min = thresholds.pm10_min * zone_multiplier
            pm10_max = thresholds.pm10_max * zone_multiplier if thresholds.pm10_max else None
            pm25_min = thresholds.pm25_min * zone_multiplier
            pm25_max = thresholds.pm25_max * zone_multiplier if thresholds.pm25_max else None
            
            # Check if all conditions are met
            conditions_met = True
            details = []
            
            # PM10 check
            if pm10 < pm10_min:
                conditions_met = False
            elif pm10_max and pm10 > pm10_max:
                conditions_met = False
            else:
                details.append(f"PM10: {pm10:.1f} μg/m³ (threshold: {pm10_min:.1f}-{pm10_max if pm10_max else '∞'})")
            
            # PM2.5 check
            if pm25 < pm25_min:
                conditions_met = False
            elif pm25_max and pm25 > pm25_max:
                conditions_met = False
            else:
                details.append(f"PM2.5: {pm25:.1f} μg/m³ (threshold: {pm25_min:.1f}-{pm25_max if pm25_max else '∞'})")
            
            # Wind speed check (if limits are set)
            if thresholds.wind_max:
                if wind_speed < thresholds.wind_min or wind_speed > thresholds.wind_max:
                    conditions_met = False
                else:
                    details.append(f"Wind: {wind_speed:.1f} m/s (threshold: {thresholds.wind_min}-{thresholds.wind_max})")
            
            # Humidity check (if limits are set)
            if thresholds.humidity_max:
                if humidity < thresholds.humidity_min or humidity > thresholds.humidity_max:
                    conditions_met = False
                else:
                    details.append(f"Humidity: {humidity:.1f}% (threshold: {thresholds.humidity_min}-{thresholds.humidity_max})")
            
            # If all conditions met, this is our severity level
            if conditions_met:
                matched_level = level
                threshold_details = details
                break
        
        # If no level matched, default to most severe
        if not matched_level:
            matched_level = self.severity_levels[-1]
            threshold_details = ["All thresholds exceeded - defaulting to most severe level"]
        
        # Step 3: Apply time-based modifiers if specified
        final_level = matched_level
        if timestamp and self.time_modifiers:
            hour = timestamp.hour
            for modifier in self.time_modifiers:
                if modifier.start_hour <= hour < modifier.end_hour:
                    # Adjust severity level
                    current_idx = self.severity_levels.index(matched_level)
                    new_idx = max(0, min(len(self.severity_levels) - 1, 
                                        current_idx + modifier.severity_adjustment))
                    final_level = self.severity_levels[new_idx]
                    
                    applied_modifiers.append({
                        "type": "time_modifier",
                        "time_window": f"{modifier.start_hour:02d}:00-{modifier.end_hour:02d}:00",
                        "adjustment": modifier.severity_adjustment,
                        "reason": modifier.reason
                    })
                    break
        
        # Step 4: Calculate confidence score
        # Confidence is based on how definitively the values fall within the range
        confidence = self._calculate_confidence(pm10, pm25, final_level.thresholds, zone_multiplier)
        
        # Return comprehensive result
        return {
            "level": final_level.level,
            "display_name": final_level.display_name,
            "color": final_level.color,
            "description": final_level.description,
            "priority": final_level.priority,
            "confidence": confidence,
            "applied_modifiers": applied_modifiers,
            "threshold_details": threshold_details,
            "input_values": {
                "pm10": pm10,
                "pm25": pm25,
                "wind_speed": wind_speed,
                "humidity": humidity
            }
        }
    
    def _calculate_confidence(self, pm10: float, pm25: float, 
                            thresholds: SeverityThresholds,
                            zone_multiplier: float) -> float:
        """
        Calculate classification confidence (0-1).
        
        Confidence is higher when values are clearly within the range
        and lower when near boundaries.
        
        Args:
            pm10: PM10 value
            pm25: PM2.5 value
            thresholds: Severity thresholds to check against
            zone_multiplier: Zone sensitivity multiplier
        
        Returns:
            Confidence score between 0 and 1
        """
        confidence_scores = []
        
        # PM10 confidence
        pm10_min = thresholds.pm10_min * zone_multiplier
        pm10_max = thresholds.pm10_max * zone_multiplier if thresholds.pm10_max else float('inf')
        pm10_range = pm10_max - pm10_min
        pm10_center = (pm10_max + pm10_min) / 2
        pm10_distance = abs(pm10 - pm10_center) / (pm10_range / 2) if pm10_range > 0 else 0
        confidence_scores.append(1 - min(1, pm10_distance))
        
        # PM2.5 confidence
        pm25_min = thresholds.pm25_min * zone_multiplier
        pm25_max = thresholds.pm25_max * zone_multiplier if thresholds.pm25_max else float('inf')
        pm25_range = pm25_max - pm25_min
        pm25_center = (pm25_max + pm25_min) / 2
        pm25_distance = abs(pm25 - pm25_center) / (pm25_range / 2) if pm25_range > 0 else 0
        confidence_scores.append(1 - min(1, pm25_distance))
        
        # Return average confidence
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
    
    def export_config(self) -> Dict:
        """
        Export current configuration in JSON-friendly format for React UI.
        
        This format is designed to be:
        1. Easy to serialize/deserialize
        2. Simple for React forms to edit
        3. Version-controllable
        4. Human-readable
        
        Returns:
            Dictionary containing complete configuration
        """
        return {
            "version": "1.0",
            "severity_levels": [
                {
                    "level": lvl.level,
                    "display_name": lvl.display_name,
                    "priority": lvl.priority,
                    "color": lvl.color,
                    "description": lvl.description,
                    "thresholds": {
                        "pm10_min": lvl.thresholds.pm10_min,
                        "pm10_max": lvl.thresholds.pm10_max,
                        "pm25_min": lvl.thresholds.pm25_min,
                        "pm25_max": lvl.thresholds.pm25_max,
                        "wind_min": lvl.thresholds.wind_min,
                        "wind_max": lvl.thresholds.wind_max,
                        "humidity_min": lvl.thresholds.humidity_min,
                        "humidity_max": lvl.thresholds.humidity_max,
                    }
                }
                for lvl in self.severity_levels
            ],
            "zone_sensitivities": {
                zone: {
                    "zone_type": sens.zone_type,
                    "threshold_multiplier": sens.threshold_multiplier,
                    "description": sens.description
                }
                for zone, sens in self.zone_sensitivities.items()
            },
            "time_modifiers": [
                {
                    "start_hour": mod.start_hour,
                    "end_hour": mod.end_hour,
                    "severity_adjustment": mod.severity_adjustment,
                    "reason": mod.reason
                }
                for mod in self.time_modifiers
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'SeverityRuleEngine':
        """
        Load rule engine from JSON configuration.
        
        This allows the React UI to save configurations and reload them.
        
        Args:
            config: Dictionary containing configuration (from export_config)
        
        Returns:
            New SeverityRuleEngine instance with loaded configuration
        """
        # Parse severity levels
        severity_levels = []
        for lvl_config in config.get("severity_levels", []):
            thresholds = SeverityThresholds(**lvl_config["thresholds"])
            level = SeverityLevel(
                level=lvl_config["level"],
                display_name=lvl_config["display_name"],
                priority=lvl_config["priority"],
                color=lvl_config["color"],
                thresholds=thresholds,
                description=lvl_config["description"]
            )
            severity_levels.append(level)
        
        # Parse zone sensitivities
        zone_sensitivities = {}
        for zone, sens_config in config.get("zone_sensitivities", {}).items():
            zone_sensitivities[zone] = ZoneSensitivity(**sens_config)
        
        # Parse time modifiers
        time_modifiers = [
            TimeModifier(**mod_config)
            for mod_config in config.get("time_modifiers", [])
        ]
        
        return cls(severity_levels, zone_sensitivities, time_modifiers)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic classification
    print("=" * 70)
    print("EXAMPLE 1: Basic Air Quality Classification")
    print("=" * 70)
    
    engine = SeverityRuleEngine()
    
    # Test case: Moderate pollution
    result = engine.classify(
        pm10=75,
        pm25=25,
        wind_speed=3.5,
        humidity=60
    )
    
    print(f"\nSeverity: {result['display_name']}")
    print(f"Level: {result['level']}")
    print(f"Color: {result['color']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Description: {result['description']}")
    print(f"\nThreshold Details:")
    for detail in result['threshold_details']:
        print(f"  - {detail}")
    
    # Example 2: Classification with zone sensitivity
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Classification with Hospital Zone Sensitivity")
    print("=" * 70)
    
    result = engine.classify(
        pm10=75,
        pm25=25,
        wind_speed=3.5,
        humidity=60,
        zone_type="hospital"
    )
    
    print(f"\nSeverity: {result['display_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nApplied Modifiers:")
    for modifier in result['applied_modifiers']:
        print(f"  - {modifier['type']}: {modifier['description']}")
    
    # Example 3: Export configuration for React UI
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Export Configuration (JSON for React UI)")
    print("=" * 70)
    
    config = engine.export_config()
    print("\nConfiguration structure:")
    print(f"  - Version: {config['version']}")
    print(f"  - Severity Levels: {len(config['severity_levels'])}")
    print(f"  - Zone Sensitivities: {len(config['zone_sensitivities'])}")
    print(f"  - Time Modifiers: {len(config['time_modifiers'])}")
    
    # Show first severity level as example
    print("\nExample Severity Level (Good):")
    print(json.dumps(config['severity_levels'][0], indent=2))
    
    # Example 4: Load from configuration
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Load Engine from Configuration")
    print("=" * 70)
    
    loaded_engine = SeverityRuleEngine.from_config(config)
    print(f"\nSuccessfully loaded engine with {len(loaded_engine.severity_levels)} severity levels")
    
    # Test the loaded engine
    result = loaded_engine.classify(pm10=150, pm25=65, wind_speed=2, humidity=45)
    print(f"Test classification: {result['display_name']}")
