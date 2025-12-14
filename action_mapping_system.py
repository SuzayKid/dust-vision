"""
Action Mapping System - Severity → Actions Layer
=================================================

This module implements a flexible action mapping system that translates
air quality severity levels into concrete mitigation actions.

Architecture:
1. Actions are defined with configurable parameters (intensity, frequency, etc.)
2. Each severity level maps to a bundle of actions
3. Actions can affect governance/compliance scoring
4. All configurations are UI-editable via JSON
5. Runtime execution dispatches actions to appropriate subsystems

Key Concepts:
- Action: A single mitigation measure (e.g., increase sprinklers)
- Action Bundle: Set of actions triggered by a severity level
- Action Dispatcher: Executes actions and tracks compliance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import json


# ============================================================================
# ACTION TYPES AND CATEGORIES
# ============================================================================

class ActionCategory(Enum):
    """
    Categories of actions for organizational purposes.
    Helps UI group related actions together.
    """
    DUST_SUPPRESSION = "dust_suppression"     # Water sprinklers, misting
    WORK_RESTRICTION = "work_restriction"     # Hour limits, activity pauses
    MATERIAL_HANDLING = "material_handling"   # Covering, storage requirements
    TRAFFIC_CONTROL = "traffic_control"       # Speed limits, route restrictions
    MONITORING = "monitoring"                 # Alert escalation, reporting
    COMPLIANCE = "compliance"                 # Scoring, penalties, notifications


class ActionPriority(Enum):
    """Priority levels for action execution order."""
    CRITICAL = 1    # Execute immediately (safety-critical)
    HIGH = 2        # Execute within minutes
    MEDIUM = 3      # Execute within hours
    LOW = 4         # Execute within day


# ============================================================================
# ACTION DEFINITIONS
# ============================================================================

@dataclass
class SprinklerAction:
    """
    Controls water sprinkler system for dust suppression.
    
    Attributes:
        intensity: Water flow intensity (0-100%)
        frequency: Minutes between activation cycles
        duration: Seconds per activation
        zones: List of zones to activate ('all' for all zones)
        auto_adjust: Whether to auto-adjust based on wind/humidity
    """
    intensity: int  # 0-100%
    frequency: int  # minutes between cycles
    duration: int   # seconds per cycle
    zones: List[str] = field(default_factory=lambda: ["all"])
    auto_adjust: bool = True
    
    def validate(self) -> bool:
        """Validate parameters are within acceptable ranges."""
        return (0 <= self.intensity <= 100 and 
                self.frequency > 0 and 
                self.duration > 0)


@dataclass
class WorkRestrictionAction:
    """
    Controls work hour restrictions and activity pauses.
    
    Attributes:
        restriction_type: Type of restriction to apply
        restricted_hours: Hours when work is restricted (24h format)
        allowed_activities: List of activities still allowed
        pause_dust_generating: Whether to pause dust-generating activities
        pause_duration: Minutes to pause (None = until severity drops)
        notify_contractors: Whether to send notifications
    """
    restriction_type: Literal["none", "partial", "full", "emergency"]
    restricted_hours: List[tuple[int, int]] = field(default_factory=list)  # [(start, end), ...]
    allowed_activities: List[str] = field(default_factory=list)
    pause_dust_generating: bool = False
    pause_duration: Optional[int] = None  # minutes
    notify_contractors: bool = True


@dataclass
class MaterialCoverAction:
    """
    Controls material covering and storage requirements.
    
    Attributes:
        require_covering: Whether all stockpiles must be covered
        material_types: Types of materials to cover ('all' or specific types)
        cover_type: Type of cover required
        inspection_frequency: Hours between cover inspections
        penalty_per_violation: Compliance score penalty per uncovered pile
    """
    require_covering: bool
    material_types: List[str] = field(default_factory=lambda: ["all"])
    cover_type: Literal["tarp", "netting", "chemical_binding", "none"] = "tarp"
    inspection_frequency: int = 24  # hours
    penalty_per_violation: float = 5.0  # compliance score points


@dataclass
class TrafficControlAction:
    """
    Controls vehicle speed and traffic patterns at site.
    
    Attributes:
        speed_limit: Maximum speed in km/h
        enforce_at_gates: Enforce at entry/exit gates
        restricted_routes: Routes to restrict or close
        require_wheel_wash: Require vehicle wheel washing
        penalty_per_violation: Compliance score penalty per violation
    """
    speed_limit: int  # km/h
    enforce_at_gates: bool = True
    restricted_routes: List[str] = field(default_factory=list)
    require_wheel_wash: bool = False
    penalty_per_violation: float = 3.0  # compliance score points


@dataclass
class MonitoringAction:
    """
    Controls monitoring intensity and alert escalation.
    
    Attributes:
        alert_level: Current alert status
        escalate_to: List of stakeholders to notify
        monitoring_interval: Minutes between sensor readings
        require_manual_checks: Require physical site inspections
        log_all_activities: Enable detailed activity logging
    """
    alert_level: Literal["normal", "advisory", "warning", "critical", "emergency"]
    escalate_to: List[str] = field(default_factory=list)  # email/phone list
    monitoring_interval: int = 15  # minutes
    require_manual_checks: bool = False
    log_all_activities: bool = False


@dataclass
class ComplianceAction:
    """
    Controls compliance scoring and contractor evaluation.
    
    Attributes:
        score_impact: Base score impact for this severity level
        impact_rate: Points deducted per time period (hour/day)
        time_unit: Unit for impact_rate calculation
        notify_contractor: Send compliance notification
        require_acknowledgment: Require contractor to acknowledge
        escalation_threshold: Score threshold that triggers escalation
        affects_payment: Whether this impacts contractor payment
    """
    score_impact: float  # base points deducted
    impact_rate: float  # points per time_unit
    time_unit: Literal["minute", "hour", "day"] = "hour"
    notify_contractor: bool = True
    require_acknowledgment: bool = False
    escalation_threshold: float = 70.0  # score below this triggers escalation
    affects_payment: bool = False


# ============================================================================
# ACTION BUNDLE
# ============================================================================

@dataclass
class ActionBundle:
    """
    Complete set of actions triggered by a severity level.
    
    Each severity level maps to one ActionBundle containing all
    the actions that should be executed when that severity is reached.
    
    Attributes:
        severity_level: Severity level this bundle corresponds to
        priority: Execution priority
        sprinkler: Sprinkler system configuration
        work_restriction: Work restriction rules
        material_cover: Material covering requirements
        traffic_control: Traffic control measures
        monitoring: Monitoring and alerting settings
        compliance: Compliance scoring rules
        custom_actions: Additional custom actions (for extensibility)
        description: Human-readable description of this bundle
    """
    severity_level: str
    priority: ActionPriority
    sprinkler: Optional[SprinklerAction] = None
    work_restriction: Optional[WorkRestrictionAction] = None
    material_cover: Optional[MaterialCoverAction] = None
    traffic_control: Optional[TrafficControlAction] = None
    monitoring: Optional[MonitoringAction] = None
    compliance: Optional[ComplianceAction] = None
    custom_actions: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def get_governance_impacting_actions(self) -> List[str]:
        """
        Returns list of actions that affect compliance scoring.
        
        These actions directly impact contractor evaluation and
        should be tracked carefully for governance reporting.
        """
        impacting = []
        
        if self.compliance:
            impacting.append("compliance")
        
        if self.work_restriction and self.work_restriction.restriction_type != "none":
            impacting.append("work_restriction")
        
        if self.material_cover and self.material_cover.require_covering:
            impacting.append("material_cover")
        
        if self.traffic_control and self.traffic_control.penalty_per_violation > 0:
            impacting.append("traffic_control")
        
        return impacting


# ============================================================================
# DEFAULT ACTION MAPPINGS
# ============================================================================

# Predefined action bundles for each severity level
# These represent best practices but can be fully customized via UI

DEFAULT_ACTION_MAPPINGS = {
    "good": ActionBundle(
        severity_level="good",
        priority=ActionPriority.LOW,
        description="Normal operations - minimal dust suppression",
        sprinkler=SprinklerAction(
            intensity=30,
            frequency=120,  # every 2 hours
            duration=180,   # 3 minutes
            auto_adjust=True
        ),
        work_restriction=WorkRestrictionAction(
            restriction_type="none",
            notify_contractors=False
        ),
        material_cover=MaterialCoverAction(
            require_covering=False,
            inspection_frequency=48
        ),
        traffic_control=TrafficControlAction(
            speed_limit=40,
            require_wheel_wash=False
        ),
        monitoring=MonitoringAction(
            alert_level="normal",
            monitoring_interval=30
        ),
        compliance=ComplianceAction(
            score_impact=0,
            impact_rate=0,
            notify_contractor=False
        )
    ),
    
    "moderate": ActionBundle(
        severity_level="moderate",
        priority=ActionPriority.MEDIUM,
        description="Increased dust suppression - enhanced monitoring",
        sprinkler=SprinklerAction(
            intensity=50,
            frequency=60,   # every hour
            duration=300,   # 5 minutes
            auto_adjust=True
        ),
        work_restriction=WorkRestrictionAction(
            restriction_type="none",
            notify_contractors=True
        ),
        material_cover=MaterialCoverAction(
            require_covering=False,
            material_types=["sand", "fine_aggregates"],
            inspection_frequency=24
        ),
        traffic_control=TrafficControlAction(
            speed_limit=30,
            require_wheel_wash=True,
            penalty_per_violation=2.0
        ),
        monitoring=MonitoringAction(
            alert_level="advisory",
            monitoring_interval=15,
            escalate_to=["site_supervisor"]
        ),
        compliance=ComplianceAction(
            score_impact=-2.0,
            impact_rate=0.5,
            time_unit="hour",
            notify_contractor=True
        )
    ),
    
    "unhealthy_sensitive": ActionBundle(
        severity_level="unhealthy_sensitive",
        priority=ActionPriority.HIGH,
        description="Significant restrictions - cover materials, limit dust activities",
        sprinkler=SprinklerAction(
            intensity=70,
            frequency=30,   # every 30 minutes
            duration=420,   # 7 minutes
            auto_adjust=True
        ),
        work_restriction=WorkRestrictionAction(
            restriction_type="partial",
            pause_dust_generating=True,
            pause_duration=60,  # 1 hour
            allowed_activities=["office_work", "maintenance", "inspection"],
            notify_contractors=True
        ),
        material_cover=MaterialCoverAction(
            require_covering=True,
            material_types=["all"],
            cover_type="tarp",
            inspection_frequency=12,
            penalty_per_violation=8.0
        ),
        traffic_control=TrafficControlAction(
            speed_limit=20,
            enforce_at_gates=True,
            require_wheel_wash=True,
            penalty_per_violation=5.0
        ),
        monitoring=MonitoringAction(
            alert_level="warning",
            monitoring_interval=10,
            escalate_to=["site_supervisor", "project_manager"],
            require_manual_checks=True
        ),
        compliance=ComplianceAction(
            score_impact=-5.0,
            impact_rate=1.5,
            time_unit="hour",
            notify_contractor=True,
            require_acknowledgment=True,
            affects_payment=False
        )
    ),
    
    "unhealthy": ActionBundle(
        severity_level="unhealthy",
        priority=ActionPriority.HIGH,
        description="Severe restrictions - pause most activities",
        sprinkler=SprinklerAction(
            intensity=90,
            frequency=20,   # every 20 minutes
            duration=600,   # 10 minutes
            zones=["all"],
            auto_adjust=False  # max intensity
        ),
        work_restriction=WorkRestrictionAction(
            restriction_type="full",
            pause_dust_generating=True,
            pause_duration=None,  # until severity drops
            allowed_activities=["emergency", "safety"],
            notify_contractors=True
        ),
        material_cover=MaterialCoverAction(
            require_covering=True,
            material_types=["all"],
            cover_type="tarp",
            inspection_frequency=6,
            penalty_per_violation=15.0
        ),
        traffic_control=TrafficControlAction(
            speed_limit=15,
            enforce_at_gates=True,
            restricted_routes=["main_haul_road", "secondary_access"],
            require_wheel_wash=True,
            penalty_per_violation=10.0
        ),
        monitoring=MonitoringAction(
            alert_level="critical",
            monitoring_interval=5,
            escalate_to=["site_supervisor", "project_manager", "environmental_officer"],
            require_manual_checks=True,
            log_all_activities=True
        ),
        compliance=ComplianceAction(
            score_impact=-10.0,
            impact_rate=3.0,
            time_unit="hour",
            notify_contractor=True,
            require_acknowledgment=True,
            escalation_threshold=75.0,
            affects_payment=True
        )
    ),
    
    "very_unhealthy": ActionBundle(
        severity_level="very_unhealthy",
        priority=ActionPriority.CRITICAL,
        description="Emergency measures - site lockdown, maximum mitigation",
        sprinkler=SprinklerAction(
            intensity=100,
            frequency=15,   # every 15 minutes
            duration=900,   # 15 minutes
            zones=["all"],
            auto_adjust=False
        ),
        work_restriction=WorkRestrictionAction(
            restriction_type="emergency",
            pause_dust_generating=True,
            pause_duration=None,
            allowed_activities=["emergency"],
            notify_contractors=True
        ),
        material_cover=MaterialCoverAction(
            require_covering=True,
            material_types=["all"],
            cover_type="tarp",
            inspection_frequency=4,
            penalty_per_violation=25.0
        ),
        traffic_control=TrafficControlAction(
            speed_limit=10,
            enforce_at_gates=True,
            restricted_routes=["all_non_emergency"],
            require_wheel_wash=True,
            penalty_per_violation=20.0
        ),
        monitoring=MonitoringAction(
            alert_level="emergency",
            monitoring_interval=5,
            escalate_to=["site_supervisor", "project_manager", "environmental_officer", "executive_team"],
            require_manual_checks=True,
            log_all_activities=True
        ),
        compliance=ComplianceAction(
            score_impact=-20.0,
            impact_rate=5.0,
            time_unit="hour",
            notify_contractor=True,
            require_acknowledgment=True,
            escalation_threshold=80.0,
            affects_payment=True
        )
    ),
    
    "hazardous": ActionBundle(
        severity_level="hazardous",
        priority=ActionPriority.CRITICAL,
        description="Complete shutdown - regulatory notification required",
        sprinkler=SprinklerAction(
            intensity=100,
            frequency=10,   # every 10 minutes
            duration=1200,  # 20 minutes
            zones=["all"],
            auto_adjust=False
        ),
        work_restriction=WorkRestrictionAction(
            restriction_type="emergency",
            pause_dust_generating=True,
            pause_duration=None,
            allowed_activities=[],  # nothing allowed
            notify_contractors=True
        ),
        material_cover=MaterialCoverAction(
            require_covering=True,
            material_types=["all"],
            cover_type="tarp",
            inspection_frequency=2,
            penalty_per_violation=50.0
        ),
        traffic_control=TrafficControlAction(
            speed_limit=5,
            enforce_at_gates=True,
            restricted_routes=["all"],
            require_wheel_wash=True,
            penalty_per_violation=50.0
        ),
        monitoring=MonitoringAction(
            alert_level="emergency",
            monitoring_interval=5,
            escalate_to=["site_supervisor", "project_manager", "environmental_officer", 
                        "executive_team", "regulatory_authority"],
            require_manual_checks=True,
            log_all_activities=True
        ),
        compliance=ComplianceAction(
            score_impact=-50.0,
            impact_rate=10.0,
            time_unit="hour",
            notify_contractor=True,
            require_acknowledgment=True,
            escalation_threshold=90.0,
            affects_payment=True
        )
    )
}


# ============================================================================
# ACTION DISPATCHER
# ============================================================================

class ActionDispatcher:
    """
    Executes action bundles and manages action lifecycle.
    
    The dispatcher:
    1. Receives severity classifications
    2. Looks up corresponding action bundle
    3. Dispatches actions to appropriate subsystems
    4. Tracks action execution status
    5. Calculates compliance impact
    6. Generates audit logs
    
    In a production system, this would integrate with:
    - IoT control systems (sprinklers, gates)
    - Notification services (SMS, email)
    - Workforce management systems
    - Compliance tracking databases
    """
    
    def __init__(self, action_mappings: Optional[Dict[str, ActionBundle]] = None):
        """
        Initialize dispatcher with action mappings.
        
        Args:
            action_mappings: Dictionary mapping severity levels to action bundles
        """
        self.action_mappings = action_mappings or DEFAULT_ACTION_MAPPINGS
        self.execution_history: List[Dict] = []
        self.active_actions: Dict[str, ActionBundle] = {}
        
        # Callback hooks for integrating with external systems
        # In production, these would connect to real control systems
        self.action_callbacks: Dict[str, Callable] = {}
    
    def dispatch(self, severity_level: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch actions for given severity level.
        
        This is the main entry point for action execution. It:
        1. Retrieves the action bundle for the severity
        2. Validates actions can be executed
        3. Executes each action in priority order
        4. Records execution for audit trail
        5. Returns execution summary
        
        Args:
            severity_level: Severity level to dispatch actions for
            context: Additional context (timestamp, location, etc.)
        
        Returns:
            Dictionary containing:
                - success: Whether dispatch succeeded
                - actions_executed: List of executed actions
                - errors: Any errors encountered
                - compliance_impact: Impact on compliance score
                - execution_id: Unique ID for this dispatch
        """
        if severity_level not in self.action_mappings:
            return {
                "success": False,
                "error": f"No action mapping found for severity: {severity_level}",
                "actions_executed": [],
                "compliance_impact": 0
            }
        
        bundle = self.action_mappings[severity_level]
        timestamp = context.get("timestamp", datetime.now())
        location = context.get("location", "unknown")
        
        # Generate unique execution ID
        execution_id = f"{severity_level}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Track execution results
        results = {
            "success": True,
            "execution_id": execution_id,
            "severity_level": severity_level,
            "timestamp": timestamp.isoformat(),
            "location": location,
            "actions_executed": [],
            "errors": [],
            "compliance_impact": 0,
            "governance_actions": bundle.get_governance_impacting_actions()
        }
        
        # Execute each action type
        if bundle.sprinkler:
            result = self._execute_sprinkler(bundle.sprinkler, context)
            results["actions_executed"].append(result)
            if not result["success"]:
                results["errors"].append(result.get("error"))
        
        if bundle.work_restriction:
            result = self._execute_work_restriction(bundle.work_restriction, context)
            results["actions_executed"].append(result)
            if not result["success"]:
                results["errors"].append(result.get("error"))
        
        if bundle.material_cover:
            result = self._execute_material_cover(bundle.material_cover, context)
            results["actions_executed"].append(result)
            if not result["success"]:
                results["errors"].append(result.get("error"))
        
        if bundle.traffic_control:
            result = self._execute_traffic_control(bundle.traffic_control, context)
            results["actions_executed"].append(result)
            if not result["success"]:
                results["errors"].append(result.get("error"))
        
        if bundle.monitoring:
            result = self._execute_monitoring(bundle.monitoring, context)
            results["actions_executed"].append(result)
            if not result["success"]:
                results["errors"].append(result.get("error"))
        
        if bundle.compliance:
            result = self._execute_compliance(bundle.compliance, context)
            results["actions_executed"].append(result)
            results["compliance_impact"] = result.get("score_impact", 0)
            if not result["success"]:
                results["errors"].append(result.get("error"))
        
        # Execute custom actions
        for action_name, action_config in bundle.custom_actions.items():
            result = self._execute_custom_action(action_name, action_config, context)
            results["actions_executed"].append(result)
            if not result["success"]:
                results["errors"].append(result.get("error"))
        
        # Mark as failed if any errors
        if results["errors"]:
            results["success"] = False
        
        # Store in execution history
        self.execution_history.append(results)
        
        # Track active actions
        self.active_actions[severity_level] = bundle
        
        return results
    
    def _execute_sprinkler(self, action: SprinklerAction, context: Dict) -> Dict:
        """Execute sprinkler control action."""
        if not action.validate():
            return {
                "action_type": "sprinkler",
                "success": False,
                "error": "Invalid sprinkler parameters"
            }
        
        # In production: Call IoT control system API
        # For now, simulate execution
        result = {
            "action_type": "sprinkler",
            "success": True,
            "details": {
                "intensity": action.intensity,
                "frequency": action.frequency,
                "duration": action.duration,
                "zones": action.zones,
                "auto_adjust": action.auto_adjust
            },
            "message": f"Sprinkler system configured: {action.intensity}% intensity, "
                      f"every {action.frequency} minutes for {action.duration} seconds"
        }
        
        # Call registered callback if exists
        if "sprinkler" in self.action_callbacks:
            self.action_callbacks["sprinkler"](action, context)
        
        return result
    
    def _execute_work_restriction(self, action: WorkRestrictionAction, context: Dict) -> Dict:
        """Execute work restriction action."""
        result = {
            "action_type": "work_restriction",
            "success": True,
            "details": {
                "restriction_type": action.restriction_type,
                "pause_dust_generating": action.pause_dust_generating,
                "pause_duration": action.pause_duration,
                "allowed_activities": action.allowed_activities
            },
            "message": f"Work restriction applied: {action.restriction_type}"
        }
        
        if action.notify_contractors:
            result["notification_sent"] = True
            # In production: Send actual notifications via SMS/email service
        
        if "work_restriction" in self.action_callbacks:
            self.action_callbacks["work_restriction"](action, context)
        
        return result
    
    def _execute_material_cover(self, action: MaterialCoverAction, context: Dict) -> Dict:
        """Execute material covering requirement action."""
        result = {
            "action_type": "material_cover",
            "success": True,
            "details": {
                "require_covering": action.require_covering,
                "material_types": action.material_types,
                "cover_type": action.cover_type,
                "inspection_frequency": action.inspection_frequency,
                "penalty_per_violation": action.penalty_per_violation
            },
            "message": f"Material cover requirement: {action.cover_type if action.require_covering else 'not required'}"
        }
        
        if action.require_covering:
            result["inspection_scheduled"] = True
            # In production: Schedule inspection tasks
        
        if "material_cover" in self.action_callbacks:
            self.action_callbacks["material_cover"](action, context)
        
        return result
    
    def _execute_traffic_control(self, action: TrafficControlAction, context: Dict) -> Dict:
        """Execute traffic control action."""
        result = {
            "action_type": "traffic_control",
            "success": True,
            "details": {
                "speed_limit": action.speed_limit,
                "enforce_at_gates": action.enforce_at_gates,
                "restricted_routes": action.restricted_routes,
                "require_wheel_wash": action.require_wheel_wash,
                "penalty_per_violation": action.penalty_per_violation
            },
            "message": f"Speed limit set to {action.speed_limit} km/h"
        }
        
        if action.enforce_at_gates:
            # In production: Update gate control systems
            result["gate_controls_updated"] = True
        
        if "traffic_control" in self.action_callbacks:
            self.action_callbacks["traffic_control"](action, context)
        
        return result
    
    def _execute_monitoring(self, action: MonitoringAction, context: Dict) -> Dict:
        """Execute monitoring and alerting action."""
        result = {
            "action_type": "monitoring",
            "success": True,
            "details": {
                "alert_level": action.alert_level,
                "escalate_to": action.escalate_to,
                "monitoring_interval": action.monitoring_interval,
                "require_manual_checks": action.require_manual_checks,
                "log_all_activities": action.log_all_activities
            },
            "message": f"Alert level: {action.alert_level}"
        }
        
        if action.escalate_to:
            result["escalation_sent"] = True
            result["escalated_to"] = action.escalate_to
            # In production: Send escalation notifications
        
        if "monitoring" in self.action_callbacks:
            self.action_callbacks["monitoring"](action, context)
        
        return result
    
    def _execute_compliance(self, action: ComplianceAction, context: Dict) -> Dict:
        """Execute compliance scoring action."""
        # Calculate time-based impact
        duration_hours = context.get("duration_hours", 1)
        time_multiplier = 1.0
        if action.time_unit == "minute":
            time_multiplier = duration_hours * 60
        elif action.time_unit == "day":
            time_multiplier = duration_hours / 24
        
        total_impact = action.score_impact + (action.impact_rate * time_multiplier)
        
        result = {
            "action_type": "compliance",
            "success": True,
            "details": {
                "base_impact": action.score_impact,
                "rate_impact": action.impact_rate,
                "time_unit": action.time_unit,
                "duration_hours": duration_hours,
                "notify_contractor": action.notify_contractor,
                "require_acknowledgment": action.require_acknowledgment,
                "affects_payment": action.affects_payment
            },
            "score_impact": total_impact,
            "message": f"Compliance score impact: {total_impact:.2f} points"
        }
        
        if action.notify_contractor:
            result["contractor_notified"] = True
            # In production: Send notification to contractor
        
        if "compliance" in self.action_callbacks:
            self.action_callbacks["compliance"](action, context)
        
        return result
    
    def _execute_custom_action(self, action_name: str, config: Any, context: Dict) -> Dict:
        """Execute custom action (for extensibility)."""
        result = {
            "action_type": f"custom_{action_name}",
            "success": True,
            "details": config,
            "message": f"Custom action '{action_name}' executed"
        }
        
        if action_name in self.action_callbacks:
            self.action_callbacks[action_name](config, context)
        
        return result
    
    def register_callback(self, action_type: str, callback: Callable):
        """
        Register callback function for action type.
        
        This allows integration with external systems:
        - IoT control systems
        - Notification services
        - Database updates
        - Third-party APIs
        
        Args:
            action_type: Type of action (e.g., 'sprinkler', 'compliance')
            callback: Function to call when action is executed
        """
        self.action_callbacks[action_type] = callback
    
    def get_execution_history(self, 
                            severity_level: Optional[str] = None,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Retrieve execution history with optional filtering.
        
        Args:
            severity_level: Filter by severity level
            start_time: Filter by start time
            end_time: Filter by end time
        
        Returns:
            List of execution records
        """
        history = self.execution_history
        
        if severity_level:
            history = [h for h in history if h.get("severity_level") == severity_level]
        
        if start_time:
            history = [h for h in history 
                      if datetime.fromisoformat(h["timestamp"]) >= start_time]
        
        if end_time:
            history = [h for h in history 
                      if datetime.fromisoformat(h["timestamp"]) <= end_time]
        
        return history
    
    def calculate_compliance_impact(self, 
                                   start_time: datetime,
                                   end_time: datetime) -> Dict[str, float]:
        """
        Calculate total compliance impact over time period.
        
        Args:
            start_time: Period start
            end_time: Period end
        
        Returns:
            Dictionary with compliance metrics by severity level
        """
        history = self.get_execution_history(start_time=start_time, end_time=end_time)
        
        impact_by_severity = {}
        total_impact = 0
        
        for record in history:
            severity = record["severity_level"]
            impact = record.get("compliance_impact", 0)
            
            if severity not in impact_by_severity:
                impact_by_severity[severity] = 0
            
            impact_by_severity[severity] += impact
            total_impact += impact
        
        return {
            "total_impact": total_impact,
            "by_severity": impact_by_severity,
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat()
        }
    
    def export_config(self) -> Dict:
        """
        Export action mappings in JSON format for React UI.
        
        Returns complete configuration that can be edited in UI
        and reloaded into the system.
        """
        def serialize_action(action):
            """Helper to serialize action objects."""
            if action is None:
                return None
            
            if isinstance(action, (SprinklerAction, WorkRestrictionAction, 
                                 MaterialCoverAction, TrafficControlAction,
                                 MonitoringAction, ComplianceAction)):
                return {k: v for k, v in action.__dict__.items()}
            return action
        
        config = {
            "version": "1.0",
            "action_mappings": {}
        }
        
        for severity_level, bundle in self.action_mappings.items():
            config["action_mappings"][severity_level] = {
                "severity_level": bundle.severity_level,
                "priority": bundle.priority.value,
                "description": bundle.description,
                "sprinkler": serialize_action(bundle.sprinkler),
                "work_restriction": serialize_action(bundle.work_restriction),
                "material_cover": serialize_action(bundle.material_cover),
                "traffic_control": serialize_action(bundle.traffic_control),
                "monitoring": serialize_action(bundle.monitoring),
                "compliance": serialize_action(bundle.compliance),
                "custom_actions": bundle.custom_actions,
                "governance_actions": bundle.get_governance_impacting_actions()
            }
        
        return config
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ActionDispatcher':
        """
        Load dispatcher from JSON configuration.
        
        Allows UI to save and reload custom action mappings.
        
        Args:
            config: Configuration dictionary from export_config()
        
        Returns:
            New ActionDispatcher with loaded configuration
        """
        def deserialize_action(action_type, data):
            """Helper to deserialize action objects."""
            if data is None:
                return None
            
            action_classes = {
                "sprinkler": SprinklerAction,
                "work_restriction": WorkRestrictionAction,
                "material_cover": MaterialCoverAction,
                "traffic_control": TrafficControlAction,
                "monitoring": MonitoringAction,
                "compliance": ComplianceAction
            }
            
            if action_type in action_classes:
                return action_classes[action_type](**data)
            return data
        
        action_mappings = {}
        
        for severity_level, bundle_data in config.get("action_mappings", {}).items():
            bundle = ActionBundle(
                severity_level=bundle_data["severity_level"],
                priority=ActionPriority(bundle_data["priority"]),
                description=bundle_data.get("description", ""),
                sprinkler=deserialize_action("sprinkler", bundle_data.get("sprinkler")),
                work_restriction=deserialize_action("work_restriction", bundle_data.get("work_restriction")),
                material_cover=deserialize_action("material_cover", bundle_data.get("material_cover")),
                traffic_control=deserialize_action("traffic_control", bundle_data.get("traffic_control")),
                monitoring=deserialize_action("monitoring", bundle_data.get("monitoring")),
                compliance=deserialize_action("compliance", bundle_data.get("compliance")),
                custom_actions=bundle_data.get("custom_actions", {})
            )
            action_mappings[severity_level] = bundle
        
        return cls(action_mappings)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ACTION MAPPING SYSTEM - DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Create dispatcher and dispatch actions
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Dispatching Actions for Moderate Severity")
    print("=" * 80)
    
    dispatcher = ActionDispatcher()
    
    # Dispatch actions for moderate severity
    result = dispatcher.dispatch(
        severity_level="moderate",
        context={
            "timestamp": datetime.now(),
            "location": "Construction Site A - Zone 3",
            "duration_hours": 2.5
        }
    )
    
    print(f"\nDispatch Status: {'✓ Success' if result['success'] else '✗ Failed'}")
    print(f"Execution ID: {result['execution_id']}")
    print(f"Compliance Impact: {result['compliance_impact']:.2f} points")
    print(f"Governance Actions: {', '.join(result['governance_actions'])}")
    
    print("\nActions Executed:")
    for action in result["actions_executed"]:
        print(f"\n  • {action['action_type'].upper()}")
        print(f"    Status: {'✓' if action['success'] else '✗'}")
        print(f"    {action['message']}")
    
    # Example 2: Dispatch critical severity
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Dispatching Actions for Hazardous Severity")
    print("=" * 80)
    
    result = dispatcher.dispatch(
        severity_level="hazardous",
        context={
            "timestamp": datetime.now(),
            "location": "Construction Site A - All Zones",
            "duration_hours": 0.5
        }
    )
    
    print(f"\nDispatch Status: {'✓ Success' if result['success'] else '✗ Failed'}")
    print(f"Compliance Impact: {result['compliance_impact']:.2f} points")
    print(f"Governance Actions: {', '.join(result['governance_actions'])}")
    
    # Show sprinkler action details
    sprinkler_action = next(a for a in result["actions_executed"] if a["action_type"] == "sprinkler")
    print(f"\nSprinkler Configuration:")
    print(f"  Intensity: {sprinkler_action['details']['intensity']}%")
    print(f"  Frequency: Every {sprinkler_action['details']['frequency']} minutes")
    print(f"  Duration: {sprinkler_action['details']['duration']} seconds")
    
    # Example 3: Register callback for external integration
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Registering Callbacks for External Systems")
    print("=" * 80)
    
    def sprinkler_callback(action: SprinklerAction, context: Dict):
        """Simulate IoT system integration."""
        print(f"\n  [IoT System] Configuring sprinklers:")
        print(f"    Zone: {', '.join(action.zones)}")
        print(f"    Intensity: {action.intensity}%")
        print(f"    Auto-adjust: {'Enabled' if action.auto_adjust else 'Disabled'}")
    
    def compliance_callback(action: ComplianceAction, context: Dict):
        """Simulate compliance database update."""
        print(f"\n  [Compliance DB] Recording violation:")
        print(f"    Impact: {action.score_impact} base points")
        print(f"    Rate: {action.impact_rate} per {action.time_unit}")
        print(f"    Affects Payment: {'Yes' if action.affects_payment else 'No'}")
    
    dispatcher.register_callback("sprinkler", sprinkler_callback)
    dispatcher.register_callback("compliance", compliance_callback)
    
    print("\nCallbacks registered. Dispatching 'unhealthy' severity...")
    result = dispatcher.dispatch(
        severity_level="unhealthy",
        context={
            "timestamp": datetime.now(),
            "location": "Construction Site A - Zone 1",
            "duration_hours": 1.0
        }
    )
    
    # Example 4: Export configuration for UI
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Export Configuration for React UI")
    print("=" * 80)
    
    config = dispatcher.export_config()
    print(f"\nConfiguration Version: {config['version']}")
    print(f"Severity Levels Configured: {len(config['action_mappings'])}")
    
    # Show moderate severity configuration
    moderate_config = config['action_mappings']['moderate']
    print(f"\nModerate Severity Configuration:")
    print(f"  Priority: {moderate_config['priority']}")
    print(f"  Description: {moderate_config['description']}")
    print(f"  Governance Actions: {', '.join(moderate_config['governance_actions'])}")
    
    if moderate_config['sprinkler']:
        print(f"\n  Sprinkler Settings:")
        print(f"    Intensity: {moderate_config['sprinkler']['intensity']}%")
        print(f"    Frequency: {moderate_config['sprinkler']['frequency']} minutes")
    
    if moderate_config['compliance']:
        print(f"\n  Compliance Settings:")
        print(f"    Base Impact: {moderate_config['compliance']['score_impact']}")
        print(f"    Rate: {moderate_config['compliance']['impact_rate']}/{moderate_config['compliance']['time_unit']}")
    
    # Example 5: Calculate compliance impact over period
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Calculate Compliance Impact")
    print("=" * 80)
    
    start = datetime.now() - timedelta(hours=24)
    end = datetime.now()
    
    impact = dispatcher.calculate_compliance_impact(start, end)
    
    print(f"\nCompliance Impact (Last 24 Hours):")
    print(f"  Total Impact: {impact['total_impact']:.2f} points")
    print(f"  By Severity Level:")
    for severity, points in impact['by_severity'].items():
        print(f"    {severity}: {points:.2f} points")
    
    # Example 6: Show JSON structure for one action bundle
    print("\n" + "=" * 80)
    print("EXAMPLE 6: JSON Structure for UI (Moderate Severity)")
    print("=" * 80)
    
    print("\n" + json.dumps(moderate_config, indent=2))
    
    print("\n" + "=" * 80)
    print("DOCUMENTATION SUMMARY")
    print("=" * 80)
    
    print("""
Action Mapping System Overview:

1. ACTION TYPES:
   - Sprinkler: Water-based dust suppression
   - Work Restriction: Limit or pause activities
   - Material Cover: Covering requirements for stockpiles
   - Traffic Control: Speed limits and route restrictions
   - Monitoring: Alert escalation and reporting
   - Compliance: Scoring and contractor evaluation

2. SEVERITY → ACTION MAPPING:
   Each severity level triggers a complete action bundle with
   specific parameters for all action types.

3. UI EDITABILITY:
   All parameters are exposed in JSON format:
   - Sprinkler intensity/frequency/duration
   - Work restriction types and allowed activities
   - Material cover requirements
   - Speed limits and penalties
   - Alert escalation lists
   - Compliance scoring rules

4. RUNTIME INTERPRETATION:
   - ActionDispatcher receives severity classification
   - Looks up corresponding ActionBundle
   - Executes each action in priority order
   - Calls registered callbacks for external integration
   - Records execution in audit trail
   - Calculates compliance impact

5. GOVERNANCE IMPACT:
   Actions marked as governance-affecting:
   - Compliance actions (scoring)
   - Work restrictions (violations)
   - Material cover violations
   - Traffic violations with penalties
   
   These actions are tracked for contractor evaluation
   and payment adjustments.

6. EXTENSIBILITY:
   - Custom actions can be added via custom_actions dict
   - Callbacks allow integration with any external system
   - Configuration is version-controlled for migrations
    """)