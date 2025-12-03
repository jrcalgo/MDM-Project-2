''' READ ME:
For integration into rest of pipeline, build the validator by calling create_validator function to retrieve ValidatorFunction object.
Then, call validate_record method to validate a single record from the upstream ComparisonFunction output of the SearchFunction.
Output is a ValidationResult object that is meant to be used by the downstream CSV/Report generation or the searchFunction refinement operations.

MAX_VALIDATION_ITERATIONS represents the maximum number of validation loops between the ValidatorFunction and the SearchFunction.
VALIDATION_MODEL currently uses gpt-4.1-mini because it is a relatively cheap and accurate model for this purpose (should maybe change later)
'''

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, asdict

from .searchFunction import openai_chat

MAX_VALIDATION_ITERATIONS = 2
VALIDATION_MODEL = "gpt-4.1-mini"

# ============ Dataclasses for LLM Output ============

@dataclass
class IssueDetail:
    """Detailed issue found during validation."""
    severity: str  # "critical", "high", "medium", "low"
    issue_type: str  # "missing_field", "hallucination", "poor_provenance", etc.
    field: str
    description: str
    evidence: str


@dataclass
class ConfidenceAssessment:
    """Assessment of the confidence score."""
    current_score: float
    is_appropriate: bool
    suggested_score: float
    reasoning: str


@dataclass
class ValidationScrutiny:
    """Complete LLM scrutiny response."""
    overall_quality: str  # "high", "medium", "low"
    provenance_quality_score: float  # 0.0-1.0
    issues: List[Dict[str, Any]]
    confidence_assessment: Dict[str, Any]
    recommendations: List[str]
    hallucination_detected: bool
    missing_fields_available: List[str]
    search_adequacy: str  # "sufficient", "insufficient", "poor"


# ============ Output Dataclass ============

@dataclass
class ValidationResult:
    """Output from validation process."""
    status: Literal["pass", "needs_refinement", "fail", "validation_done"]
    confidence_meets_threshold: bool
    threshold_used: float
    actual_confidence: float
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    validation_notes: str
    approved_for_csv: bool
    feedback_for_search: Optional[Dict[str, Any]]
    provenance_quality_score: float
    iteration_count: int


# ============ Helper Functions ============

def is_us_country(country: str) -> bool:
    """Determine if country is United States."""
    if not country:
        return False
    c = country.strip().upper()
    us_variants = {"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}
    return c in us_variants


# ============ ValidatorFunction Class ============

class ValidatorFunction:
    """
    Simple validator for MDM Pipeline.
    
    Validates search results against confidence thresholds and quality criteria.
    Provides feedback for iterative refinement or approval for CSV generation.
    """
    
    # Critical system prompt for LLM scrutiny
    SCRUTINY_SYSTEM_PROMPT = """You are a professional data quality auditor for an MDM system. Your job is to be PRECISE and RIGOROUS.

    You will receive:
    1. Search results from various APIs (Google KG, Wikidata, Tavily, OSM)
    2. Comparison analysis between original and enriched data
    3. The enriched MDM record with confidence score

    Your task: CRITICALLY evaluate the quality and trustworthiness of the results.

    BE WARY OF:
    - Confidence scores that seem inflated given weak evidence
    - Fields populated without strong provenance
    - Weak or irrelevant search results being used
    - LLM hallucinations (data not present in tool results)
    - Missing data that tools likely provided but wasn't extracted
    - Geographic inconsistencies
    - Name/address mismatches being ignored
    - Low evidence counts relative to data completeness
    - Vague or generic data that could apply to multiple companies

    DEMAND:
    - Strong, specific provenance for every populated field
    - Appropriate confidence scores reflecting actual data quality
    - Complete extraction of available data from tools
    - Proper interpretation of tool results
    - Cross-validation between multiple sources

    BE HARSH BUT FAIR:
    - If confidence score is > 0.7 for US OR > 0.8 for Non-US but evidence is weak, flag it as inappropriate
    - If canonical name doesn't closely match input, question the match
    - If address data is incomplete but OSM likely has it, mark as missing_fields_available
    - If data looks fabricated without tool evidence, mark hallucination_detected as true

    Return STRICT JSON matching the ValidationScrutiny schema. Be critical - if results are weak, say so clearly with specific evidence."""

    def __init__(self, openai_api_key: Optional[str] = None, debug: bool = False):
        """
        Initialize validator.
        
        Args:
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            debug: Enable debug logging
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.debug = debug
    
    def _log(self, message: str):
        """Log message if debug enabled."""
        if self.debug:
            print(f"[VALIDATOR] {message}")
    
    def _check_confidence_threshold(
        self,
        search_output: Dict[str, Any]
    ) -> tuple[bool, float, float]:
        """
        Check if confidence meets threshold based on country.
        
        Returns: (meets_threshold, threshold_used, actual_confidence)
        """
        # Extract country from input or enriched data
        country = ""
        if "input" in search_output:
            country = search_output["input"].get("country", "")
        if not country and search_output.get("enriched_mdm"):
            country = search_output["enriched_mdm"].get("country", "")
        
        # Determine threshold: US >= 0.7, Non-US >= 0.8
        is_us = is_us_country(country)
        threshold = 0.7 if is_us else 0.8
        
        actual_confidence = 0.0
        if search_output.get("enriched_mdm"):
            actual_confidence = float(search_output["enriched_mdm"].get("confidence", 0.0) or 0.0)
        
        meets_threshold = actual_confidence >= threshold
        
        self._log(f"Country: {country} | US: {is_us} | Threshold: {threshold} | Confidence: {actual_confidence:.3f} | Meets: {meets_threshold}")
        
        return meets_threshold, threshold, actual_confidence
    
    def _build_scrutiny_prompt(
        self,
        search_output: Dict[str, Any],
        comparison_report: Optional[Dict[str, Any]]
    ) -> str:
        """Build user prompt for LLM scrutiny."""
        context = {
            "task": "Critically evaluate the search and enrichment quality",
            "input_record": search_output.get("input", {}),
            "search_metadata": {
                "success": search_output.get("success"),
                "message": search_output.get("message"),
                "evidence_count": search_output.get("evidence_count"),
                "public_presence": search_output.get("public_presence"),
                "websites": search_output.get("websites", []),
                "notes": search_output.get("notes", "")
            },
            "enriched_mdm": search_output.get("enriched_mdm", {}),
            "comparison_summary": comparison_report.get("summary", {}) if comparison_report else {},
            "instructions": [
                "Assess if confidence score is justified by evidence quality",
                "Identify any hallucinated data (not from tools)",
                "Check if available tool data was fully extracted",
                "Evaluate geographic/name consistency",
                "Check if provenance is strong and specific",
                "Recommend specific improvements for search refinement"
            ],
            "expected_response_schema": {
                "overall_quality": "high|medium|low",
                "provenance_quality_score": "float 0.0-1.0",
                "issues": [
                    {
                        "severity": "critical|high|medium|low",
                        "issue_type": "string",
                        "field": "string",
                        "description": "string",
                        "evidence": "string"
                    }
                ],
                "confidence_assessment": {
                    "current_score": "float",
                    "is_appropriate": "bool",
                    "suggested_score": "float",
                    "reasoning": "string"
                },
                "recommendations": ["list of specific recommendations"],
                "hallucination_detected": "bool",
                "missing_fields_available": ["list of field names"],
                "search_adequacy": "sufficient|insufficient|poor"
            }
        }
        
        return json.dumps(context, ensure_ascii=False, indent=2)
    
    def _llm_scrutinize(
        self,
        search_output: Dict[str, Any],
        comparison_report: Optional[Dict[str, Any]]
    ) -> ValidationScrutiny:
        """
        Use LLM to critically analyze search results.
        
        Returns: ValidationScrutiny with structured analysis
        """
        self._log("Running LLM scrutiny...")
        
        if not search_output.get("success", False):
            self._log("Search not successful, returning minimal scrutiny")
            return ValidationScrutiny(
                overall_quality="low",
                provenance_quality_score=0.0,
                issues=[],
                confidence_assessment={
                    "current_score": 0.0,
                    "is_appropriate": True,
                    "suggested_score": 0.0,
                    "reasoning": "Search was not successful"
                },
                recommendations=["Retry search with alternative query formulations"],
                hallucination_detected=False,
                missing_fields_available=[],
                search_adequacy="poor"
            )
        
        user_prompt = self._build_scrutiny_prompt(search_output, comparison_report)
        
        try:
            response = openai_chat(
                [
                    {"role": "system", "content": self.SCRUTINY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=VALIDATION_MODEL,
                temperature=0.0,
                json_mode=True
            )
            
            response_data = json.loads(response)
            
            scrutiny = ValidationScrutiny(
                overall_quality=response_data.get("overall_quality", "low"),
                provenance_quality_score=float(response_data.get("provenance_quality_score", 0.5)),
                issues=response_data.get("issues", []),
                confidence_assessment=response_data.get("confidence_assessment", {
                    "current_score": 0.0,
                    "is_appropriate": False,
                    "suggested_score": 0.5,
                    "reasoning": "No assessment provided"
                }),
                recommendations=response_data.get("recommendations", []),
                hallucination_detected=response_data.get("hallucination_detected", False),
                missing_fields_available=response_data.get("missing_fields_available", []),
                search_adequacy=response_data.get("search_adequacy", "insufficient")
            )
            
            self._log(f"LLM analysis quality: {scrutiny.overall_quality}")
            self._log(f"LLM found {len(scrutiny.issues)} issues")
            
            return scrutiny
            
        except Exception as e:
            #TODO: Add better fallback logic here
            self._log(f"LLM scrutiny error: {e}")

            return ValidationScrutiny(
                overall_quality="low",
                provenance_quality_score=0.5,
                issues=[{
                    "severity": "medium",
                    "issue_type": "llm_error",
                    "field": "n/a",
                    "description": "LLM analysis failed",
                    "evidence": str(e)
                }],
                confidence_assessment={
                    "current_score": search_output.get("enriched_mdm", {}).get("confidence", 0.0) or 0.0,
                    "is_appropriate": False,
                    "suggested_score": 0.5,
                    "reasoning": "Cannot assess due to LLM error"
                },
                recommendations=["Manual review recommended due to validation error"],
                hallucination_detected=False,
                missing_fields_available=[],
                search_adequacy="insufficient"
            )
    
    def _generate_feedback(
        self,
        search_output: Dict[str, Any],
        scrutiny: ValidationScrutiny,
        confidence_meets_threshold: bool,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Generate structured feedback for search refinement function.
        
        Returns: Feedback dictionary for SearchFunction
        """
        confidence_current = scrutiny.confidence_assessment.get("current_score", 0.0)
        feedback = {
            "refinement_needed": True,
            "confidence_gap": threshold - confidence_current,
            "target_threshold": threshold,
            "issues": scrutiny.issues,
            "recommendations": scrutiny.recommendations,
            "suggested_actions": []
        }
        
        # Adds specific suggested actions based on issues, input missing fields, etc.
        if scrutiny.hallucination_detected:
            feedback["suggested_actions"].append("Re-run search with stricter provenance requirements")
        
        if scrutiny.missing_fields_available:
            feedback["suggested_actions"].append(
                f"Re-extract fields from tool results: {', '.join(scrutiny.missing_fields_available)}"
            )
        
        if scrutiny.search_adequacy in ["insufficient", "poor"]:
            feedback["suggested_actions"].append("Expand search to additional sources")
            feedback["suggested_actions"].append("Try alternative query formulations")
        
        if not confidence_meets_threshold:
            feedback["suggested_actions"].append(
                f"Improve data quality to meet {threshold:.1f} confidence threshold"
            )
        
        input_name = search_output.get("input", {}).get("name", "")
        if input_name and not all(ord(c) < 128 for c in input_name):
            feedback["suggested_actions"].append("Try both original and romanized company names")
        
        enriched = search_output.get("enriched_mdm", {})
        if enriched:
            critical_fields = ["canonical_name", "address", "city", "country"]
            missing = [f for f in critical_fields if not enriched.get(f)]
            if missing:
                feedback["suggested_actions"].append(
                    f"Focus search on missing critical fields: {', '.join(missing)}"
                )
        
        return feedback
    
    def _make_decision(
        self,
        search_output: Dict[str, Any],
        scrutiny: ValidationScrutiny,
        confidence_meets_threshold: bool,
        iteration: int
    ) -> tuple[str, bool, Optional[Dict[str, Any]]]:
        """
        Make final validation decision.
        
        Returns: (status, approved_for_csv, feedback_for_search)
        """
        search_success = search_output.get("success", False)
        
        critical_high_issues = [
            i for i in scrutiny.issues
            if i.get("severity") in ["critical", "high"]
        ]
        
        if iteration >= MAX_VALIDATION_ITERATIONS:
            if search_success and confidence_meets_threshold and len(critical_high_issues) == 0:
                self._log("Max iterations reached but results acceptable")
                return "pass", True, None
            else:
                self._log("Max iterations reached, ending validation")
                return "validation_done", False, None
        
        if not search_success:
            suggested_score = scrutiny.confidence_assessment.get("suggested_score", 0.5)
            feedback = self._generate_feedback(search_output, scrutiny, confidence_meets_threshold, suggested_score)
            return "fail", False, feedback
        
        if critical_high_issues:
            self._log(f"Found {len(critical_high_issues)} critical/high severity issues")
            suggested_score = scrutiny.confidence_assessment.get("suggested_score", 0.5)
            feedback = self._generate_feedback(search_output, scrutiny, confidence_meets_threshold, suggested_score)
            return "needs_refinement", False, feedback
        
        if not confidence_meets_threshold:
            self._log("Confidence below threshold")
            suggested_score = scrutiny.confidence_assessment.get("suggested_score", 0.5)
            feedback = self._generate_feedback(search_output, scrutiny, confidence_meets_threshold, suggested_score)
            return "needs_refinement", False, feedback
        
        if scrutiny.issues:
            if iteration < 2:
                self._log(f"Found {len(scrutiny.issues)} issues, requesting refinement")
                suggested_score = scrutiny.confidence_assessment.get("suggested_score", 0.5)
                feedback = self._generate_feedback(search_output, scrutiny, confidence_meets_threshold, suggested_score)
                return "needs_refinement", False, feedback
            else:
                self._log(f"Accepting with {len(scrutiny.issues)} minor issues after {iteration} iterations")
                return "pass", True, None
        
        self._log("All validation checks passed")
        return "pass", True, None
    
    def validate_record(
        self,
        search_output: Dict[str, Any],
        comparison_report: Optional[Dict[str, Any]] = None,
        iteration: int = 0
    ) -> ValidationResult:

        self._log(f"Starting validation (iteration {iteration})...")
        input_name = search_output.get("input", {}).get("name", "Unknown")
        self._log(f"Input company: {input_name}")
        
        # Step 1: Check confidence threshold
        confidence_meets_threshold, threshold, actual_confidence = self._check_confidence_threshold(search_output)
        
        # Step 2: LLM scrutiny
        scrutiny = self._llm_scrutinize(search_output, comparison_report)
        
        # Step 3: Make decision
        status, approved_for_csv, feedback_for_search = self._make_decision(
            search_output,
            scrutiny,
            confidence_meets_threshold,
            iteration
        )
        
        # Step 4: Build validation notes
        notes_parts = []
        if not search_output.get("success"):
            notes_parts.append(f"Search failed: {search_output.get('message', 'Unknown error')}")
        elif not confidence_meets_threshold:
            notes_parts.append(f"Confidence {actual_confidence:.3f} below threshold {threshold:.1f}")
        
        if scrutiny.issues:
            notes_parts.append(f"Found {len(scrutiny.issues)} issues")
        
        if iteration >= 3:
            notes_parts.append("Max iterations reached")
        
        if approved_for_csv:
            notes_parts.append("Approved for CSV generation")
        
        validation_notes = " | ".join(notes_parts) if notes_parts else "Validation complete"
        
        all_issues = scrutiny.issues
        
        all_recommendations = list(scrutiny.recommendations)
        if feedback_for_search:
            all_recommendations.extend(feedback_for_search.get("suggested_actions", []))
        
        result = ValidationResult(
            status=status,
            confidence_meets_threshold=confidence_meets_threshold,
            threshold_used=threshold,
            actual_confidence=actual_confidence,
            issues_found=all_issues,
            recommendations=list(dict.fromkeys(all_recommendations)),
            validation_notes=validation_notes,
            approved_for_csv=approved_for_csv,
            feedback_for_search=feedback_for_search,
            provenance_quality_score=scrutiny.provenance_quality_score,
            iteration_count=iteration
        )
        
        self._log(f"Validation complete: {status} | Approved: {approved_for_csv}")
        
        return result


# ============ Convenience Functions ============

def create_validator(openai_api_key: Optional[str] = None, debug: bool = False) -> ValidatorFunction:
    """
    Create and return a ValidatorFunction instance.
    
    Args:
        debug: Enable debug logging
    
    Returns:
        ValidatorFunction instance
    """
    return ValidatorFunction(openai_api_key=openai_api_key, debug=debug)


if __name__ == "__main__":
    import sys
    print("Testing ValidatorFunction with sample data...")
    try:
        with open("mdm_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: mdm_results.json not found")
        sys.exit(1)
    
    validator = create_validator(openai_api_key=os.getenv("OPENAI_API_KEY"), debug=True)
    test_indices = [3, 5, 0]  # Success, success, fail
    
    for idx in test_indices:
        if idx >= len(results):
            continue
        
        sample = results[idx]
        print(f"\n{'='*80}")
        print(f"Testing record {idx}: {sample['input']['name']}")
        print(f"{'='*80}")
        
        validation_result = validator.validate_record(
            search_output=sample,
            comparison_report=None,
            iteration=0
        )
        
        print("\nValidation Result:")
        result_dict = asdict(validation_result)
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))
        print(f"\nDecision: {'APPROVE FOR CSV' if validation_result.approved_for_csv else 'NEEDS REFINEMENT'}")
