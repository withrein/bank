from typing import List
from models import CandidateScore, AgentState
from config import Config

class ShortlistingAgent:
    """Agent responsible for shortlisting top candidates for HR review"""
    
    def __init__(self):
        self.max_candidates = Config.MAX_CANDIDATES_TO_SHORTLIST
        self.min_score_threshold = Config.MINIMUM_SCORE_THRESHOLD
    
    def shortlist_candidates(self, candidate_scores: List[CandidateScore]) -> List[CandidateScore]:
        """Select top candidates based on scores and criteria"""
        if not candidate_scores:
            return []
        
        # Filter candidates who meet minimum score threshold
        qualified_candidates = [
            candidate for candidate in candidate_scores 
            if candidate.overall_score >= self.min_score_threshold
        ]
        
        # If no candidates meet threshold, take top candidates anyway (with warning)
        if not qualified_candidates:
            print(f"⚠️  No candidates meet minimum score threshold of {self.min_score_threshold}")
            print("   Taking top candidates regardless of score...")
            qualified_candidates = candidate_scores
        
        # Sort by overall score (should already be sorted from scoring agent)
        qualified_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Take top N candidates
        shortlisted = qualified_candidates[:self.max_candidates]
        
        return shortlisted
    
    def process(self, state: AgentState) -> AgentState:
        """Process candidate shortlisting in the agent state"""
        if not state.candidate_scores:
            state.errors.append("No candidate scores available for shortlisting")
            return state
        
        try:
            print("🎯 Shortlisting Agent: Selecting top candidates...")
            state.current_step = "shortlisting_candidates"
            
            # Shortlist top candidates
            shortlisted_candidates = self.shortlist_candidates(state.candidate_scores)
            state.shortlisted_candidates = shortlisted_candidates
            
            print(f"✅ Shortlisting Agent: Selected {len(shortlisted_candidates)} candidates")
            
            state.current_step = "candidates_shortlisted"
            
        except Exception as e:
            error_msg = f"Shortlisting Agent error: {str(e)}"
            print(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state 