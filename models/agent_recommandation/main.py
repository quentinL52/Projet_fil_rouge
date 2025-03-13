from crew import JobRecommendationCrew

def main():
    try:
        crew_instance = JobRecommendationCrew()
        result = crew_instance.kickoff(inputs={"candidate_name": "Quentin Loumeau"})
        print("\n=== RECOMMANDATIONS D'EMPLOI ===\n")
        print(result)
        
    except Exception as e:
        print(f"Erreur dans le programme principal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()