from itertools import permutations
scores_repartition_by_players = {
    6: [1,2,3,4,5,7],
    7: [1,2,3,4,5,7,9],
    8: [1,2,3,4,5,6,8,10],
    9: [1,2,3,4,5,6,7,9,11],
    10: [1,2,3,4,5,6,7,8,10,12],
    11: [1,2,3,4,5,6,7,8,9,11,13],
    12: [1,2,3,4,5,6,7,8,9,10,12,15]
}


def error_handling_scores(players, mogi_context=None, nb_races=None):
    """Handles errors in player and score data, by ensuring the scores are valid with previous ones.

    Args:
        players (Dict[str, int]): A dictionary of player names and their scores.
        mogi_context (Optional[Dict[str, int]]): Contextual information of the mogi. (every player, and the score they have so far)
        nb_races (Optional[int]): The number of races that happened in the mogi so far.
    """
    if not mogi_context or not nb_races:
        # !TODO: Add rectification logic (A score can't lower than the max possible score)
        return players
    
    score_repartition = scores_repartition_by_players.get(len(players), [])
    if not score_repartition:
        return players
    
    # Step 1: Identify correct and incorrect players
    correct_players = {}
    incorrect_players = {}
    
    for player, detected_score in players.items():
        context_score = mogi_context.get(player, 0)
        player_found_correct = False
        
        # Check if any valid race score + context equals detected score
        for race_score in score_repartition:
            if context_score + race_score == detected_score:
                correct_players[player] = race_score
                player_found_correct = True
                break
        
        if not player_found_correct:
            incorrect_players[player] = detected_score
    
    # Step 2: Find unused scores
    used_scores = set(correct_players.values())
    unused_scores = [score for score in score_repartition if score not in used_scores]
    
    # Step 3: Handle correction based on number of incorrect players
    if len(incorrect_players) == 0:
        # All players are correct
        return {**correct_players, **{player: mogi_context[player] + score 
                                   for player, score in correct_players.items()}}
    
    elif len(incorrect_players) == 1:
        # Single incorrect player - use the remaining unused score
        incorrect_player = list(incorrect_players.keys())[0]
        if len(unused_scores) == 1:
            correct_players[incorrect_player] = unused_scores[0]
    
    else:
        # Multiple incorrect players - use distance-based assignment
        correct_players.update(assign_scores_by_distance(
            incorrect_players, unused_scores, mogi_context
        ))
    
    # Convert back to total scores (context + race score)
    result = {}
    for player in players:
        if player in correct_players:
            result[player] = mogi_context[player] + correct_players[player]
        else:
            result[player] = players[player]  # Fallback to original if something went wrong
    
    return result

def assign_scores_by_distance(incorrect_players, unused_scores, mogi_context):
    """Assign unused scores to incorrect players based on minimum total distance."""
    if len(incorrect_players) != len(unused_scores):
        # Can't fix if numbers don't match - return empty dict
        return {}
    
    players_list = list(incorrect_players.keys())
    detected_scores = [incorrect_players[player] for player in players_list]
    
    best_assignment = {}
    min_total_distance = float('inf')
    
    # Try all possible permutations of unused scores
    for score_permutation in permutations(unused_scores):
        total_distance = 0
        current_assignment = {}
        
        for i, player in enumerate(players_list):
            race_score = score_permutation[i]
            expected_total = mogi_context[player] + race_score
            detected_total = detected_scores[i]
            
            # Calculate distance between expected and detected total
            distance = abs(expected_total - detected_total)
            total_distance += distance
            current_assignment[player] = race_score
        
        # Keep track of the assignment with minimum total distance
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_assignment = current_assignment.copy()
    
    return best_assignment
                

def error_handling_names(players, mogi_context_names):
    """takes two dicts to fix name detection issues with easyOCR

    Args:
        players (Dict[str, int]): A dictionary of player names and their scores.
        mogi_context ([Dict[str, int]]): Contextual information of the mogi. (every player, and the score they have so far)
    returns:
        Dict[str, int]: A dictionary of player names and their corrected scores.
    """
    for player in list(players.keys()):  # Convert to list to avoid iteration issues
        if player not in mogi_context_names.keys():
            # try to find the closest name in mogi_context_names
            closest_name = None
            closest_distance = float('inf')
            for mogi_name in mogi_context_names.keys():
                distance = levenshtein_distance(player, mogi_name)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_name = mogi_name
            if closest_distance <= 2:  # threshold for considering a name as a match
                players[closest_name] = players.pop(player)
                print(f"Renamed player {player} to {closest_name}, (closest match)")
    return players

def levenshtein_distance(s1, s2):
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    if s1[0] == s2[0]:
        return levenshtein_distance(s1[1:], s2[1:])
    return 1 + min(
        levenshtein_distance(s1[1:], s2),    # deletion
        levenshtein_distance(s1, s2[1:]),    # insertion
        levenshtein_distance(s1[1:], s2[1:])  # substitution
    )
