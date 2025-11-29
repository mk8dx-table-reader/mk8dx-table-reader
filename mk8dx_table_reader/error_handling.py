from exceptions.DetectionException import DetectionError
import warnings
import traceback

scores_repartition_by_players = {
    6: [1,2,3,4,5,7],
    7: [1,2,3,4,5,7,9],
    8: [1,2,3,4,5,6,8,10],
    9: [1,2,3,4,5,6,7,9,11],
    10: [1,2,3,4,5,6,7,8,10,12],
    11: [1,2,3,4,5,6,7,8,9,11,13],
    12: [1,2,3,4,5,6,7,8,9,10,12,15]
}

def error_handling_scores(players):
    """Handles errors in score data, by ensuring the scores are valid depending on the number of players.

    """
    num_players = len(players)
    if num_players < 6 or num_players > 12:
        raise ValueError("Number of players must be between 6 and 12 for score validation.")
    
    valid_scores = sum(scores_repartition_by_players[num_players])
    for i in range(num_players):
        if players[i] == "Error" or players[i] == '':
            raise DetectionError("Score detection error for player {}".format(i+1))
        players[i] = int(players[i])
        
    players_scores_added_up = sum(players)
    test = players_scores_added_up % valid_scores
    if (test != 0):
        # pass
        warning_msg = "Scores do not add up correctly. They should sum to a multiple of {} Actual sum: {}".format(valid_scores, players_scores_added_up)
        warnings.warn(warning_msg, RuntimeWarning)

