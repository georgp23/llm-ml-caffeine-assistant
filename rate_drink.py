def rate_drink(goal, time_of_day, user_state, preferred_effects, avoid_effects, urgency, drink_profiles):

    score = 0.0

    # goal rules
    if goal == "sleep":
        score += 2 * drink_profiles.get("sleep", 0.0) # sleep is an important goal
    elif goal == "mood":
        score += drink_profiles.get("mood_boost", 0.0)
    elif goal == "energy":
        score += drink_profiles.get("energy", 0.0)
    elif goal == "balance":
        score += drink_profiles.get("balance", 0.0)
    elif goal == "focus":
        score += drink_profiles.get("clear_head")

    # user state rules
    if user_state == "tired":
        score += drink_profiles.get("energy", 0.0)
    elif user_state == "anxious":
        score -= drink_profiles.get("anxiety", 0.0)
        score -= drink_profiles.get("jitters", 0.0)
        score += drink_profiles.get("relax", 0.0)
    elif user_state == "wired":
        score += drink_profiles.get("balance", 0.0)
        score += drink_profiles.get("relax", 0.0)
    elif user_state == "foggy":
        score += drink_profiles.get("clear_head", 0.0)
    elif user_state == "stressed":
        score += drink_profiles.get("relax", 0.0)
        score -= drink_profiles.get("anxiety", 0.0)
    elif user_state == "rested":
        score += drink_profiles.get("balance", 0.0)
    
    # effect rules
    for effect in preferred_effects:
        # composite effect dealt with seperately 
        if effect == "calm_energy":
            score += 0.5 * (drink_profiles.get("energy") + drink_profiles.get("relax"))
        else: 
            score += drink_profiles.get(effect, 0.0)

    # avoid rules
    for avoid in avoid_effects:
        score -= drink_profiles.get(avoid, 0.0)
    
    # urgency rules using users urgency along with energy profile
    urgency_map = {
        "low": 0.2,
        "medium": 0.5,
        "high":  1.0
    }
    score += urgency_map.get(urgency, 0.5) * drink_profiles.get("energy", 0.0)

    # ensures score stays between -1 and 1
    score = max(min(score, 1.0), -1.0)

    return round(score, 3)

