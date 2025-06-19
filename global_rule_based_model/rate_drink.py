def rate_drink(goal, time_of_day, user_state, preferred_effects, avoid_effects, urgency, drink_name, drink_profiles):
    score = 0.0
    total_weight = 0.0  # Track the total weight of contributions

    # Get the attributes for the specific drink
    drink_attributes = drink_profiles.get(drink_name, {})

    # goal rules
    if goal == "sleep":
        weight = 2.0
        score += weight * drink_attributes.get("sleep", 0.0)
        total_weight += weight
    elif goal == "mood":
        weight = 1.2
        score += weight * drink_attributes.get("mood_boost", 0.0)
        total_weight += weight
    elif goal == "energy":
        weight = 1.0
        score += weight * drink_attributes.get("energy", 0.0)
        total_weight += weight
    elif goal == "balance":
        weight = 1.0
        score += weight * drink_attributes.get("balance", 0.0)
        total_weight += weight
    elif goal == "focus":
        weight = 1.0
        score += weight * drink_attributes.get("clear_head", 0.0)
        total_weight += weight

    # User state rules
    if user_state == "tired":
        weight = 1.0
        score += weight * drink_attributes.get("energy", 0.0)
        total_weight += weight
    elif user_state == "anxious":
        weight = 1.5
        score -= weight * drink_attributes.get("anxiety", 0.0)
        score -= weight * drink_attributes.get("jitters", 0.0)
        score += weight * drink_attributes.get("relax", 0.0)
        total_weight += weight
    elif user_state == "wired":
        weight = 1.0
        score += weight * drink_attributes.get("balance", 0.0)
        score += weight * drink_attributes.get("relax", 0.0)
        total_weight += weight
    elif user_state == "foggy":
        weight = 1.0
        score += weight * drink_attributes.get("clear_head", 0.0)
        total_weight += weight
    elif user_state == "stressed":
        weight = 1.5
        score += weight * drink_attributes.get("relax", 0.0)
        score -= weight * drink_attributes.get("anxiety", 0.0)
        total_weight += weight
    elif user_state == "rested":
        weight = 1.0
        score += weight * drink_attributes.get("balance", 0.0)
        total_weight += weight

    # Effect rules
    for effect in preferred_effects:
        if effect == "calm_energy":
            weight = 0.5
            score += weight * (drink_attributes.get("energy", 0.0) + drink_attributes.get("relax", 0.0))
            total_weight += weight
        else:
            weight = 1.0
            score += weight * drink_attributes.get(effect, 0.0)
            total_weight += weight

    # Avoid rules
    for avoid in avoid_effects:
        weight = 1.5  # Increased penalty weight to avoid mass ratings of 1.0
        score -= weight * drink_attributes.get(avoid, 0.0)
        total_weight += weight

    # urgency rules
    urgency_map = {
        "low": 0.2,
        "medium": 0.5,
        "high": 1.0
    }
    weight = urgency_map.get(urgency, 0.5)
    score += weight * drink_attributes.get("energy", 0.0)
    total_weight += weight

    # normalise the score by dividing by the total weight
    if total_weight > 0:
        score /= total_weight

    # ensures score stays between -1 and 1
    score = max(min(score, 1.0), -1.0)

    return round(score, 3)

