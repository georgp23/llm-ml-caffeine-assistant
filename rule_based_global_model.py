import random
import csv

# Define fixed categories
goal = ["energy", "focus", "relax", "sleep", "mood", "balance", "no_applicable_categories"]
time_of_day = ["morning", "afternoon", "evening", "night", "unknown"]
days_of_week = ["weekday", "weekend", "unknown"]
preferred_effects = ["calm_energy", "sustained_focus", "mood_boost", "energy", "clear_head", "gentle_stimulation", "no_applicable_categories"]
avoid_effects = ["jitters", "crash", "anxiety", "insomnia", "stomach_upset", "no_applicable_categories"]
urgency = ["low", "medium", "high", "no_applicable_categories"]
user_state = ["tired", "anxious", "wired", "foggy", "stressed", "rested", "no_applicable_categories"]
drink = ["green_tea", "matcha", "black_tea", "yerba_mate", "brewed_coffee", "cold_brew", "espresso", "decaf_coffee", "energy_drink"]

def rate_drink(goal, time_of_day, user_state, preferred_effects, avoid_effects, urgency, drink_profile):

    score = 0.0

    # goal rules
    if goal == "sleep":
        score += 2 * drink_profile.get("sleep", 0.0) # sleep is an important goal
    elif goal == "mood":
        score += drink_profile.get("mood_boost", 0.0)
    elif goal == "energy":
        score += drink_profile.get("energy", 0.0)
    elif goal == "balance":
        score += drink_profile.get("balance", 0.0)
    elif goal == "focus":
        score += drink_profile.get("clear_head")

    # user state rules
    if user_state == "tired":
        score += drink_profile.get("energy", 0.0)
    elif user_state == "anxious":
        score -= drink_profile.get("anxiety", 0.0)
        score -= drink_profile.get("jitters", 0.0)
        score += drink_profile.get("relax", 0.0)
    elif user_state == "wired":
        score += drink_profile.get("balance", 0.0)
        score += drink_profile.get("relax", 0.0)
    elif user_state == "foggy":
        score += drink_profile.get("clear_head", 0.0)
    elif user_state == "stressed":
        score += drink_profile.get("relax", 0.0)
        score -= drink_profile.get("anxiety", 0.0)
    elif user_state == "rested":
        score += drink_profile.get("balance", 0.0)
    
    # effect rules
    for effect in preferred_effects:
        # composite effect dealt with seperately 
        if effect == "calm_energy":
            score += 0.5 * (drink_profile.get("energy") + drink_profile.get("relax"))
        else: 
            score += drink_profile.get(effect, 0.0)

    # avoid rules
    for avoid in avoid_effects:
        score -= drink_profile.get(avoid, 0.0)
    
    # urgency rules using users urgency along with energy profile
    urgency_map = {
        "low": 0.2,
        "medium": 0.5,
        "high":  1.0
    }
    score += urgency_map.get(urgency, 0.5) * drink_profile.get("energy", 0.0)

    # ensures score stays between -1 and 1
    score = max(min(score, 1.0), -1.0)

    return round(score, 3)

