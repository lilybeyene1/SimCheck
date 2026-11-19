
"""
SIMCHECK Phase 2: Champion Data Collection

Retrieve official champion data from Riot Games
for our League-inspired combat environment.
"""

import json
import hashlib

from pathlib import Path
from urllib.request import urlopen


# STEP 1: Define our data source.
#
# Keep the patch fixed for reproducibility.

PATCH = "16.22.1"

CHAMPIONS = ["Vi", "Jinx"]

BASE_URL = (
    "https://ddragon.leagueoflegends.com"
    f"/cdn/{PATCH}/data/en_US/champion"
)


# STEP 2: Create a separate Phase 2 data folder.
#
# This keeps all Phase 1 files unchanged.

ROOT = Path(__file__).resolve().parent

DATA_DIRECTORY = ROOT / "champion_data"

DATA_DIRECTORY.mkdir(exist_ok=True)

champion_profiles = {}


# STEP 3: Retrieve the official champion data.

for champion in CHAMPIONS:

    url = f"{BASE_URL}/{champion}.json"

    with urlopen(url, timeout=30) as response:
        raw_data = response.read()

    data = json.loads(raw_data)

    assert data["version"] == PATCH

    # Preserve the original Riot data.

    raw_path = DATA_DIRECTORY / f"{champion}.json"

    raw_path.write_bytes(raw_data)

    character = data["data"][champion]

    stats = character["stats"]


    # STEP 4: Extract relevant base statistics.

    profile = {
        "name": character["name"],
        "health": stats["hp"],
        "health_growth": stats["hpperlevel"],
        "attack_damage": stats["attackdamage"],
        "attack_damage_growth": stats["attackdamageperlevel"],
        "armor": stats["armor"],
        "armor_growth": stats["armorperlevel"],
        "magic_resistance": stats["spellblock"],
        "movement_speed": stats["movespeed"],
        "attack_range": stats["attackrange"],
        "attack_speed": stats["attackspeed"],
        "abilities": {},
        "source_url": url,
        "sha256": hashlib.sha256(raw_data).hexdigest()
    }


    # STEP 5: Extract champion abilities.

    for key, spell in zip(
        ["Q", "W", "E", "R"],
        character["spells"]
    ):

        profile["abilities"][key] = {
            "name": spell["name"],
            "description": spell["description"],
            "cooldown": spell["cooldown"],
            "range": spell["range"],
            "effect": spell.get("effect"),
            "tooltip": spell.get("tooltip")
        }

    champion_profiles[champion] = profile


    # STEP 6: Display collected information.

    print("\n" + "=" * 55)
    print(champion.upper())
    print("=" * 55)

    for key, value in profile.items():

        if key not in [
            "abilities", "source_url", "sha256"
        ]:
            print(f"{key}: {value}")

    print("\nAbilities:")

    for key, ability in profile["abilities"].items():

        print(f"\n{key}: {ability['name']}")
        print("Cooldown:", ability["cooldown"])
        print("Range:", ability["range"])
        print("Description:", ability["description"])


# STEP 7: Save the processed champion profiles.

output = {
    "patch": PATCH,
    "champions": champion_profiles
}

with open(
    DATA_DIRECTORY / "champion_profiles.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(output, file, indent=4)


print("\n" + "=" * 55)
print("SIMCHECK PHASE 2")
print("=" * 55)

print("Data Dragon version:", PATCH)
print("Champions collected:", len(champion_profiles))
print("Source: Riot Games Data Dragon")
print("Ground-truth simulator queries used: 0")
print("Champion data collection complete.")
