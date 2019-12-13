from scripts.renderer import visual_hull, absorption_only, emission_absorption

renderer_dict = {
    'visual_hull': visual_hull.VisualHullRenderer,
    'absorption_only': absorption_only.AbsorptionOnlyRenderer,
    'emission_absorption': emission_absorption.EmissionAbsorptionRenderer
}