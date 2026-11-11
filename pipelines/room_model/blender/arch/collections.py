"""The collections of the scene, in the order the spec lists them."""
import bpy

NAMES = ("Coque", "Menuiseries", "Moulures", "Cotes", "À vérifier", "Scan")
EXPORTED = ("Coque", "Menuiseries", "Moulures", "À vérifier")  # the static GLB; Cotes (dimensions) and Scan stay out


def create(scene: bpy.types.Scene) -> dict[str, bpy.types.Collection]:
    """Fresh collections linked under the scene collection; the Scan one is hidden in viewport and render."""
    out = {}
    for name in NAMES:
        coll = bpy.data.collections.new(name)
        scene.collection.children.link(coll)
        out[name] = coll
    out["Scan"].hide_viewport = out["Scan"].hide_render = True
    return out
