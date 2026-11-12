"""Plain named materials; the viewport colour drives the Workbench renders, the node colour the glTF export."""
import bpy

PALETTE = {  # name: (RGBA, roughness)
    "Mur": ((0.85, 0.84, 0.80, 1.0), 0.9),
    "Sol": ((0.55, 0.45, 0.35, 1.0), 0.7),
    "Plafond": ((0.92, 0.92, 0.95, 1.0), 0.9),
    "Menuiserie": ((0.55, 0.38, 0.22, 1.0), 0.6),
    "Verre": ((0.60, 0.80, 0.90, 0.3), 0.05),
    "Moulure": ((0.90, 0.90, 0.88, 1.0), 0.8),
}


def ensure_materials() -> dict[str, bpy.types.Material]:
    out = {}
    for name, (color, roughness) in PALETTE.items():
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_fake_user = True  # an unused material would be dropped on save
        mat.diffuse_color = color
        mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Alpha"].default_value = color[3]
        if color[3] < 1.0:
            mat.surface_render_method = "BLENDED"
        out[name] = mat
    return out
