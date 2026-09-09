"""Build the Earth-observation satellite and export it as a web-ready GLB.

Run headless:

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python tools/blender/build_satellite.py

Design intent: this must read as a SCIENCE instrument, not a spacecraft from a
film. That means plausible proportions (a bus roughly a metre across with solar
wings several times its length), a clearly identifiable nadir-pointing optical
barrel, and no weaponry, no glowing parts, no greebling for its own sake. It is
seen small and in silhouette against the Earth, so the silhouette is what is
being designed: two long wings, a compact body, one visible instrument.

Budget: the whole model stays a few thousand triangles. It is never more than a
couple of hundred pixels on screen, so detail beyond that is bytes the visitor
pays for and cannot see.
"""

import math
import os
import sys

import bpy

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend",
    "public",
    "landing",
    "satellite.glb",
)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def material(name: str, base, metallic: float, roughness: float, emission=None):
    """A plain PBR material. glTF carries base colour, metallic and roughness."""

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 1.0
    return mat


def add(obj, mat):
    obj.data.materials.append(mat)
    return obj


def build() -> None:
    clear_scene()

    # Materials: a warm foil-gold bus, dark blue-grey panels, plain metal booms.
    # Gold multi-layer insulation is what real Earth-observation buses actually
    # wear, and it also gives the one warm note against a blue planet.
    foil = material("Bus Foil", (0.62, 0.48, 0.20), 0.85, 0.32)
    panel = material("Solar Panel", (0.055, 0.075, 0.13), 0.55, 0.28)
    metal = material("Structure", (0.44, 0.45, 0.47), 0.9, 0.42)
    optic = material("Optic", (0.03, 0.04, 0.05), 0.2, 0.12)

    # --- bus: a compact box, ~1.1 x 1.0 x 1.4 m -------------------------- #
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    bus = bpy.context.object
    bus.name = "Bus"
    bus.scale = (0.55, 0.5, 0.7)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_flat()
    add(bus, foil)

    # A shallow bevel catches the rim light and stops the box reading as a
    # placeholder primitive. Small width - this is seen at a distance.
    bevel = bus.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.02
    bevel.segments = 2

    # --- instrument barrel: nadir-pointing, the reason it is up there ----- #
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.2, depth=0.5, vertices=24, location=(0, 0, -0.85)
    )
    barrel = bpy.context.object
    barrel.name = "Instrument"
    bpy.ops.object.shade_smooth()
    add(barrel, metal)

    # The aperture itself: dark, flat, unmistakably a lens looking down.
    bpy.ops.mesh.primitive_circle_add(
        radius=0.17, vertices=24, fill_type="NGON", location=(0, 0, -1.101)
    )
    lens = bpy.context.object
    lens.name = "Aperture"
    add(lens, optic)

    # --- solar wings: two, symmetric, on short booms ---------------------- #
    for side in (-1, 1):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.035, depth=0.5, vertices=8,
            location=(side * 0.78, 0, 0), rotation=(0, math.pi / 2, 0),
        )
        boom = bpy.context.object
        boom.name = f"Boom_{side}"
        bpy.ops.object.shade_smooth()
        add(boom, metal)

        # Two panels per wing. Real wings are segmented, and the seam reads at
        # distance as "solar array" rather than "grey rectangle".
        for seg in (0, 1):
            x = side * (1.28 + seg * 1.02)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0, 0))
            wing = bpy.context.object
            wing.name = f"Panel_{side}_{seg}"
            wing.scale = (0.5, 0.42, 0.012)
            bpy.ops.object.transform_apply(scale=True)
            add(wing, panel)

    # --- antenna: one thin dish, offset so the silhouette is not symmetric #
    # A perfectly symmetric object reads as a logo; one offset detail reads as
    # a built thing.
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.18, radius2=0.0, depth=0.12, vertices=20,
        location=(0.16, 0.34, 0.72), rotation=(math.pi * 0.82, 0, 0),
    )
    dish = bpy.context.object
    dish.name = "Antenna"
    bpy.ops.object.shade_smooth()
    add(dish, metal)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.012, depth=0.22, vertices=6, location=(0.16, 0.3, 0.62)
    )
    mast = bpy.context.object
    mast.name = "Mast"
    add(mast, metal)

    # --- join into one object: one draw call in the browser --------------- #
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = bus
    bpy.ops.object.join()
    bus.name = "Satellite"

    # Normalise scale so the web scene can size it in one place.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bus.location = (0, 0, 0)

    tris = sum(len(p.vertices) - 2 for p in bus.data.polygons)
    print(f"SATELLITE_TRIS {tris}")


def export() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=OUT,
        export_format="GLB",
        use_selection=True,
        export_apply=True,          # bake the bevel modifier
        export_draco_mesh_compression_enable=False,  # tiny mesh; Draco costs a decoder
        export_yup=True,
        export_cameras=False,
        export_lights=False,
    )
    print(f"SATELLITE_GLB {OUT} {os.path.getsize(OUT)}")


build()
export()
print("BUILD_DONE")
sys.exit(0)
